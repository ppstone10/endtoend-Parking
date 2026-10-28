"""180秒精确重放历史Adam更新并拆解终段停止决策传输；不是新候选。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from torch.nn import functional as F

from experiments.behavior_exploration import first_stop
from experiments.behavior_preservation import behavior_loss
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import exploration_lock,artifact_path
from experiments.retention_sampling import TRAINING_SEEDS
from experiments.s6_recovery_learning import batch_identity
from experiments.s6_terminal_objective import objective_loss,terminal_query
from experiments.s6_update_transport import gradient_list,vector_dot,adam_counterfactual,state_equal
from model.network import variable_loss_fn,terminal_pose_loss
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s6_terminal_objective import prepare_context as prepare_objective_context
from scripts.prepare_recovery_balance_ab import file_digest
from training.reporting import atomic_write_json


PREVIOUS=ROOT/'runs/diagnostics/s6-terminal-objective-preflight-20261022-v1'
MASK=ROOT/'runs/diagnostics/s6-terminal-consumption-preflight-20261022-v1'
HASHES={'config.json':'f35d324aef1a82044adcaa371bf8ae9a839b689d7df5d065179c4d5e7305bd1a',
        'report.json':'8ca5cee925399fefc0156134870b86f1c3b53c59d1134a595baf22bf25edce8e'}
ARMS=('terminal_mask','boundary_only','boundary_terminal')


def prepare_context(budget):
    verify_files({str(PREVIOUS/k):v for k,v in HASHES.items()})
    old=json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'))
    report=json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    if report['status']!='completed' or report['allowed_to_prepare_training_plan']:
        raise ValueError('不是完整未放行的终点目标对照')
    context=prepare_objective_context(budget)
    if json.loads(json.dumps(context['config']))!=old:
        raise ValueError('原配置/排程未完整复现')
    config=deepcopy(old)
    config['frozen_files'].update({str(PREVIOUS/k):v for k,v in HASHES.items()})
    config['frozen_files'].update({str(artifact_path(PREVIOUS,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s6_update_transport
    for path in (Path(s6_update_transport.__file__).resolve(),Path(__file__).resolve()):
        config['code'][str(path)]=file_digest(path)
    config.update(kind='s6_update_transport_audit_v1',budget_seconds=180,
                  audit_arms=list(ARMS),new_candidate=False,automatic_deployment=False)
    points={}
    for seed in TRAINING_SEEDS:
        points[seed]={'terminal_mask':context['terminal_report']['completed_seeds'][str(seed)]}
        for arm in ARMS[1:]: points[seed][arm]=report['completed_arms'][arm][str(seed)]
        identities=[batch_identity(b) for b in context['plans']['boundary_terminal'][seed]]
        if any(p['presented_batch_identities']!=identities for p in points[seed].values()):
            raise ValueError('实际历史梯度身份不一致')
    verify_config(config,budget)
    context.update(config=config,cached_points=points)
    return context


@torch.no_grad()
def query_metrics(model,sample):
    out=model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
    logit=float(out.stop_logits[0,18]); threshold=float(torch.logit(torch.tensor(model.stop_threshold,dtype=torch.float64)))
    return dict(logit18=logit,probability18=float(out.stop_logits[0,18].sigmoid()),
        length=first_stop(out.stop_logits[0].sigmoid(),model.stop_threshold),
        gap_to_threshold=threshold-logit)


def loss_for(out,batch,arm):
    if arm!='terminal_mask': return objective_loss(out.points,out.stop_logits,batch,arm)
    stack=lambda k:torch.stack([s[k] for s in batch])
    return behavior_loss(out.points,out.stop_logits,stack('points'),stack('stop'),stack('mask'),[s['role'] for s in batch])


def step(model,optimizer,batch,arm,budget):
    budget.check(); stack=lambda k:torch.stack([s[k] for s in batch])
    optimizer.zero_grad(set_to_none=True)
    out=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
    loss=loss_for(out,batch,arm)
    if not torch.isfinite(loss): raise ValueError('非有限历史loss')
    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()),float(norm)


def active_step(model,optimizer,batch,arm,budget):
    budget.check(); sample=batch[2]
    if not terminal_query(sample): raise ValueError('不是完整终段查询')
    stack=lambda k:torch.stack([s[k] for s in batch]); parameters=list(model.parameters())
    names=[name for name,_ in model.named_parameters()]
    before=query_metrics(model,sample)
    out=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
    target,stop,mask=stack('points'),stack('stop'),stack('mask'); roles=[s['role'] for s in batch]
    base=behavior_loss(out.points,out.stop_logits,target,stop,mask,roles)
    probability=stop.detach(); entropy=-(torch.special.xlogy(probability,probability)+torch.special.xlogy(1-probability,1-probability))
    per_stop=.2*(F.binary_cross_entropy_with_logits(out.stop_logits,probability,reduction='none')-entropy).mean(1)
    stop_term=torch.stack([per_stop[torch.tensor([r==role for r in roles])].mean() for role in sorted(set(roles))]).mean()
    pose_term=base-stop_term
    terminal=out.points.sum()*0
    if arm!='terminal_mask':
        replacement=variable_loss_fn(target[2:3],out.stop_logits[2:3],target[2:3],mask[2:3],stop_weight=.2,
            balance_stop=True,stop_target_mode='terminal',yaw_error_mode='wrapped')
        stop_term=stop_term+(replacement-.2*F.binary_cross_entropy_with_logits(out.stop_logits[2],stop[2]))/4
    if arm=='boundary_terminal':
        terminal=terminal_pose_loss(out.points[2:3],target[2:3],mask[2:3],yaw_weight=10,yaw_error_mode='wrapped')/4
    loss=loss_for(out,batch,arm)
    output_gradient=float(torch.autograd.grad(loss,out.stop_logits,retain_graph=True)[0][2,18])
    gradients=gradient_list(loss,parameters)
    components={name:gradient_list(value,parameters) for name,value in
                dict(pose=pose_term,stop=stop_term,terminal=terminal).items()}
    reconstruction=[sum(parts) for parts in zip(*components.values(),strict=True)]
    error=[a-b for a,b in zip(reconstruction,gradients,strict=True)]
    if vector_dot(error,error)>1e-8*max(1,vector_dot(gradients,gradients)):
        raise ValueError('梯度分解未复现原实际目标')
    query=model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
    jacobian=gradient_list(query.stop_logits[0,18],parameters)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    raw=[p.grad.detach().clone() for p in parameters]
    if any(not torch.equal(g,h) for g,h in zip(raw,gradients,strict=True)):
        raise ValueError('autograd与原backward梯度不一致')
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
    clipped=[p.grad.detach().clone() for p in parameters]
    initial=deepcopy(model); initial_state=deepcopy(optimizer.state_dict())
    before_parameters=[p.detach().clone() for p in parameters]
    probes={}
    for mode in ('actual','zero_current','zero_first_moment'):
        budget.check()
        candidate,candidate_state=adam_counterfactual(initial,initial_state,clipped,mode)
        delta=[p.detach()-a for p,a in zip(candidate.parameters(),before_parameters,strict=True)]
        probes[mode]=dict(after=query_metrics(candidate,sample),parameter_step_norm=vector_dot(delta,delta)**.5,
                         linear_logit_change=vector_dot(jacobian,delta))
        if mode=='actual': expected_model=deepcopy(candidate.state_dict()); expected_optimizer=candidate_state
    optimizer.step()
    if not state_equal(expected_model,model.state_dict()) or not state_equal(expected_optimizer,optimizer.state_dict()):
        raise ValueError('有限Adam副本与实际更新不一致')
    groups={}
    for group in ('stop_head','point_head','shared_decoder','encoder_condition'):
        use=[i for i,n in enumerate(names) if
            (group=='stop_head' and n.startswith('decoder.stop_head.')) or
            (group=='point_head' and n.startswith('decoder.point_head.')) or
            (group=='shared_decoder' and n.startswith('decoder.') and not n.startswith(('decoder.stop_head.','decoder.point_head.'))) or
            (group=='encoder_condition' and not n.startswith('decoder.'))]
        groups[group]=dict(gradient_norm=sum(float(gradients[i].double().square().sum()) for i in use)**.5,
                          negative_gradient_logit_derivative=-sum(float((jacobian[i].double()*gradients[i].double()).sum()) for i in use))
    return dict(before=before,after=query_metrics(model,sample),loss=float(loss.detach()),raw_gradient_norm=norm,
        clip_scale=min(1,1/(norm+1e-6)),query_logit18_loss_gradient=output_gradient,
        components={name:dict(norm=vector_dot(g,g)**.5,
        negative_gradient_logit_derivative=-vector_dot(jacobian,g)) for name,g in components.items()},
        parameter_groups=groups,finite_adam_counterfactuals=probes,
        decomposition_error_norm=vector_dot(error,error)**.5)


def precheck():
    context=prepare_context(PreflightBudget(180))
    print('HISTORICAL IDENTITIES VERIFIED; NO GRADIENT, UPDATE OR ROLLOUT',flush=True)
    return context


def run(output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); budget=PreflightBudget(180)
    report=dict(status='running',stage='identity',new_candidate=False,automatic_deployment=False,
        main_protocol_passed=False,allowed_to_prepare_training_plan=False,replays={})
    def save():
        report['elapsed_seconds']=time.perf_counter()-started; atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget); config=context['config']
            atomic_write_json(output/'config.json',config); report['config_sha256']=file_digest(output/'config.json'); save()
            for seed in TRAINING_SEEDS:
                torch.manual_seed(seed); plan=context['plans']['boundary_terminal'][seed]
                model,state=load_candidate(config['parent_checkpoints'][str(seed)],context['parent'])
                model.train(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-5); optimizer.load_state_dict(deepcopy(state))
                if {int(s['step']) for s in optimizer.state.values()}!={240}: raise ValueError('非原Adam240')
                sample=plan[16][2]; initial=query_metrics(model,sample); prefix=[]
                expected=config['planned_batch_identities']['boundary_terminal'][str(seed)]
                for i,batch in enumerate(plan[:16]):
                    if batch_identity(batch)!=expected[i]: raise ValueError('前缀梯度身份漂移')
                    prefix.append(step(model,optimizer,batch,'terminal_mask',budget)[0])
                prefix_model=deepcopy(model); prefix_state=deepcopy(optimizer.state_dict()); rng=torch.get_rng_state()
                for arm in ARMS:
                    budget.check(); torch.set_rng_state(rng); model=deepcopy(prefix_model)
                    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5); optimizer.load_state_dict(deepcopy(prefix_state))
                    print(f'REPLAY {arm} order={seed}',flush=True)
                    details=active_step(model,optimizer,plan[16],arm,budget)
                    history=prefix+[details['loss']]; later=[]
                    for i in range(17,24):
                        if batch_identity(plan[i])!=expected[i]: raise ValueError('后缀梯度身份漂移')
                        value,_=step(model,optimizer,plan[i],arm,budget); history.append(value)
                        later.append(dict(batch_index=i,**query_metrics(model,sample)))
                    cached=torch.load(context['cached_points'][seed][arm]['checkpoint'],map_location='cpu',weights_only=False)
                    if not state_equal(model.state_dict(),cached['model_state']) or not state_equal(optimizer.state_dict(),cached['optimizer_state']):
                        raise ValueError('原最终模型/Adam未逐张量复现，不解释审计')
                    if history!=context['cached_points'][seed][arm]['loss_history']:
                        raise ValueError('原24损失序列未精确复现')
                    report['replays'].setdefault(seed,{})[arm]=dict(initial=initial,active_step=details,later_batches=later,
                        final=query_metrics(model,sample),exact_model_and_adam=True,exact_loss_history=True,
                        actual_batch_identities=expected,final_step=264)
                    save()
            verify_config(config,budget); budget.check()
            report.update(status='verified',stage='decision',frozen_code_files=len(config['code']),
                decision='historical_update_transport_verified_not_closed_loop_benefit')
            save(); print('VERIFIED',report['elapsed_seconds'],flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',error=str(exc),
                          decision='incomplete_do_not_infer_update_cause'); save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path); parser.add_argument('--precheck',action='store_true')
    args=parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output')
    else: run(args.output)
