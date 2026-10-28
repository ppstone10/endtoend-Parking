"""120秒400/410同剂量学习机制对照；无旧回放，非候选。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration import first_stop
from experiments.behavior_exploration_data import run_case,save_record,evaluate_cases
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.recovery_execution import ContinuationContext,record_arrays
from experiments.s6_recovery_learning import batch_identity
from experiments.s6_replanning_chain import pair_schedule,full_reference_joint_loss,reference_audit
from experiments.stop_endpoint_intervals import first_stop_intervals
from interfaces import Trajectory
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s6_replanning_chain import BIAS,PREVIOUS,QUERIES,exact_arrays
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


DATA=ROOT/'runs/diagnostics/s6-replanning-chain-data-preflight-20261022-v2'
HASHES={'config.json':'9e3eda4f68eca14d8106551e2edcd5a2471cdccdddd45b9bf395e4ae3a0b2584',
        'report.json':'e496231d81bc0dfdce68378a4096aa33b1ef9ba012e59bd769dec621ef3dad9d'}
FITTED=ROOT/'runs/diagnostics/s6-chain-learning-preflight-20261022-v1'
FITTED_HASHES={'config.json':'16b5f5f0c494e194e786cd00a31b41e1a315b8084d8b67847a01e9b7c70517f3',
    'report.json':'cb09c763f8f1943fa08d7fba595e43897b4c3dd639b98880f20522a9cc7e16dd',
    'state.pt':'e3cc3194adc991bbed0263d2c227e5cb5aeaf0e9e862c8cb671e3390d4d605b1'}


def prepare_context(budget):
    torch.set_num_threads(1); verify_files({str(DATA/k):v for k,v in HASHES.items()})
    config=json.loads((DATA/'config.json').read_text(encoding='utf-8'))
    report=json.loads((DATA/'report.json').read_text(encoding='utf-8'))
    if report['status']!='completed' or not report['ready_for_local_learning_probe']:
        raise ValueError('410真实查询未过完整执行/导出/新增门禁')
    config=deepcopy(config)
    config['frozen_files'].update({str(DATA/k):v for k,v in HASHES.items()})
    config['frozen_files'].update({str(artifact_path(DATA,k)):v for k,v in report['artifacts_sha256'].items()})
    config['code'][str(Path(__file__).resolve())]=file_digest(__file__)
    verify_config(config,budget)
    samples=torch.load(DATA/'queries.pt',map_location='cpu',weights_only=False)['samples']
    if len(samples)!=2 or samples[0]['query_step']!=400 or samples[1]['query_step']!=410:
        raise ValueError('不是固定400/410两个真实状态')
    identities=batch_identity(samples)
    if identities[0]!=config['sample_identity'] or identities[1]['input_sha256']!=report['query_410']['input_sha256'] or identities[1]['label_sha256']!=report['query_410']['label_sha256']:
        raise ValueError('原查询身份/标签漂移')
    config.update(kind='s6_chain_learning_preflight_v1',budget_seconds=120,steps=120,start_step=240,end_step=360,
        comparison='single400 cached120 versus alternating400/410 each60; same full-reference joint objective',
        local_only=True,old_replay_present=False,planned_batch_identities=[batch_identity([s]) for s in pair_schedule(*samples)])
    parent=load_model_checkpoint(INPUTS['R0']).model
    model,state=load_candidate(config['parent_checkpoints']['20261020'],parent)
    budget.check(); return dict(config=config,samples=samples,parent=parent,model=model,optimizer_state=state)


def precheck():
    context=prepare_context(PreflightBudget(120))
    print('STRICT CHAIN INPUTS/120 UPDATE PLAN VERIFIED; NO TRAINING OR ROLLOUT',flush=True)
    return context


def fit_pair(parent,state,samples,planned,budget):
    if not state['state'] or {int(s['step']) for s in state['state'].values()}!={240}:
        raise ValueError('非原Adam240起点')
    torch.manual_seed(20261020); model=deepcopy(parent).cpu().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5); optimizer.load_state_dict(deepcopy(state))
    if any(g['lr']!=1e-5 for g in optimizer.param_groups): raise ValueError('学习率漂移')
    schedule=pair_schedule(*samples); history=[]; presented=[]
    if len(planned)!=120: raise ValueError('缺固定120批次身份')
    for i,sample in enumerate(schedule):
        budget.check(); identity=batch_identity([sample])
        if identity!=planned[i]: raise ValueError('梯度前输入/标签身份漂移')
        optimizer.zero_grad(set_to_none=True)
        out=model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
        loss=full_reference_joint_loss(out.points,out.stop_logits,sample)
        if not torch.isfinite(loss): raise ValueError('非有限loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        history.append(float(loss.detach())); presented.append(identity)
    if {int(s['step']) for s in optimizer.state.values()}!={360}: raise ValueError('非原固定Adam360终态')
    model.eval(); return model,optimizer,history,presented


@torch.no_grad()
def query_metrics(model,sample):
    out=model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
    length=int(sample['mask'].sum()); diff=out.points[0,:length]-sample['points'][:length]
    yaw=torch.atan2(diff[:,2].sin(),diff[:,2].cos())
    return dict(step=sample['query_step'],predicted_length=first_stop(out.stop_logits[0].sigmoid(),model.stop_threshold),
        target_length=length,xy_rmse=float(diff[:,:2].square().sum(-1).mean().sqrt()),
        yaw_rmse_deg=float(yaw.square().mean().sqrt()*180/np.pi))


def local_evaluation(case,model,contexts,budget,output,prefix):
    rolling,engine,_=run_case(case,model,budget,context=contexts[400])
    reference=Trajectory(rolling.record.refs[0].copy(),model.dt)
    fixed,fixed_engine,_=run_case(case,None,budget,context=contexts[400],trajectory=reference)
    a,u=record_arrays(rolling); b,v=record_arrays(fixed); n=min(10,len(a),len(b))
    if n<1 or not np.array_equal(a[:n],b[:n]) or not np.array_equal(u[:n],v[:n]):
        raise ValueError('原首次参考同首控制窗口未对齐')
    at410,engine410,_=run_case(case,model,budget,context=contexts[410])
    save_record(output,f'{prefix}-400-rolling',rolling,contexts[400],first_reference=reference.points)
    save_record(output,f'{prefix}-400-fixed',fixed,contexts[400],reference=reference.points)
    save_record(output,f'{prefix}-410-rolling',at410,contexts[410])
    return dict(rolling400=result_row(case,rolling,engine),fixed400=result_row(case,fixed,fixed_engine),
        rolling410=result_row(case,at410,engine410),exact_first_window_steps=n,
        first_reference=reference_audit(reference.points,contexts[400].state,case,budget.check))


def run(output,*,complete_evaluation=False):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); budget=PreflightBudget(75 if complete_evaluation else 120)
    report=dict(status='running',stage='identity',purpose='local_chain_mechanism_not_candidate',
        old_replay_present=False,allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-started; atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget); config=context['config']; samples=context['samples']
            if complete_evaluation:
                verify_files({str(FITTED/k):v for k,v in FITTED_HASHES.items()})
                failed=json.loads((FITTED/'report.json').read_text(encoding='utf-8'))
                prior=json.loads((FITTED/'config.json').read_text(encoding='utf-8'))
                current=json.loads(json.dumps(config)); cli=str(Path(__file__).resolve())
                prior['code'].pop(cli); current['code'].pop(cli)
                if current!=prior or failed['status']!='failed' or failed['stage']!='local_execution' or failed['presented_batch_identities']!=config['planned_batch_identities']:
                    raise ValueError('非同一已完成拟合的评测故障，不允许复用')
                checkpoint=torch.load(FITTED/'state.pt',map_location='cpu',weights_only=False)
                if checkpoint['step']!=360 or {int(s['step']) for s in checkpoint['optimizer_state']['state'].values()}!={360} or len(failed['history'])!=120:
                    raise ValueError('拟合未完整120更新，不允许补评测')
                config['frozen_files'].update({str(FITTED/k):v for k,v in FITTED_HASHES.items()})
                config.update(budget_seconds=75,completion_only=True,original_fit=str(FITTED))
            atomic_write_json(output/'config.json',config); report['config_sha256']=file_digest(output/'config.json')
            previous=json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
            bias_control=json.loads((BIAS/'report.json').read_text(encoding='utf-8'))
            report.update(cached_single_control=previous['completed_arms']['boundary_terminal'],cached_bias_control=bias_control); save()
            if complete_evaluation:
                print('EVALUATE SAME FROZEN ADAM360; ZERO TRAINING UPDATES',flush=True)
                model,_=load_candidate(FITTED/'state.pt',context['parent'])
                history=failed['history']; presented=failed['presented_batch_identities']
                torch.save(checkpoint,output/'state.pt')
                report.update(training_updates_this_run=0,total_updates=120,reused_fitted_checkpoint_sha256=FITTED_HASHES['state.pt'])
            else:
                print('FIT alternating400/410 updates=120 NOT A CANDIDATE',flush=True)
                model,optimizer,history,presented=fit_pair(context['model'],context['optimizer_state'],samples,config['planned_batch_identities'],budget)
                torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),step=360,
                    threshold=model.stop_threshold,purpose='local_chain_no_retention_non_deployable'),output/'state.pt')
                report.update(training_updates_this_run=120,total_updates=120)
            report.update(stage='local_execution',history=history,presented_batch_identities=presented,
                          query_metrics=[query_metrics(model,s) for s in samples]); save()
            data=DatasetGenerator.load(INPUTS['train']); manifest=matrix.load_dataset_manifest(INPUTS['train'])
            vehicle=VehicleConfig(**manifest['vehicle_model'])
            cases={i:build_case(data,manifest,i,vehicle) for i in sorted(set(config['probe_indices']+config['fit_indices']))}
            case=cases[2149]; original,engine,_=run_case(case,context['model'],budget)
            exact_arrays(original,QUERIES/'source-20261020-2149/learner.npz')
            at400=next(c for c in engine.contexts if c.step==400)
            prototype,_=load_candidate(BIAS/'state.pt',context['parent'])
            original_proto,proto_engine,_=run_case(case,prototype,budget,context=at400)
            exact_arrays(original_proto,DATA/'prototype-rolling.npz')
            c=next(c for c in proto_engine.contexts if c.step==10)
            contexts={400:at400,410:ContinuationContext(410,c.state,c.mpc,c.pipeline)}
            report['native']=local_evaluation(case,model,contexts,budget,output,'native')
            if complete_evaluation:
                for name in ('native-400-rolling.npz','native-400-fixed.npz','native-410-rolling.npz'):
                    with np.load(output/name) as actual,np.load(FITTED/name) as expected:
                        if any(not np.array_equal(actual[k],expected[k]) for k in ('states','commands')):
                            raise ValueError('原已执行局部分支未精确复现')
                report['original_three_local_executions_exact']=True
            report['probe']=evaluate_cases(config['probe_indices'],cases,model,budget,output,'probe')
            report['fit']=evaluate_cases(config['fit_indices'],cases,model,budget,output,'fit'); save()
            with torch.no_grad(): original_prediction=model.forward_with_stop(samples[0]['bev'][None],samples[0]['goal'][None],samples[0]['state'][None])
            intervals=first_stop_intervals(original_prediction.stop_logits,torch.ones_like(original_prediction.stop_logits),threshold=model.stop_threshold)
            if bool(intervals['nonempty'][0,18]):
                lower=float(intervals['lower'][0,18]); upper=float(intervals['upper'][0,18])
                if not np.isfinite(lower+upper) or upper<=lower: raise ValueError('无有限唯一中点，不扫描偏置')
                calibrated=deepcopy(model)
                with torch.no_grad(): calibrated.decoder.stop_head.bias.add_((lower+upper)/2)
                with torch.no_grad(): updated=calibrated.forward_with_stop(samples[0]['bev'][None],samples[0]['goal'][None],samples[0]['state'][None])
                if not torch.equal(original_prediction.points,updated.points) or first_stop(updated.stop_logits[0].sigmoid(),model.stop_threshold)!=19:
                    raise ValueError('偏置隔离改变了点值或未选19点')
                torch.save(dict(model_state=calibrated.state_dict(),optimizer_state=None,step=360,
                    threshold=model.stop_threshold,resumable=False,purpose='single_query_bias_non_deployable'),output/'bias-state.pt')
                report['bias_isolation']=dict(interval=[lower,upper],offset=(lower+upper)/2,
                    equivalent_global_logit_cutoff_change=True,all320_points_unchanged=True,
                    execution=local_evaluation(case,calibrated,contexts,budget,output,'bias'))
                report['bias_probe']=evaluate_cases(config['probe_indices'],cases,calibrated,budget,output,'bias-probe')
                report['bias_fit']=evaluate_cases(config['fit_indices'],cases,calibrated,budget,output,'bias-fit')
            else:
                report['bias_isolation']=dict(skipped=True,reason='no_positive_bias_interval_for_teacher_index18_do_not_scan')
            verify_config(config,budget); budget.check()
            native=report['native']; strict=lambda p:all(p[k]['reached_with_margin'] for k in ('rolling400','fixed400','rolling410'))
            report.update(status='completed',stage='decision',local_native_chain_signal=strict(native),
                conditioned_bias_signal=bool(not report['bias_isolation'].get('skipped') and strict(report['bias_isolation']['execution'])),
                decision='local_chain_mechanism_only_not_global_preservation',
                artifacts_sha256={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*') if p.is_file()
                    and p!=output/'report.json' and p.name not in ('run.lock','console.log')})
            save(); print('FINISHED',report['elapsed_seconds'],report['local_native_chain_signal'],report['conditioned_bias_signal'],flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',error=str(exc),
                          decision='incomplete_do_not_infer_chain_learning'); save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path); parser.add_argument('--precheck',action='store_true')
    parser.add_argument('--complete-evaluation',action='store_true',help='只复用固定故障前已完成的120步检查点，剩余75秒，不更新')
    args=parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output')
    else: run(args.output,complete_evaluation=args.complete_evaluation)
