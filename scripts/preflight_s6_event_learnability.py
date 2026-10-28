"""150秒同120步目标学习性隔离：两个非候选，不含旧回放或自动晋升。"""
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
from experiments.behavior_exploration_data import run_case,save_record,evaluate_cases
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.recovery_execution import record_arrays
from experiments.s6_recovery_learning import batch_identity
from experiments.s6_terminal_objective import ARMS,objective_loss,terminal_query
from interfaces import Trajectory
from runtime.sources import NetworkSource
from scripts.audit_s6_update_transport import query_metrics
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


AUDIT=ROOT/'runs/diagnostics/s6-update-transport-audit-20261022-v1'
CONTROL=ROOT/'runs/diagnostics/s6-terminal-consumption-preflight-20261022-v1/single-query-diagnostic'
QUERIES=ROOT/'runs/diagnostics/learner-failure-teacher-preflight-20261022-v1'
TEACHER=ROOT/'runs/diagnostics/s6-recovery-learning-preflight-20261022-v1/independent-transfer/terminal-teacher-fixed.npz'
HASHES={'config.json':'15278a221ccf39db4ba02c21a95fbf736b43675b8aea897bb0ce5ae13b4da48e',
        'report.json':'49b42d11c5da44c6dab8ec501cf8eebfc00ff305c82370009d6d5fca8f477d8a'}


def prepare_context(budget):
    torch.set_num_threads(1)
    verify_files({str(AUDIT/k):v for k,v in HASHES.items()})
    verify_files({str(CONTROL/'report.json'):'e7756b14dc3c33c5a00c7d810b78431b8a52c629c9e66dd1fb5b94ce2755d6cc'})
    config=json.loads((AUDIT/'config.json').read_text(encoding='utf-8'))
    audit=json.loads((AUDIT/'report.json').read_text(encoding='utf-8'))
    control=json.loads((CONTROL/'report.json').read_text(encoding='utf-8'))
    if audit['status']!='verified' or control['status']!='completed' or control['allowed_to_prepare_training_plan']:
        raise ValueError('需要完整优化传输审计和非候选原120步控制')
    for values in audit['replays'].values():
        for point in values.values():
            if not point['exact_model_and_adam'] or not point['exact_loss_history']:
                raise ValueError('历史重放未精确复现')
    config=deepcopy(config)
    config['frozen_files'].update({str(AUDIT/k):v for k,v in HASHES.items()})
    config['frozen_files'][str(CONTROL/'report.json')]=file_digest(CONTROL/'report.json')
    config['frozen_files'].update({str(artifact_path(CONTROL,k)):v for k,v in control['artifacts_sha256'].items()})
    config['code'][str(Path(__file__).resolve())]=file_digest(__file__)
    verify_config(config,budget)
    sample=next(s for s in torch.load(QUERIES/'queries.pt',map_location='cpu',weights_only=False)['samples'] if s['query_step']==400)
    sample=dict(sample,mask=(torch.arange(320)<19).float())
    identity=batch_identity([sample])[0]
    if (not terminal_query(sample) or identity!=control['sample_identity'] or
            identity!=config['planned_batch_identities']['boundary_terminal']['20261020'][16][2]):
        raise ValueError('非原同输入/19点标签查询')
    old_checkpoint=torch.load(CONTROL/'state.pt',map_location='cpu',weights_only=False)
    if old_checkpoint['step']!=360 or {int(s['step']) for s in old_checkpoint['optimizer_state']['state'].values()}!={360}:
        raise ValueError('原累计loss控制不是固定120更新')
    config.update(kind='s6_event_learnability_preflight_v1',budget_seconds=150,arms=list(ARMS),
        unique_inputs=1,presentations=120,start_step=240,end_step=360,old_replay_present=False,
        purpose='single_query_objective_mechanism_not_candidate',sample_identity=identity,
        comparison='same sample/Adam240/lr/clip/120 updates; cumulative cached versus boundary and joint objectives')
    parent=load_model_checkpoint(INPUTS['R0']).model
    model,optimizer_state=load_candidate(config['parent_checkpoints']['20261020'],parent)
    budget.check()
    return dict(config=config,sample=sample,parent=parent,model=model,optimizer_state=optimizer_state,control=control)


def fit_model(parent,optimizer_state,sample,arm,budget,identity):
    if arm not in ARMS or not terminal_query(sample): raise ValueError('非预定目标/完整查询')
    if not optimizer_state['state'] or {int(s['step']) for s in optimizer_state['state'].values()}!={240}:
        raise ValueError('非原Adam240起点')
    torch.manual_seed(20261020); model=deepcopy(parent).cpu().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5); optimizer.load_state_dict(deepcopy(optimizer_state))
    if any(group['lr']!=1e-5 for group in optimizer.param_groups): raise ValueError('学习率漂移')
    history=[]; norms=[]
    for _ in range(120):
        budget.check()
        if batch_identity([sample])[0]!=identity: raise ValueError('唯一查询输入/标签身份漂移')
        optimizer.zero_grad(set_to_none=True)
        out=model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
        loss=objective_loss(out.points,out.stop_logits,[sample],arm)
        if not torch.isfinite(loss): raise ValueError('非有限loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step(); history.append(float(loss.detach())); norms.append(float(norm))
    if len(history)!=120 or {int(s['step']) for s in optimizer.state.values()}!={360}:
        raise ValueError('固定120更新/Adam360不符')
    model.eval(); return model,optimizer,history,norms


@torch.no_grad()
def metrics(model,sample):
    out=model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
    diff=out.points[0,:19]-sample['points'][:19]
    yaw=torch.atan2(diff[:,2].sin(),diff[:,2].cos())
    return dict(**query_metrics(model,sample),xy_rmse=float(diff[:,:2].square().sum(-1).mean().sqrt()),
                yaw_rmse_deg=float(yaw.square().mean().sqrt()*180/np.pi))


def precheck():
    context=prepare_context(PreflightBudget(150))
    print('SAME QUERY/120-UPDATE CONTROL VERIFIED; NO GRADIENT, UPDATE OR ROLLOUT',flush=True)
    return context


def run(output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); budget=PreflightBudget(150)
    report=dict(status='running',stage='identity',purpose='single_query_objective_mechanism_not_candidate',
        unique_inputs=1,presentations_per_arm=120,start_step=240,end_step=360,old_replay_present=False,
        allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False,completed_arms={})
    def save():
        report['elapsed_seconds']=time.perf_counter()-started; atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget); config=context['config']; sample=context['sample']
            atomic_write_json(output/'config.json',config); report['config_sha256']=file_digest(output/'config.json')
            report.update(before=metrics(context['model'],sample),cached_cumulative_control=context['control']); save()
            data=DatasetGenerator.load(INPUTS['train']); manifest=matrix.load_dataset_manifest(INPUTS['train'])
            vehicle=VehicleConfig(**manifest['vehicle_model'])
            cases={i:build_case(data,manifest,i,vehicle) for i in sorted(set(config['probe_indices']+config['fit_indices']))}
            case=cases[2149]; baseline,engine,_=run_case(case,context['model'],budget)
            states,commands=record_arrays(baseline)
            with np.load(QUERIES/'source-20261020-2149/learner.npz') as expected:
                if not np.array_equal(states,expected['states']) or not np.array_equal(commands,expected['commands']):
                    raise ValueError('原失败回合未逐元素复现')
            query_context=next(c for c in engine.contexts if c.step==400)
            if not np.array_equal(query_context.state.to_array()[:3],sample['global_pose']):
                raise ValueError('原查询位姿不匹配')
            teacher_ref=Trajectory(NetworkSource._to_global(np.asarray(sample['points'])[:19],query_context.state),context['model'].dt)
            teacher,teacher_engine,_=run_case(case,None,budget,context=query_context,trajectory=teacher_ref)
            states,commands=record_arrays(teacher)
            with np.load(TEACHER) as expected:
                if not np.array_equal(states,expected['states']) or not np.array_equal(commands,expected['commands']):
                    raise ValueError('原教师31步未逐元素复现')
            report.update(teacher_control=result_row(case,teacher,teacher_engine),teacher_exact=True,original_query_exact=True)
            save_record(output,'teacher-control',teacher,query_context,reference=teacher_ref.points); save()
            for arm in ARMS:
                budget.check(); report['stage']=arm; save(); print(f'FIT {arm} single-query updates=120 NOT A CANDIDATE',flush=True)
                model,optimizer,history,norms=fit_model(context['model'],context['optimizer_state'],sample,arm,budget,config['sample_identity'])
                directory=output/arm; directory.mkdir()
                torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),step=360,
                    threshold=model.stop_threshold,purpose='single_query_no_retention_non_deployable'),directory/'state.pt')
                point=dict(after=metrics(model,sample),loss_history=history,gradient_norms=norms,checkpoint=str(directory/'state.pt'))
                atomic_write_json(directory/'report.json',point)
                continuous,eng,_=run_case(case,model,budget,context=query_context)
                reference=Trajectory(continuous.record.refs[0].copy(),model.dt)
                fixed,fixed_eng,_=run_case(case,None,budget,context=query_context,trajectory=reference)
                a,u=record_arrays(continuous); b,v=record_arrays(fixed)
                if min(len(a),len(b))<10 or not np.array_equal(a[:10],b[:10]) or not np.array_equal(u[:10],v[:10]):
                    raise ValueError('同首次预测首10状态/控制未精确对齐')
                point.update(continuous=result_row(case,continuous,eng),fixed=result_row(case,fixed,fixed_eng),first_ten_exact=True)
                point['local_executable']=bool(point['continuous']['reached_with_margin'] and point['fixed']['reached_with_margin'])
                save_record(directory,'continuous',continuous,query_context,first_reference=reference.points)
                save_record(directory,'fixed-first',fixed,query_context,reference=reference.points)
                point['probe']=evaluate_cases(config['probe_indices'],cases,model,budget,directory,'probe')
                point['fit']=evaluate_cases(config['fit_indices'],cases,model,budget,directory,'fit')
                report['completed_arms'][arm]=point; atomic_write_json(directory/'report.json',point); save()
            verify_config(config,budget); budget.check()
            report.update(status='completed',stage='mechanistic_decision',
                local_executable_arms=[a for a,p in report['completed_arms'].items() if p['local_executable']],
                decision='fixed_local_objective_probe_not_retention_or_generalization',
                artifacts_sha256={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*') if p.is_file()
                    and p!=output/'report.json' and p.name not in ('run.lock','console.log')})
            save(); print('FINISHED',report['elapsed_seconds'],report['local_executable_arms'],flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',error=str(exc),
                          decision='incomplete_no_learnability_conclusion'); save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path); parser.add_argument('--precheck',action='store_true')
    args=parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output')
    else: run(args.output)
