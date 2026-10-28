"""90秒首次更新分叉和410真实恢复标签门禁；不训练或部署。"""
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
from experiments.behavior_exploration_data import run_case,save_record
from experiments.effective_recovery_data import input_id,label_id,novel_samples
from experiments.executable_behavior import PreflightBudget,RecordedFixedExpert
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.learner_failure_teacher import validated_query
from experiments.recovery_execution import ContinuationContext,MarginAuditEngine,record_arrays,verify_network_suffix
from experiments.s6_replanning_chain import complete_consumption_mask,reference_audit,reference_at_query
from experiments.s6_recovery_learning import batch_identity
from interfaces import Trajectory
from runtime.sources import NetworkSource
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_learner_failure_teacher import verify_config,LT17
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s6_event_learnability import QUERIES,TEACHER
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


PREVIOUS=ROOT/'runs/diagnostics/s6-event-learnability-preflight-20261022-v1'
BIAS=PREVIOUS/'local-stop-bias-verified'
HASHES={'config.json':'4514646e3d5cabff3dfeef7ff131a011b05dd51f1a1f9de10ce97edac872d702',
        'report.json':'35e97822c96c27b2a6bdc210eabda59e2da85640241fe97b9aecdafc5b92323c',
        'local-stop-bias-verified/report.json':'859282a657ae0e1b658e6c5b8293388ae44c32956dbf8090fe92dc250b6a749e'}
POLICY='single-query-joint-bias-diagnostic-not-candidate'


def prepare_context(budget):
    torch.set_num_threads(1)
    verify_files({str(PREVIOUS/k):v for k,v in HASHES.items()})
    config=json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'))
    previous=json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    bias=json.loads((BIAS/'report.json').read_text(encoding='utf-8'))
    if (previous['status']!='completed' or bias['status']!='completed' or
            not bias['fixed']['reached_with_margin'] or bias['continuous']['success'] or
            not bias['native_predict_reference_matches_oracle'] or not bias['all320_points_unchanged']):
        raise ValueError('不是已核验的安全首参考/失败连续更新非候选控制')
    config=deepcopy(config)
    config['frozen_files'].update({str(PREVIOUS/k):v for k,v in HASHES.items()})
    for root,report in ((PREVIOUS,previous),(BIAS,bias)):
        config['frozen_files'].update({str(artifact_path(root,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s6_replanning_chain
    for path in (Path(s6_replanning_chain.__file__).resolve(),Path(__file__).resolve()):
        config['code'][str(path)]=file_digest(path)
    config.update(kind='s6_replanning_chain_data_preflight_v1',budget_seconds=90,query_policy=POLICY,
                  primary_query=410,monitor_query=420,training_performed=False)
    verify_config(config,budget)
    parent=load_model_checkpoint(INPUTS['R0']).model
    source,_=load_candidate(config['parent_checkpoints']['20261020'],parent)
    prototype,_=load_candidate(BIAS/'state.pt',parent)
    queries=torch.load(QUERIES/'queries.pt',map_location='cpu',weights_only=False)['samples']
    first=dict(next(s for s in queries if s['query_step']==400),mask=(torch.arange(320)<19).float())
    if batch_identity([first])[0]!=config['sample_identity']: raise ValueError('原400标签身份漂移')
    with np.load(TEACHER) as data: reference=data['reference'].copy()
    budget.check()
    return dict(config=config,parent=parent,source=source,prototype=prototype,previous=previous,bias=bias,
                first=first,queries=queries,expert_reference=reference)


def precheck():
    context=prepare_context(PreflightBudget(90))
    print('FROZEN PROTOTYPE/TEACHER IDENTITIES VERIFIED; NO TRAINING OR ROLLOUT',flush=True)
    return context


def exact_arrays(result,path):
    states,commands=record_arrays(result)
    with np.load(path) as expected:
        if not np.array_equal(states,expected['states']) or not np.array_equal(commands,expected['commands']):
            raise ValueError('原实际执行未逐元素复现')
    return states,commands


def run_retained_reference(case,context,budget):
    # 参考值相同但新对象会触发MPC进度重置；真正保留必须使用私有MPC里的原对象。
    mpc=deepcopy(context.mpc); mpc.budget=budget; mpc.execution_trace=[]
    if mpc._progress_traj is None: raise ValueError('缺可保留的原MPC参考')
    source=RecordedFixedExpert(deepcopy(context.pipeline),mpc._progress_traj,mpc,budget)
    if source.trajectory is not mpc._progress_traj: raise ValueError('保留参考对象身份丢失')
    engine=MarginAuditEngine(mpc=mpc,source=source,planning_checker=case['planner']._collision_checker,
        replan_every=10,max_steps=600-context.step,**case['engine_kwargs'])
    reset=mpc.reset; mpc.reset=lambda:None
    try: result=engine.run(deepcopy(context.state),case['goal'])
    finally: mpc.reset=reset
    return result,engine


def collect(context,budget,output,report,save):
    config=context['config']; data=DatasetGenerator.load(INPUTS['train']); manifest=matrix.load_dataset_manifest(INPUTS['train'])
    case=build_case(data,manifest,2149,VehicleConfig(**manifest['vehicle_model']))
    original,original_engine,_=run_case(case,context['source'],budget)
    original_states,_=exact_arrays(original,QUERIES/'source-20261020-2149/learner.npz')
    start=next(c for c in original_engine.contexts if c.step==400)
    rolling,engine,frames=run_case(case,context['prototype'],budget,context=start)
    rolling_states,_=exact_arrays(rolling,BIAS/'continuous.npz')
    save_record(output,'prototype-rolling',rolling,start)
    initial_reference=Trajectory(rolling.record.refs[0].copy(),context['prototype'].dt)
    held,held_engine,_=run_case(case,None,budget,context=start,trajectory=initial_reference)
    exact_arrays(held,BIAS/'fixed-first.npz'); save_record(output,'prototype-fixed',held,start,reference=initial_reference.points)
    contexts={400+c.step:ContinuationContext(400+c.step,c.state,c.mpc,c.pipeline) for c in engine.contexts}
    for step in (400,410,420):
        if step not in contexts: raise ValueError('缺预定安全重规划状态，不换时点')
        i=(step-400)//10; frame=frames[i]; reference=reference_at_query(rolling.record,400,step)
        if not np.array_equal(contexts[step].state.to_array()[:3],frame['pose']): raise ValueError('相对/绝对查询相位不符')
        report['reference_updates'].append(dict(step=step,input_sha256=input_id(frame),
            actual_consumed_raw_indices=np.flatnonzero(frame['mask']).tolist(),
            **reference_audit(reference,contexts[step].state,case,budget.check)))
        np.savez_compressed(output/f'reference-{step}.npz',reference=reference,context_state=contexts[step].state.to_array())
    torch.save(dict(frames=frames[:3],purpose='actual_non_candidate_policy_queries'),output/'visited-frames.pt')
    at=contexts[410]
    native,native_engine,_=run_case(case,context['prototype'],budget,context=at)
    matched=verify_network_suffix(rolling,native,10)
    a,u=record_arrays(native); b,v=record_arrays(rolling)
    if not np.array_equal(a,b[10:]) or not np.array_equal(u,v[10:]): raise ValueError('410原网络后缀不精确')
    updated_reference=Trajectory(reference_at_query(rolling.record,400,410),context['prototype'].dt)
    updated,updated_engine,_=run_case(case,None,budget,context=at,trajectory=updated_reference)
    c,w=record_arrays(updated)
    if min(len(a),len(c))<10 or not np.array_equal(a[:10],c[:10]) or not np.array_equal(u[:10],w[:10]):
        raise ValueError('更新参考同首10转移未对齐')
    republished,republished_engine,_=run_case(case,None,budget,context=at,trajectory=initial_reference)
    retained,retained_engine=run_retained_reference(case,at,budget)
    held_suffix=verify_network_suffix(held,retained,10)
    save_record(output,'410-native',native,at,first_reference=updated_reference.points)
    save_record(output,'410-updated-fixed',updated,at,reference=updated_reference.points)
    save_record(output,'410-initial-retained',retained,at,reference=initial_reference.points)
    save_record(output,'410-initial-republished',republished,at,reference=initial_reference.points)
    report.update(prototype_replayed=True,original_C240_replayed=True,
        branches_410=dict(native=result_row(case,native,native_engine),updated_fixed=result_row(case,updated,updated_engine),
                          initial_retained=result_row(case,retained,retained_engine),
                          initial_republished=result_row(case,republished,republished_engine),native_suffix=matched,
                          retained_suffix=held_suffix,updated_first_ten_exact=True,
                          first_progress=dict(native=native_engine.mpc.execution_trace[0],
                            updated=updated_engine.mpc.execution_trace[0],retained=retained_engine.mpc.execution_trace[0],
                            republished=republished_engine.mpc.execution_trace[0]))); save()
    expert=Trajectory(context['expert_reference'],context['prototype'].dt)
    teacher,teacher_engine,teacher_frames=run_case(case,None,budget,context=at,trajectory=expert)
    teacher_states,_=record_arrays(teacher)
    save_record(output,'410-expert-full',teacher,at,reference=expert.points)
    teacher_row=result_row(case,teacher,teacher_engine)
    first_prefix=np.concatenate((original_states[:400],rolling_states))
    sample,window=validated_query(frames[1],teacher_frames,case,410,POLICY,context['parent'],budget,first_prefix,teacher_states)
    proposal=dict(teacher=teacher_row,window=window,original_mask_indices=torch.nonzero(sample['mask']).flatten().tolist(),
                  first_offset=teacher_frames[0]['offset'],teacher_length=teacher_frames[0]['length'],accepted=False)
    completed=None
    try:
        length=teacher_frames[0]['length']; offset=teacher_frames[0]['offset']
        mask=complete_consumption_mask(len(expert.points),offset,length,teacher_engine.mpc.execution_trace)
        completed=dict(sample,mask=mask)
        exported=Trajectory(NetworkSource._to_global(np.asarray(completed['points'])[:length],at.state),context['prototype'].dt)
        replay,export_engine,_=run_case(case,None,budget,context=at,trajectory=exported)
        save_record(output,'410-exported-label',replay,at,reference=exported.points)
        export_row=result_row(case,replay,export_engine)
        bank=torch.load(LT17/'data.pt',map_location='cpu',weights_only=False)
        accepted,novelty=novel_samples(bank['old']+bank['novel']+context['queries'],[completed])
        proposal.update(export_execution=export_row,novelty=novelty,complete_mask_indices=torch.nonzero(mask).flatten().tolist(),
            input_sha256=input_id(completed),label_sha256=label_id(completed),
            accepted=bool(teacher_row['reached_with_margin'] and window['clean'] and export_row['reached_with_margin'] and len(accepted)==1))
    except ValueError as exc:
        proposal['rejection']=str(exc)
    report['query_410']=proposal
    torch.save(dict(first_teacher_frame=teacher_frames[0],sample=completed,accepted=proposal['accepted']),output/'query-410.pt')
    monitor,monitor_engine,_=run_case(case,None,budget,context=contexts[420],trajectory=expert)
    report['monitor_420']=result_row(case,monitor,monitor_engine)
    save_record(output,'420-expert-monitor',monitor,contexts[420],reference=expert.points)
    if proposal['accepted']:
        torch.save(dict(samples=[context['first'],completed],purpose='local_chain_mechanism_not_candidate',
                        source_policy=POLICY),output/'queries.pt')
    report['ready_for_local_learning_probe']=proposal['accepted']; save()


def run(output,*,budget_seconds=90):
    if budget_seconds not in (60,90): raise ValueError('只允许原90秒或修复后的剩余60秒')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); budget=PreflightBudget(budget_seconds)
    report=dict(status='running',stage='identity',reference_updates=[],training_performed=False,
        allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-started; atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget); config=context['config']; config['budget_seconds']=budget_seconds
            atomic_write_json(output/'config.json',config)
            report['config_sha256']=file_digest(output/'config.json'); report['stage']='actual_update_and_query'; save()
            collect(context,budget,output,report,save)
            verify_config(config,budget); budget.check()
            report.update(status='completed',stage='decision',decision='strict_410_query_ready_for_local_pair_probe'
                if report['ready_for_local_learning_probe'] else '410_label_gate_failed_no_training',
                artifacts_sha256={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*') if p.is_file()
                    and p!=output/'report.json' and p.name not in ('run.lock','console.log')})
            save(); print('FINISHED',report['decision'],report['elapsed_seconds'],flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',error=str(exc),
                          decision='incomplete_do_not_infer_update_cause'); save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path); parser.add_argument('--precheck',action='store_true')
    parser.add_argument('--budget-seconds',type=int,choices=(60,90),default=90)
    args=parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output')
    else: run(args.output,budget_seconds=args.budget_seconds)
