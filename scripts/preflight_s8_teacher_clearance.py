"""180秒零训练：六实际S8查询固定教师规划0.3m、原MPC执行0.2m对照。"""
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
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.recovery_execution import record_arrays
from experiments.reference_compensation import compensate_reference_yaw
from experiments.s8_entry_recovery import select_entry_contexts,validated_entry_query
from experiments.s8_teacher_clearance import clearance_teacher_planner
from interfaces import Trajectory
from runtime.sources import NetworkSource
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_failure_teacher_refinement import safe_connected_reference
from scripts.preflight_learner_failure_teacher import verify_config,LT17
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s8_entry_recovery import SOURCES,EXPECTED_STEPS,PREVIOUS,exact_record,trusted_load
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


ENTRY=ROOT/'runs/diagnostics/s8-entry-recovery-preflight-20261022-v1'
HASHES={'config.json':'4683e8e80fdeb2c6a3358ecc00491c6b5ded0ec4a8e95a3b7390ee5451ee8f32',
    'report.json':'cf7ac8c6bbbb891097f410fe2bcc732e9cb2d265b4db60893ac3be4d3e74abc3'}


def prepare_config(budget):
    verify_files({str(ENTRY/k):v for k,v in HASHES.items()})
    report=json.loads((ENTRY/'report.json').read_text(encoding='utf-8'))
    cfg=json.loads((ENTRY/'config.json').read_text(encoding='utf-8'));verify_config(cfg,budget)
    if report['status']!='completed' or report['complete_queries']!=6 or report['training_updates']!=0:
        raise ValueError('非完整零训练六查询前置')
    for name,digest in report['artifacts_sha256'].items():
        budget.check();verify_files({str(artifact_path(ENTRY,name)):digest})
    cfg=deepcopy(cfg);cfg.update(kind='s8_teacher_clearance_preflight_v1',budget_seconds=180,
        teacher_planning_margin=.3,execution_audit_margin=.2,
        reference_policy='independent_teacher_plan_margin_0.3_existing_two_yaw_updates_execution_margin_0.2')
    cfg['frozen_files'].update({str(ENTRY/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(ENTRY,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s8_teacher_clearance
    for p in (Path(__file__).resolve(),Path(s8_teacher_clearance.__file__).resolve()):cfg['code'][str(p)]=file_digest(p)
    verify_config(cfg,budget);return cfg,report


def teacher_query(case,context,visited,learner_states,parent,policy,budget,directory,row,save):
    planner=clearance_teacher_planner(case['planner']);checker=planner._collision_checker
    state=context.state;goal=case['goal']
    row.update(teacher_start_safe=checker.pose_free(state.x,state.y,state.yaw),
        teacher_goal_safe=checker.pose_free(goal.x,goal.y,goal.yaw),
        teacher_planning_margin=planner.collision_margin,execution_audit_margin=case['planner'].collision_margin)
    if not row['teacher_start_safe'] or not row['teacher_goal_safe']:
        row['decision']='inflated_start_or_goal_infeasible';save();return None
    try:original=planner.plan(state,goal)
    except (ValueError,RuntimeError) as exc:
        row.update(decision='planner_rejected',error=str(exc));save();return None
    teacher=original
    for update in range(3):
        budget.check();safe=safe_connected_reference(teacher,checker,state)
        attempt=dict(update=update,reference_connected_safe=safe,accepted=False)
        row['attempts'].append(attempt);save()
        if not safe:attempt['decision']='unsafe_inflated_connected_reference';save();break
        print('CLEARANCE TEACHER',policy,case['index'],context.step,update,flush=True)
        result,engine,frames=run_case(case,None,budget,context=context,trajectory=teacher)
        states,_=record_arrays(result)
        save_record(directory,f'teacher-{update}',result,context,reference=teacher.points,
            dt=np.asarray(teacher.dt),progress_trace=np.asarray(engine.mpc.execution_trace),
            margin_violations=np.asarray(engine.margin_violations))
        execution=result_row(case,result,engine);attempt['execution']=execution
        sample=None
        if frames and frames[0]['mask'].any():
            sample,window=validated_entry_query(visited,frames,case,context.step,policy,parent,budget,learner_states,states)
            attempt.update(window=window,input_sha256=input_id(sample),label_sha256=label_id(sample))
            if execution['reached_with_margin'] and window['clean']:
                length=frames[0]['length']
                exported=Trajectory(NetworkSource._to_global(np.asarray(sample['points'])[:length],state),teacher.dt)
                export_result,export_engine,_=run_case(case,None,budget,context=context,trajectory=exported)
                save_record(directory,f'export-{update}',export_result,context,reference=exported.points,
                    dt=np.asarray(exported.dt),margin_violations=np.asarray(export_engine.margin_violations))
                attempt['export_execution']=result_row(case,export_result,export_engine)
                attempt['accepted']=bool(attempt['export_execution']['reached_with_margin'])
        torch.save(dict(first_teacher_frame=frames[0] if frames else None,proposed_query=sample,
            accepted=attempt['accepted']),directory/f'query-{update}.pt')
        attempt['decision']='strict_export_verified' if attempt['accepted'] else 'execution_or_export_rejected';save()
        if attempt['accepted']:
            sample.update(teacher_kind='fixed_0.3m_planner_existing_yaw_compensation',teacher_update=update,
                teacher_evidence=str(directory),source_student_policy=policy)
            row.update(accepted=True,decision='strict_teacher_available');save();return sample
        if execution['reached_with_margin'] or update==2 or not len(states):break
        teacher=compensate_reference_yaw(original,teacher,states,engine.mpc.execution_trace)
    row['decision']='no_verified_strict_teacher';save();return None


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();budget=PreflightBudget(180)
    report=dict(status='running',stage='identity',sources=[],training_updates=0,learning_benefit_tested=False,
        allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-start;atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            cfg,previous=prepare_config(budget);atomic_write_json(output/'config.json',cfg);report['config_sha256']=file_digest(output/'config.json')
            parent=load_model_checkpoint(INPUTS['R0']).model
            data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train']);vehicle=VehicleConfig(**manifest['vehicle_model'])
            cases={i:build_case(data,manifest,i,vehicle) for i in (2338,2351)};verified=[]
            for seed,index in SOURCES:
                budget.check();report['stage']=f'source-{seed}-{index}';save()
                directory=output/f'source-{seed}-{index}';directory.mkdir()
                case=cases[index];model,_=load_candidate(cfg['checkpoints']['preservation_student'][str(seed)],parent)
                original,engine,frames=run_case(case,model,budget)
                archive=PREVIOUS/f'preservation_student-{seed}/source-{index}-control0.npz'
                states,_=exact_record(original,archive);save_record(directory,'student',original)
                contexts,selection=select_entry_contexts(engine.contexts,engine.margin_violations)
                if selection['selected_steps']!=EXPECTED_STEPS[seed,index]:raise ValueError('六查询点位漂移')
                baseline=next(s for s in previous['sources'] if s['seed']==seed and s['dataset_index']==index)
                source=dict(seed=seed,dataset_index=index,student=result_row(case,original,engine),selection=selection,queries=[])
                report['sources'].append(source);save()
                for at in contexts:
                    budget.check();query_dir=directory/f'query-{at.step}';query_dir.mkdir()
                    suffix,_,_=run_case(case,model,budget,context=at)
                    exact_record(suffix,archive,at.step);save_record(query_dir,'student-suffix',suffix,at)
                    row=dict(step=at.step,remaining_steps=600-at.step,exact_student_suffix=True,attempts=[],accepted=False,
                        baseline_cached=next(q for q in baseline['queries'] if q['step']==at.step)['attempts'])
                    source['queries'].append(row);save()
                    sample=teacher_query(case,at,frames[at.step//10],states,parent,
                        f'preservation-{seed}-Adam268-clearance',budget,query_dir,row,save)
                    if sample is not None:verified.append(sample)
                    if case['planner'].collision_margin!=.2:raise ValueError('原执行余量被意外修改')
            legacy=trusted_load(LT17/'data.pt',cfg);prior=legacy['old']+legacy['novel']
            prior+=trusted_load(ROOT/'runs/diagnostics/s6-chain-learning-preflight-20261022-v2/success-teacher-bank/bank.pt',cfg)['samples']
            retention=trusted_load(ROOT/'runs/diagnostics/s8-preservation-audit-20261022-v1/queries.pt',cfg)
            prior += [s for by in retention['queries_by_seed'].values() for samples in by.values() for s in samples]
            for name in ('learner-failure-teacher-preflight-20261022-v1','s6-replanning-chain-data-preflight-20261022-v2'):
                prior+=trusted_load(ROOT/f'runs/diagnostics/{name}/queries.pt',cfg)['samples']
            prior+=trusted_load(ENTRY/'queries.pt',cfg)['verified_samples']
            new,novelty=novel_samples(prior,verified)
            torch.save(dict(purpose='clearance_teacher_feasibility_only_not_gradient_authorization',verified_samples=verified,
                novel_samples=new),output/'queries.pt')
            complete=sum(len(s['queries']) for s in report['sources'])
            verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',complete_queries=complete,verified_queries=len(verified),new_inputs=len(new),
                novelty=novelty,strict_teacher_feasibility_passed=complete==6 and len(verified)==6,
                decision='fixed_clearance_teacher_evidence_only_not_learning_benefit')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p!=output/'report.json' and p.name not in ('run.lock','console.log')}
            save();print('FINISHED',report['verified_queries'],report['new_inputs'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def precheck():
    cfg,_=prepare_config(PreflightBudget(45));print('CLEARANCE INPUTS VERIFIED; ZERO PLANNING/ROLLOUT/TRAINING',flush=True);return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precheck',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.precheck:
        if a.output:p.error('precheck不写output')
        precheck()
    elif a.output is None:p.error('需要不存在的新output')
    else:run(a.output)
