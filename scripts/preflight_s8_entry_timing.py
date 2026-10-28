"""90秒零训练：同六实际查询的未补偿参考固定双倍时长反事实。"""
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
from experiments.s8_entry_recovery import select_entry_contexts,validated_entry_query
from experiments.s8_reference_timing import double_duration_reference
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
    old=json.loads((ENTRY/'config.json').read_text(encoding='utf-8'));verify_config(old,budget)
    if report['status']!='completed' or report['complete_queries']!=6 or report['training_updates']!=0:
        raise ValueError('非完整零训练六查询前置')
    for name,digest in report['artifacts_sha256'].items():
        budget.check();verify_files({str(artifact_path(ENTRY,name)):digest})
    cfg=deepcopy(old);cfg.update(kind='s8_entry_double_duration_preflight_v1',budget_seconds=90,
        timing_factor=2,reference_policy='archived_uncompensated_reference_original_knots_plus_midpoints_no_replan')
    cfg['frozen_files'].update({str(ENTRY/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(ENTRY,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s8_reference_timing
    for p in (Path(__file__).resolve(),Path(s8_reference_timing.__file__).resolve()):cfg['code'][str(p)]=file_digest(p)
    verify_config(cfg,budget);return cfg,report


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();budget=PreflightBudget(90)
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
                model,_=load_candidate(cfg['checkpoints']['preservation_student'][str(seed)],parent);case=cases[index]
                original,engine,frames=run_case(case,model,budget)
                archive=PREVIOUS/f'preservation_student-{seed}/source-{index}-control0.npz'
                states,_=exact_record(original,archive)
                contexts,selection=select_entry_contexts(engine.contexts,engine.margin_violations)
                if selection['selected_steps']!=EXPECTED_STEPS[seed,index]:raise ValueError('六查询点位漂移')
                baseline_source=next(s for s in previous['sources'] if s['seed']==seed and s['dataset_index']==index)
                source=dict(seed=seed,dataset_index=index,selection=selection,queries=[]);report['sources'].append(source);save()
                for at in contexts:
                    budget.check();query_dir=directory/f'query-{at.step}';query_dir.mkdir()
                    suffix,_,_=run_case(case,model,budget,context=at);exact_record(suffix,archive,at.step)
                    raw=ENTRY/f'source-{seed}-{index}/query-{at.step}/teacher-0.npz'
                    with np.load(raw) as z:baseline=Trajectory(z['reference'].copy(),float(z['dt']))
                    slow=double_duration_reference(baseline)
                    row=dict(step=at.step,remaining_steps=600-at.step,exact_student_suffix=True,
                        baseline_cached=next(q for q in baseline_source['queries'] if q['step']==at.step)['attempts'][0]['execution'],
                        original_points=len(baseline.points),slow_points=len(slow.points),original_knots_exact=np.array_equal(slow.points[::2],baseline.points),
                        reference_connected_safe=safe_connected_reference(slow,case['planner']._collision_checker,at.state),accepted=False)
                    source['queries'].append(row);save()
                    if not row['reference_connected_safe']:row['decision']='resampled_reference_unsafe';save();continue
                    print('SLOW REFERENCE',seed,index,at.step,len(slow.points),flush=True)
                    executed,slow_engine,teacher_frames=run_case(case,None,budget,context=at,trajectory=slow)
                    teacher_states,_=record_arrays(executed)
                    save_record(query_dir,'slow',executed,at,reference=slow.points,dt=np.asarray(slow.dt),
                        margin_violations=np.asarray(slow_engine.margin_violations),progress_trace=np.asarray(slow_engine.mpc.execution_trace))
                    row['execution']=result_row(case,executed,slow_engine)
                    sample=None
                    if teacher_frames and teacher_frames[0]['mask'].any():
                        sample,window=validated_entry_query(frames[at.step//10],teacher_frames,case,at.step,
                            f'preservation-{seed}-Adam268-double-duration',parent,budget,states,teacher_states)
                        row.update(window=window,input_sha256=input_id(sample),label_sha256=label_id(sample))
                        if row['execution']['reached_with_margin'] and window['clean']:
                            length=teacher_frames[0]['length']
                            exported=Trajectory(NetworkSource._to_global(np.asarray(sample['points'])[:length],at.state),slow.dt)
                            exported_result,exported_engine,_=run_case(case,None,budget,context=at,trajectory=exported)
                            save_record(query_dir,'export',exported_result,at,reference=exported.points,dt=np.asarray(exported.dt),
                                margin_violations=np.asarray(exported_engine.margin_violations))
                            row['export_execution']=result_row(case,exported_result,exported_engine)
                            row['accepted']=bool(row['export_execution']['reached_with_margin'])
                    torch.save(dict(first_teacher_frame=teacher_frames[0] if teacher_frames else None,sample=sample,
                        accepted=row['accepted']),query_dir/'query.pt')
                    if row['accepted']:
                        sample.update(teacher_kind='archived_uncorrected_reference_double_duration',teacher_evidence=str(query_dir))
                        verified.append(sample)
                    row['decision']='strict_slow_export_verified' if row['accepted'] else 'slow_execution_or_export_rejected';save()
            legacy=trusted_load(LT17/'data.pt',cfg);prior=legacy['old']+legacy['novel']
            prior+=trusted_load(ROOT/'runs/diagnostics/s6-chain-learning-preflight-20261022-v2/success-teacher-bank/bank.pt',cfg)['samples']
            retention=trusted_load(ROOT/'runs/diagnostics/s8-preservation-audit-20261022-v1/queries.pt',cfg)
            prior += [s for by in retention['queries_by_seed'].values() for samples in by.values() for s in samples]
            for name in ('learner-failure-teacher-preflight-20261022-v1','s6-replanning-chain-data-preflight-20261022-v2'):
                prior+=trusted_load(ROOT/f'runs/diagnostics/{name}/queries.pt',cfg)['samples']
            prior+=trusted_load(ENTRY/'queries.pt',cfg)['verified_samples']
            new,novelty=novel_samples(prior,verified)
            torch.save(dict(purpose='timing_teacher_feasibility_only_not_gradient_authorization',verified_samples=verified,novel_samples=new),output/'queries.pt')
            verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',complete_queries=sum(len(s['queries']) for s in report['sources']),
                verified_queries=len(verified),new_inputs=len(new),novelty=novelty,decision='fixed_timing_teacher_evidence_only_not_learning_benefit')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p!=output/'report.json' and p.name not in ('run.lock','console.log')}
            save();print('FINISHED',report['verified_queries'],report['new_inputs'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def precheck():
    cfg,_=prepare_config(PreflightBudget(45));print('TIMING INPUTS VERIFIED; ZERO PLANNING/ROLLOUT/TRAINING',flush=True);return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precheck',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.precheck:
        if a.output:p.error('precheck不写output')
        precheck()
    elif a.output is None:p.error('需要不存在的新output')
    else:run(a.output)
