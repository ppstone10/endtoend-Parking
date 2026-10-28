"""60秒零训练：007唯一局部安全航向脉冲的原上下文完整分支验证。"""
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
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.recovery_execution import record_arrays
from interfaces import Trajectory
from scripts.diagnose_s8_yaw_response import ENTRY,PREVIOUS,exact_record
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


DIAGNOSTIC=ROOT/'runs/diagnostics/s8-yaw-response-diagnostic-20261022-v1'
HASHES={'report.json':'98b62d2ff6afc14ba76633a4d49eb648334ec485ae696659c53311eaa5f8864e',
    'config.json':'edcf4610abdd27c546907a2de782add2f3554eea62d2a4c1b0a65e4dc91140cd'}
EXPECTED=(20261021,2351,40,1,197,-.01)


def select_candidate(report):
    eligible=[]
    for event in report['events']:
        for response in event['responses']:
            if response['amplitude'] and all(response[k] for k in ('reference_internal_safe',
                    'original_start_connected_safe','event_window_connected_safe','actual_step_safe')):
                eligible.append((event['seed'],event['source_index'],event['query_step'],event['teacher_update'],
                    event['event'],response['amplitude']))
    if eligible!=[EXPECTED]:raise ValueError('固定唯一局部安全见证身份/数量漂移')
    return eligible[0]


def prepare_config(budget):
    verify_files({str(DIAGNOSTIC/k):v for k,v in HASHES.items()})
    report=json.loads((DIAGNOSTIC/'report.json').read_text(encoding='utf-8'))
    cfg=json.loads((DIAGNOSTIC/'config.json').read_text(encoding='utf-8'));verify_config(cfg,budget)
    if report['status']!='completed' or report['complete_events']!=5 or report['teacher_labels_created']:
        raise ValueError('非完整无监督诊断')
    candidate=select_candidate(report)
    for name,digest in report['artifacts_sha256'].items():
        budget.check();verify_files({str(artifact_path(DIAGNOSTIC,name)):digest})
    cfg=deepcopy(cfg);cfg.update(kind='s8_single_yaw_pulse_full_branch_v1',budget_seconds=60,candidate=list(candidate))
    cfg['frozen_files'].update({str(DIAGNOSTIC/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(DIAGNOSTIC,k)):v for k,v in report['artifacts_sha256'].items()})
    cfg['code'][str(Path(__file__).resolve())]=file_digest(__file__)
    verify_config(cfg,budget);return cfg


def first_difference(first,second):
    count=min(len(first),len(second));changed=np.flatnonzero(np.any(first[:count]!=second[:count],axis=1))
    return int(changed[0]) if len(changed) else count if len(first)!=len(second) else None


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();budget=PreflightBudget(60);torch.set_num_threads(1)
    report=dict(status='running',stage='identity',training_updates=0,teacher_labels_created=False,
        learning_benefit_tested=False,allowed_to_prepare_training_plan=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-started;atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            cfg=prepare_config(budget);atomic_write_json(output/'config.json',cfg);report['config_sha256']=file_digest(output/'config.json')
            seed,index,step,update,event,amplitude=EXPECTED
            data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train']);vehicle=VehicleConfig(**manifest['vehicle_model'])
            case=build_case(data,manifest,index,vehicle);parent=load_model_checkpoint(INPUTS['R0']).model
            model,_=load_candidate(cfg['checkpoints']['preservation_student'][str(seed)],parent)
            original,engine,_=run_case(case,model,budget)
            archive=PREVIOUS/f'preservation_student-{seed}/source-{index}-control0.npz';exact_record(original,archive)
            context=next(c for c in engine.contexts if c.step==step)
            suffix,_,_=run_case(case,model,budget,context=context);exact_record(suffix,archive,step)
            baseline_path=ENTRY/f'source-{seed}-{index}/query-{step}/teacher-{update}.npz'
            with np.load(baseline_path) as z:baseline=Trajectory(z['reference'].copy(),float(z['dt']))
            old,old_engine,_=run_case(case,None,budget,context=context,trajectory=baseline)
            old_states,old_commands=exact_record(old,baseline_path)
            with np.load(baseline_path) as z:
                if not np.array_equal(z['margin_violations'],old_engine.margin_violations):raise ValueError('原完整教师余量漂移')
            probe=DIAGNOSTIC/f'event-{seed}-{index}-{step}-{update}/probe-{amplitude:+.2f}.npz'
            with np.load(probe) as z:reference=Trajectory(z['reference'].copy(),baseline.dt)
            report.update(stage='single_full_branch',candidate=list(EXPECTED),baseline=result_row(case,old,old_engine));save()
            executed,new_engine,_=run_case(case,None,budget,context=context,trajectory=reference)
            states,commands=record_arrays(executed)
            save_record(output,'pulse',executed,context,reference=reference.points,dt=np.asarray(reference.dt),
                margin_violations=np.asarray(new_engine.margin_violations),progress_trace=np.asarray(new_engine.mpc.execution_trace))
            report.update(execution=result_row(case,executed,new_engine),
                first_command_difference=first_difference(old_commands,commands),
                first_state_difference=first_difference(old_states,states),
                original_event_step_present=event<len(new_engine.margin_violations),
                original_event_step_safe=not new_engine.margin_violations[event] if event<len(new_engine.margin_violations) else None)
            verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',decision='single_full_branch_evidence_only_no_labels_or_training')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p!=output/'report.json' and p.name not in ('run.lock','console.log')}
            save();print('FINISHED',report['execution'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def precheck():
    cfg=prepare_config(PreflightBudget(45));print('SINGLE PULSE INPUTS VERIFIED; ZERO ROLLOUT/TRAINING',flush=True);return cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--precheck',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.precheck:
        if args.output:parser.error('precheck不写output')
        precheck()
    elif args.output is None:parser.error('需要不存在的新output')
    else:run(args.output)
