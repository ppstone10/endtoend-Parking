"""180秒零训练：S8固定训练源起点与首次危险前访问的严格教师可执行性。"""
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
from interfaces import Trajectory
from runtime.sources import NetworkSource
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_failure_teacher_refinement import safe_connected_reference
from scripts.preflight_learner_failure_teacher import verify_config,LT17
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from scripts.validate_s8_preservation_sources import prepare_context as source_context
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


PREVIOUS=ROOT/'runs/diagnostics/s8-preservation-source-validation-20261022-v1'
HASHES={'config.json':'792f42ee8c6597b4cb7bf265b2fb2141ed23ffeef50b21ec6ebc3d38f37752dc',
    'report.json':'3dc03602400946b53f82eb8e48f046d19238fcede9b1335116f392254a83d3cf'}
SOURCES=((20261020,2338),(20261020,2351),(20261021,2338),(20261021,2351))
EXPECTED_STEPS={(20261020,2338):[0],(20261020,2351):[0,20],(20261021,2338):[0],(20261021,2351):[0,40]}


def prepare_config(budget):
    verify_files({str(PREVIOUS/k):v for k,v in HASHES.items()})
    report=json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    cfg=json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'));verify_config(cfg,budget)
    if report['status']!='completed' or report['complete_episodes']!=18 or report['training_updates']!=0:
        raise ValueError('不是完整LT19来源核验')
    for name,digest in report['artifacts_sha256'].items():
        budget.check();verify_files({str(artifact_path(PREVIOUS,name)):digest})
    rebuilt=source_context(budget)['config']
    if json.loads(json.dumps(rebuilt))!=cfg:raise ValueError('LT19配置无法重建')
    cfg=deepcopy(cfg);cfg.update(kind='s8_entry_recovery_preflight_v1',budget_seconds=180,
        sources=[dict(seed=s,source_index=i,expected_query_steps=EXPECTED_STEPS[s,i]) for s,i in SOURCES],
        max_teacher_updates=2,reference_policy='fresh_fixed_plan_existing_yaw_compensation',
        training_updates=0,learning_benefit_tested=False)
    cfg['frozen_files'].update({str(PREVIOUS/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(PREVIOUS,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s8_entry_recovery
    for p in (Path(__file__).resolve(),Path(s8_entry_recovery.__file__).resolve()):cfg['code'][str(p)]=file_digest(p)
    verify_config(cfg,budget);return cfg


def exact_record(result,path,step=0):
    states,commands=record_arrays(result)
    with np.load(path) as z:
        if not np.array_equal(states,z['states'][step:]) or not np.array_equal(commands,z['commands'][step:]):
            raise ValueError('学生原回合/实际上下文后缀未逐元素复现')
    return states,commands


def teacher_query(case,context,visited,learner_states,parent,policy,budget,directory,row,save):
    try:original=case['planner'].plan(context.state,case['goal'])
    except (ValueError,RuntimeError) as exc:
        row.update(decision='planner_rejected',error=str(exc));save();return None
    teacher=original
    for update in range(3):
        budget.check();safe=safe_connected_reference(teacher,case['planner']._collision_checker,context.state)
        attempt=dict(update=update,reference_connected_safe=safe,accepted=False)
        row['attempts'].append(attempt);save()
        if not safe:attempt['decision']='unsafe_connected_reference';save();break
        print('TEACHER',policy,case['index'],context.step,update,flush=True)
        try:result,engine,frames=run_case(case,None,budget,context=context,trajectory=teacher)
        except (ValueError,RuntimeError) as exc:
            attempt.update(decision='teacher_execution_error',error=str(exc));save();break
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
                exported=Trajectory(NetworkSource._to_global(np.asarray(sample['points'])[:length],context.state),teacher.dt)
                export_result,export_engine,_=run_case(case,None,budget,context=context,trajectory=exported)
                save_record(directory,f'export-{update}',export_result,context,reference=exported.points,
                    dt=np.asarray(exported.dt),margin_violations=np.asarray(export_engine.margin_violations))
                attempt['export_execution']=result_row(case,export_result,export_engine)
                attempt['accepted']=bool(attempt['export_execution']['reached_with_margin'])
        torch.save(dict(first_teacher_frame=frames[0] if frames else None,proposed_query=sample,
            accepted=attempt['accepted']),directory/f'query-{update}.pt')
        attempt['decision']='strict_export_verified' if attempt['accepted'] else 'execution_or_export_rejected';save()
        if attempt['accepted']:
            sample.update(teacher_kind='fresh_fixed_planner_existing_yaw_compensation',teacher_update=update,
                teacher_evidence=str(directory),source_student_policy=policy)
            row.update(accepted=True,decision='strict_teacher_available');save();return sample
        if execution['reached_with_margin'] or update==2 or not len(states):break
        teacher=compensate_reference_yaw(original,teacher,states,engine.mpc.execution_trace)
    row['decision']='no_verified_strict_teacher';save();return None


def trusted_load(path,cfg):
    resolved=str(Path(path).resolve())
    registered={str(Path(k).resolve()) for k in cfg['frozen_files']}
    if resolved not in registered:raise ValueError('旧监督缓存没有冻结身份')
    return torch.load(path,map_location='cpu',weights_only=False)


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
            cfg=prepare_config(budget);atomic_write_json(output/'config.json',cfg);report['config_sha256']=file_digest(output/'config.json')
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
                if selection['selected_steps']!=EXPECTED_STEPS[seed,index]:raise ValueError('预声明0/20/40查询点位漂移')
                source=dict(seed=seed,dataset_index=index,student=result_row(case,original,engine),selection=selection,queries=[])
                report['sources'].append(source);save()
                for at in contexts:
                    budget.check();query_dir=directory/f'query-{at.step}';query_dir.mkdir()
                    suffix,suffix_engine,_=run_case(case,model,budget,context=at)
                    exact_record(suffix,archive,at.step);save_record(query_dir,'student-suffix',suffix,at)
                    row=dict(step=at.step,remaining_steps=600-at.step,exact_student_suffix=True,attempts=[],accepted=False)
                    source['queries'].append(row);save()
                    sample=teacher_query(case,at,frames[at.step//10],states,parent,f'preservation-{seed}-Adam268',budget,query_dir,row,save)
                    if sample is not None:verified.append(sample)
            legacy=trusted_load(LT17/'data.pt',cfg)
            prior=legacy['old']+legacy['novel']
            success=trusted_load(ROOT/'runs/diagnostics/s6-chain-learning-preflight-20261022-v2/success-teacher-bank/bank.pt',cfg)
            prior+=success['samples']
            retention=trusted_load(ROOT/'runs/diagnostics/s8-preservation-audit-20261022-v1/queries.pt',cfg)
            prior += [s for by_source in retention['queries_by_seed'].values() for samples in by_source.values() for s in samples]
            for name in ('learner-failure-teacher-preflight-20261022-v1','s6-replanning-chain-data-preflight-20261022-v2'):
                prior+=trusted_load(ROOT/f'runs/diagnostics/{name}/queries.pt',cfg)['samples']
            new,novelty=novel_samples(prior,verified)
            torch.save(dict(purpose='teacher_feasibility_evidence_not_gradient_authorization',verified_samples=verified,
                novel_samples=new),output/'queries.pt')
            complete_queries=sum(len(s['queries']) for s in report['sources'])
            report.update(complete_queries=complete_queries,verified_queries=len(verified),new_inputs=len(new),novelty=novelty,
                strict_teacher_feasibility_passed=complete_queries==6 and len(verified)==6)
            verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',decision='teacher_feasibility_only_not_learning_benefit')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p!=output/'report.json' and p.name not in ('run.lock','console.log')}
            save();print('FINISHED',report['decision'],report['verified_queries'],report['new_inputs'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def precheck():
    config=prepare_config(PreflightBudget(45));print('ENTRY INPUTS/FIXED SOURCES VERIFIED; ZERO PLANNING/ROLLOUT/TRAINING',flush=True)
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precheck',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.precheck:
        if a.output:p.error('precheck不写output')
        precheck()
    elif a.output is None:p.error('需要不存在的新output')
    else:run(a.output)
