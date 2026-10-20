"""准备/人工运行事件对齐与S8前10步探针；不训练、不部署。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from controller import MPCController
from dataset import DatasetGenerator
from experiments import validation_matrix
from experiments.recovery_execution import record_arrays,verify_network_suffix
from experiments.reference_event_probe import run_event_branch,summarize_events,validate_event
from experiments.reference_update_stability import ReferenceUpdateCollector,result_summary
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import NetworkSource,TerminalChecker
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest,runtime_identity
from sim import DifferentialDriveModel,VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PRIOR=ROOT/'runs/diagnostics/reference-update-stability-20261019-v1/config.json'
PRIOR_SHA='3474f667b06253515926546f4071087cf754c434cb505a61f6221d6e3175c211'


def fixed_cases():
    events=[(1298,210,'planning'),(1455,180,'planning'),(1559,310,'planning'),
            (1441,220,'contact'),(2364,10,'planning')]
    cases=[{'index':i,'step':step,'risk_type':risk,'role':'safety_regression_event','horizon_limited':False} for i,step,risk in events]
    cases.extend({'index':i,'step':step,'risk_type':None,'role':'success_control','horizon_limited':False} for i,step in [(1334,60),(1680,110)])
    cases.extend({'index':i,'step':0,'risk_type':None,'role':'early_execution','horizon_limited':True} for i in [2348,2330,2332,2317])
    return cases


def execution_identity():
    prior=json.loads(PRIOR.read_text(encoding='utf-8'))
    return {**prior['code'],**code_identity(),str(Path(__file__).resolve()):file_digest(__file__)}


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('必须使用全新目录')
    if file_digest(PRIOR)!=PRIOR_SHA:raise ValueError('LT07配置漂移')
    prior=json.loads(PRIOR.read_text(encoding='utf-8'))
    for path,digest in prior['code'].items():
        if file_digest(path)!=digest:raise ValueError(f'旧执行源码漂移: {path}')
    inputs=deepcopy(prior['inputs'])
    inputs['lt07_config']={'path':str(PRIOR),'sha256':PRIOR_SHA}
    report_path=PRIOR.parent/'results/report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report['status']!='completed' or report['smoke'] or len(report['tasks'])!=24 or report['config_sha256']!=PRIOR_SHA:
        raise ValueError('需要完整LT07正式报告')
    inputs['lt07_report']={'path':str(report_path),'sha256':file_digest(report_path)}
    cases=fixed_cases()
    for case in cases:
        row=next(r for r in report['tasks'] if r['dataset_index']==case['index'])
        case.update(task_id=row['task_id'],scene_name=row['scene_name'])
        if case['risk_type']:validate_event(row,case['step'],case['risk_type'])
        if case['role']=='success_control' and not row['original']['success']:raise ValueError('成功对照身份变化')
        folder=report_path.parent/f'task-{case["index"]:04d}'
        if json.loads((folder/'report.json').read_text(encoding='utf-8'))!=row:raise ValueError('LT07子报告不一致')
        if file_digest(folder/'trajectories.npz')!=row['trajectories_sha256']:raise ValueError('LT07轨迹摘要不符')
        for name in ('report.json','trajectories.npz'):
            path=folder/name
            inputs[f'task_{case["index"]}_{name}']={'path':str(path),'sha256':file_digest(path)}
    for value in inputs.values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('旧输入漂移')
    config={'kind':'reference_event_execution_probe','status':'prepared','inputs':inputs,
            'code':execution_identity(),'runtime':runtime_identity(),'cases':cases,'smoke_indices':[1298,2348],
            'output':str(output/'results'),'seed':0,'max_steps':600,'replan_every':10,'early_steps':10,
            'posthoc_mechanism_cohort_not_population_evaluation':True,
            'for_training':False,'automatic_training':False,'automatic_deployment':False}
    output.mkdir(parents=True)
    path=atomic_write_json(output/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path),'sources':len(cases),'inputs':len(inputs),'code':len(config['code'])},ensure_ascii=False))
    return config


def load_config(path,expected_sha):
    if file_digest(path)!=expected_sha:raise ValueError('配置摘要变化')
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    if config['kind']!='reference_event_execution_probe' or config['code']!=execution_identity() or config['runtime']!=runtime_identity():
        raise ValueError('种类/执行代码/环境变化')
    for value in config['inputs'].values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('输入身份变化')
    return config


def verify_archived_source(result,archive,*,prefix_only):
    arrays=record_arrays(result)
    for name,actual in zip(('original_states','original_commands'),arrays):
        expected=archive[name][:len(actual)] if prefix_only else archive[name]
        if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=0,atol=1e-9):
            raise ValueError('源状态/命令没有复现LT07完整轨迹或指定前缀')
    return {'matched':True,'prefix_only':prefix_only,'steps':result.steps}


def run_case(config,case,destination,*,smoke=False):
    destination.mkdir(parents=True,exist_ok=False)
    index=case['index']
    data_path=Path(config['inputs']['train']['path'])
    data=DatasetGenerator.load(data_path)
    metadata=data['task_meta'][index]
    if metadata['task_id']!=case['task_id'] or metadata['scene_name']!=case['scene_name']:raise ValueError('训练任务身份变化')
    manifest=validation_matrix.load_dataset_manifest(data_path)
    vehicle=VehicleConfig(**manifest['vehicle_model'])
    restored=validation_matrix.reconstruct_dataset_task(metadata,root_seed=manifest['seed'],vehicle=vehicle)
    model=load_model_checkpoint(config['inputs']['model']['path']).model
    planner,pipeline=validation_matrix._build_pipeline(validation_matrix.EXPERIMENT_SPECS['E5'],restored.task,vehicle)
    state=VehicleState.from_array(data['states'][index])
    initial=deepcopy(pipeline)
    initial.set_target_goals([restored.goal])
    if not np.array_equal(initial.capture_bev(state.x,state.y,state.yaw).data,data['bevs'][index]):raise ValueError('初态BEV归档不一致')
    contact=RectangleFootprintCollisionChecker(restored.task.scene.env,vehicle_length=vehicle.length,vehicle_width=vehicle.width,
                       collision_margin=0.,resolution=vehicle.collision_check_resolution)
    kwargs={'vehicle_model':DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
            'terminal':TerminalChecker(restored.tol_pos,restored.tol_yaw),'env':restored.task.scene.env,
            'collision_checker':contact,**vehicle.collision_kwargs()}
    source_budget=config['early_steps'] if case['horizon_limited'] else (case['step']+10 if smoke else config['max_steps'])
    engine=ReferenceUpdateCollector(mpc=MPCController(dt=.1,horizon=10,seed=config['seed']+index,**vehicle.mpc_kwargs()),
               source=NetworkSource(pipeline,model),planning_checker=planner._collision_checker,
               replan_every=config['replan_every'],max_steps=source_budget,**kwargs)
    print(f'#{index} {case["role"]} COLLECT BEGIN budget={source_budget}',flush=True)
    original=engine.run(state,restored.goal)
    archive=np.load(config['inputs'][f'task_{index}_trajectories.npz']['path'])
    source_verification=verify_archived_source(original,archive,prefix_only=smoke or case['horizon_limited'])
    context=next((c for c in engine.contexts if c.step==case['step']),None)
    if context is None:raise ValueError('冻结事件没有安全的指定时刻上下文，不自动换点')
    if case['risk_type']:
        update=next(u for u in engine.updates if u['step']==case['step'])
        if not update['safety_regression'][case['risk_type']]:raise ValueError('参考回退事件未复现')
    row={'dataset_index':index,'task_id':case['task_id'],'scene_name':case['scene_name'],
         'role':case['role'],'context_step':case['step'],'risk_type':case['risk_type'],
         'horizon_limited':case['horizon_limited'],'smoke':smoke,'for_training':False,
         'full_task_evaluation':not smoke and not case['horizon_limited'],
         'initial_bev_exactly_matches_training_archive':True,'actual_pipeline_target_override':pipeline.target_override,
         'source_verification':source_verification,'original':result_summary(original,engine.margin_violations,smoke=smoke),
         'updates':engine.updates,'branches':{}}
    arrays=dict(zip(('original_states','original_commands'),record_arrays(original)))
    arrays['context_state']=context.state.to_array()
    arrays.update({f'reference_{n:03d}':points for n,points in enumerate(engine.reference_plans)})
    budget=10 if smoke or case['horizon_limited'] else config['max_steps']-case['step']
    modes=['fresh','expert_once'] if case['step']==0 else ['fresh','delay_once','expert_once']
    for mode in modes:
        print(f'#{index} step={case["step"]} {mode} BEGIN budget={budget}',flush=True)
        result,margins,audit,counts=run_event_branch(context,mode,model,planner,restored.goal,kwargs,
                                   max_steps=budget,replan_every=config['replan_every'])
        if mode=='fresh':row['fresh_suffix_verification']=verify_network_suffix(original,result,case['step'],prefix_only=smoke)
        row['branches'][mode]={**result_summary(result,margins,smoke=smoke),**counts}
        arrays[mode+'_states'],arrays[mode+'_commands']=record_arrays(result)
        audit_path=atomic_write_json(destination/(mode+'-audit.json'),audit)
        row['branches'][mode]['audit_sha256']=file_digest(audit_path)
    np.savez_compressed(destination/'trajectories.npz',**arrays)
    row['trajectories_sha256']=file_digest(destination/'trajectories.npz')
    atomic_write_json(destination/'report.json',row)
    print(f'#{index} COMPLETE role={case["role"]} horizon_limited={case["horizon_limited"]} smoke={smoke}',flush=True)
    return row


def run(config_path,expected_sha,*,smoke=False,check_only=False):
    config=load_config(config_path,expected_sha)
    torch.set_num_threads(1)
    if check_only:
        print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True)
        return
    output=Path(config_path).resolve().parent/'preflight' if smoke else Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/'identity.json',{'config_sha256':expected_sha,'config':config,'formal':not smoke})
    started=time.perf_counter()
    rows=[]
    try:
        for case in config['cases']:
            if smoke and case['index'] not in config['smoke_indices']:continue
            rows.append(run_case(config,case,output/f'task-{case["index"]:04d}',smoke=smoke))
        load_config(config_path,expected_sha)
        atomic_write_json(output/'report.json',{'status':'completed','kind':'event_audit_not_training_data','smoke':smoke,
                'config_sha256':expected_sha,'elapsed_sec':time.perf_counter()-started,
                'summary':summarize_events(rows,smoke=smoke),'tasks':rows})
    except BaseException as exc:
        atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_sources':len(rows)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--config-sha256')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--smoke',action='store_true');modes.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        if args.config or args.config_sha256 or args.smoke or args.check_only:parser.error('prepare不可混用运行参数')
        prepare(args.prepare)
    else:
        if not args.config or not args.config_sha256:parser.error('运行必须提供配置与冻结摘要')
        run(args.config,args.config_sha256,smoke=args.smoke,check_only=args.check_only)


if __name__=='__main__':main()
