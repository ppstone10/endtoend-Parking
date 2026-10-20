"""人工运行S4连续恢复教师和S8几何尺度航向代价的独立前置验证。"""
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
from experiments import validation_matrix as vm
from experiments.recovery_execution import ContextCollector,record_arrays,verify_network_suffix
from experiments.recovery_compatibility import run_compatibility,summarize_compatibility
from experiments.reference_update_stability import result_summary
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import NetworkSource,TerminalChecker
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest,runtime_identity
from scripts.probe_reference_events import verify_archived_source
from sim import DifferentialDriveModel,VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PRIOR=ROOT/'runs/diagnostics/reference-event-probe-20261019-v1/config.json'
PRIOR_SHA='14b46ab57beb0ddc61da863572a6891da0a1181c20a32ec867a2a4d224a4659f'


def execution_identity():
    prior=json.loads(PRIOR.read_text(encoding='utf-8'))
    return {**prior['code'],**code_identity(),str(Path(__file__).resolve()):file_digest(__file__)}


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('必须使用全新目录')
    if file_digest(PRIOR)!=PRIOR_SHA:raise ValueError('LT08配置漂移')
    prior=json.loads(PRIOR.read_text(encoding='utf-8'))
    for path,digest in prior['code'].items():
        if file_digest(path)!=digest:raise ValueError('旧执行源漂移')
    report_path=PRIOR.parent/'results/report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report['status']!='completed' or report['smoke'] or len(report['tasks'])!=11:raise ValueError('要求完整LT08报告')
    inputs=deepcopy(prior['inputs'])
    inputs['lt08_config']={'path':str(PRIOR),'sha256':PRIOR_SHA}
    inputs['lt08_report']={'path':str(report_path),'sha256':file_digest(report_path)}
    stopped_root=ROOT/'runs/diagnostics/recovery-compatibility-20261020-v1'
    stopped_report=stopped_root/'preflight/report.json'
    stopped=json.loads(stopped_report.read_text(encoding='utf-8'))
    if stopped['status']!='completed' or not stopped['smoke']:
        raise ValueError('需要保留S8候选短预检停止证据')
    for name,path in [('stopped_s8_config',stopped_root/'config.json'),('stopped_s8_preflight',stopped_report)]:
        inputs[name]={'path':str(path),'sha256':file_digest(path)}
    cases=select_formal_cases(prior['cases'])
    for case in cases:
        case['cohort']='S4' if case['scene_name']=='S4_dump_area' else 'S8'
        row=next(t for t in report['tasks'] if t['dataset_index']==case['index'])
        folder=report_path.parent/f'task-{case["index"]:04d}'
        if json.loads((folder/'report.json').read_text(encoding='utf-8'))!=row or file_digest(folder/'trajectories.npz')!=row['trajectories_sha256']:
            raise ValueError('LT08逐源账本漂移')
        for name in ('report.json','trajectories.npz'):
            path=folder/name
            inputs[f'lt08_{case["index"]}_{name}']={'path':str(path),'sha256':file_digest(path)}
    for value in inputs.values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('旧输入漂移')
    config={'kind':'continuous_recovery_and_footprint_yaw','status':'prepared','inputs':inputs,
            'code':execution_identity(),'runtime':runtime_identity(),'cases':cases,'smoke_indices':[1298],
            'output':str(output/'results'),'seed':0,'max_steps':600,'replan_every':10,'early_steps':10,
            'footprint_gain':1.,'progress_floor_ratio':.8,'for_training':False,
            's8_geometry_cost_stopped_after_bounded_counterexample':True,
            'automatic_training':False,'automatic_expansion':False,'automatic_deployment':False}
    output.mkdir(parents=True)
    path=atomic_write_json(output/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path),'cases':len(cases),'inputs':len(inputs),'code':len(config['code'])}))
    return config


def select_formal_cases(cases):
    selected=[deepcopy(case) for case in cases if case['scene_name']=='S4_dump_area']
    if [case['index'] for case in selected]!=[1298,1455,1559,1441,1334,1680]:
        raise ValueError('S4正式来源/顺序漂移')
    return selected


def load_config(path,sha):
    if file_digest(path)!=sha:raise ValueError('配置摘要变化')
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    if c['kind']!='continuous_recovery_and_footprint_yaw' or c['code']!=execution_identity() or c['runtime']!=runtime_identity():
        raise ValueError('种类/源码/环境变化')
    for value in c['inputs'].values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('输入变化')
    if c['footprint_gain']!=1. or c['progress_floor_ratio']!=.8:raise ValueError('不是固定试验协议')
    return c


def verify_teacher_prefix(states,commands,archive,planning_failure):
    expected=min(10,len(archive['expert_once_states']))
    count=min(expected,len(states))
    for key,actual in [('expert_once_states',states),('expert_once_commands',commands)]:
        np.testing.assert_allclose(actual[:count],archive[key][:count],atol=1e-9,rtol=0)
    if count<expected and not planning_failure:raise ValueError('连续教师首个相同参考段意外提前终止')
    return {'matched':True if count==expected else None,'compared_steps':count,'expected_steps':expected,
            'incomplete_planning_failure':planning_failure if count<expected else None}


def run_case(config,case,destination,*,smoke=False):
    destination.mkdir(parents=True,exist_ok=False)
    index=case['index'];data_path=Path(config['inputs']['train']['path'])
    data=DatasetGenerator.load(data_path);meta=data['task_meta'][index]
    if meta['task_id']!=case['task_id']:raise ValueError('任务身份变化')
    manifest=vm.load_dataset_manifest(data_path);vehicle=VehicleConfig(**manifest['vehicle_model'])
    restored=vm.reconstruct_dataset_task(meta,root_seed=manifest['seed'],vehicle=vehicle)
    model=load_model_checkpoint(config['inputs']['model']['path']).model
    planner,pipeline=vm._build_pipeline(vm.EXPERIMENT_SPECS['E5'],restored.task,vehicle)
    state=VehicleState.from_array(data['states'][index]);initial=deepcopy(pipeline);initial.set_target_goals([restored.goal])
    if not np.array_equal(initial.capture_bev(state.x,state.y,state.yaw).data,data['bevs'][index]):raise ValueError('初态BEV不符')
    contact=RectangleFootprintCollisionChecker(restored.task.scene.env,vehicle_length=vehicle.length,vehicle_width=vehicle.width,
                    collision_margin=0.,resolution=vehicle.collision_check_resolution)
    kwargs={'vehicle_model':DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),'terminal':TerminalChecker(restored.tol_pos,restored.tol_yaw),
            'env':restored.task.scene.env,'collision_checker':contact,**vehicle.collision_kwargs()}
    source_budget=10 if case['cohort']=='S8' else (case['step']+30 if smoke else 600)
    engine=ContextCollector(mpc=MPCController(dt=.1,horizon=10,seed=index+config['seed'],**vehicle.mpc_kwargs()),
                    source=NetworkSource(pipeline,model),planning_checker=planner._collision_checker,
                    max_steps=source_budget,replan_every=10,**kwargs)
    print(f'#{index} {case["cohort"]} COLLECT BEGIN budget={source_budget}',flush=True)
    original=engine.run(state,restored.goal)
    archive=np.load(config['inputs'][f'lt08_{index}_trajectories.npz']['path'])
    source_check=verify_archived_source(original,archive,prefix_only=smoke or case['cohort']=='S8')
    context=next((c for c in engine.contexts if c.step==case['step']),None)
    if context is None:raise ValueError('冻结上下文缺失，不自动换点')
    row={'dataset_index':index,'task_id':case['task_id'],'cohort':case['cohort'],'role':case['role'],
         'context_step':case['step'],'smoke':smoke,'horizon_limited':case['cohort']=='S8','for_training':False,
         'source_verification':source_check,'initial_bev_exactly_matches_training_archive':True,
         'actual_pipeline_target_override':pipeline.target_override,'original':result_summary(original,engine.margin_violations,smoke=smoke),'branches':{}}
    arrays=dict(zip(('original_states','original_commands'),record_arrays(original)));arrays['context_state']=context.state.to_array()
    variants=[('original','network'),('original','expert_continuous')] if case['cohort']=='S4' else [
              ('original','network'),('original','expert_once'),('footprint_yaw','network'),('footprint_yaw','expert_once')]
    budget=(20 if smoke else 600-case['step']) if case['cohort']=='S4' else 10
    first_windows={}
    for solver,source in variants:
        name=solver+'_'+source
        print(f'#{index} {name} BEGIN budget={budget}',flush=True)
        result,margins,audit,details=run_compatibility(context,source,solver,model,planner,restored.goal,kwargs,
                    vehicle_length=vehicle.length,vehicle_width=vehicle.width,max_steps=budget,replan_every=10)
        states,commands=record_arrays(result)
        if name=='original_network':row['fresh_suffix_verification']=verify_network_suffix(original,result,case['step'],prefix_only=smoke)
        if source=='expert_once' and solver=='original':
            for key,actual in [('expert_once_states',states),('expert_once_commands',commands)]:
                np.testing.assert_allclose(actual,archive[key],atol=1e-9,rtol=0)
            row['expert_baseline_lt08_matched']=True
        if source=='expert_continuous':
            row['teacher_first_segment_lt08_verification']=verify_teacher_prefix(states,commands,archive,result.meta.get('planning_failure'))
        if audit:
            if solver=='original':first_windows[source]=audit[0]['window']
            else:np.testing.assert_allclose(audit[0]['window'],first_windows[source],atol=1e-9,rtol=0)
        teacher_refs=details.pop('teacher_references')
        if source=='expert_continuous':
            path=atomic_write_json(destination/'teacher-references.json',teacher_refs)
            details['teacher_references_sha256']=file_digest(path)
        row['branches'][name]={**result_summary(result,margins,smoke=smoke),**details}
        arrays[name+'_states'],arrays[name+'_commands']=states,commands
        path=atomic_write_json(destination/(name+'-audit.json'),audit)
        row['branches'][name]['audit_sha256']=file_digest(path)
    np.savez_compressed(destination/'trajectories.npz',**arrays)
    row['trajectories_sha256']=file_digest(destination/'trajectories.npz');atomic_write_json(destination/'report.json',row)
    print(f'#{index} COMPLETE cohort={case["cohort"]} smoke={smoke}',flush=True)
    return row


def run(config_path,sha,*,smoke=False,check_only=False):
    config=load_config(config_path,sha);torch.set_num_threads(1)
    if check_only:
        print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True);return
    output=Path(config_path).resolve().parent/'preflight' if smoke else Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/'identity.json',{'config_sha256':sha,'config':config,'formal':not smoke})
    started=time.perf_counter();rows=[]
    try:
        for case in config['cases']:
            if smoke and case['index'] not in config['smoke_indices']:continue
            rows.append(run_case(config,case,output/f'task-{case["index"]:04d}',smoke=smoke))
        load_config(config_path,sha)
        atomic_write_json(output/'report.json',{'status':'completed','smoke':smoke,'config_sha256':sha,
             'elapsed_sec':time.perf_counter()-started,'summary':summarize_compatibility(rows,smoke=smoke),'tasks':rows})
    except BaseException as exc:
        atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_sources':len(rows)});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--config',type=Path);p.add_argument('--config-sha256')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--smoke',action='store_true');modes.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.prepare:
        if a.config or a.config_sha256 or a.smoke or a.check_only:p.error('prepare不可混用运行参数')
        prepare(a.prepare)
    else:
        if not a.config or not a.config_sha256:p.error('必须提供配置与冻结摘要')
        run(a.config,a.config_sha256,smoke=a.smoke,check_only=a.check_only)


if __name__=='__main__':main()
