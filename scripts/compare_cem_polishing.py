"""准备/人工执行原CEM与原代价局部精修的六源对照。"""
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
from experiments.cem_polishing import CEMPolishedMPC
from experiments.polishing_execution import run_polished_branch,summarize
from experiments.recovery_compatibility import run_compatibility
from experiments.recovery_execution import ContextCollector,record_arrays
from experiments.reference_event_probe import rollout_controls
from experiments.reference_update_stability import result_summary,window_geometry
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import NetworkSource,TerminalChecker
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest,runtime_identity
from scripts.probe_reference_events import verify_archived_source
from sim import DifferentialDriveModel,VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PRIOR=ROOT/'runs/diagnostics/recovery-compatibility-20261020-v3/config.json'
PRIOR_SHA='dcf2dd9be560a0c97c406ad0c2ba81bcc7d66177da555a68fbbb5cc8a49f2917'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def execution_identity():
    return {**read_json(PRIOR)['code'],**code_identity(),str(Path(__file__).resolve()):file_digest(__file__)}


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('必须使用全新目录')
    if file_digest(PRIOR)!=PRIOR_SHA:raise ValueError('LT09配置漂移')
    prior=read_json(PRIOR)
    for path,digest in prior['code'].items():
        if file_digest(path)!=digest:raise ValueError('旧94执行源漂移')
    inputs=deepcopy(prior['inputs']);report_path=PRIOR.parent/'results/report.json';report=read_json(report_path)
    if report['status']!='completed' or report['smoke'] or len(report['tasks'])!=6:raise ValueError('需要LT09完整六源')
    for name,path in [('lt09_config',PRIOR),('lt09_report',report_path)]:inputs[name]={'path':str(path),'sha256':file_digest(path)}
    for row in report['tasks']:
        index=row['dataset_index'];folder=report_path.parent/f'task-{index:04d}'
        if read_json(folder/'report.json')!=row or file_digest(folder/'trajectories.npz')!=row['trajectories_sha256']:
            raise ValueError('LT09逐源报告/轨迹不符')
        for name in ('teacher-references.json','original_network-audit.json','original_expert_continuous-audit.json'):
            key='teacher_references_sha256' if name.startswith('teacher-') else 'audit_sha256'
            branch='original_expert_continuous' if name.startswith(('teacher-','original_expert')) else 'original_network'
            if file_digest(folder/name)!=row['branches'][branch][key]:raise ValueError('LT09参考/审计摘要不符')
        for name in ('report.json','trajectories.npz','teacher-references.json','original_network-audit.json','original_expert_continuous-audit.json'):
            inputs[f'lt09_{index}_{name}']={'path':str(folder/name),'sha256':file_digest(folder/name)}
    for value in inputs.values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('输入变化')
    config={'kind':'cem_polishing_six_source_ab','status':'prepared','inputs':inputs,'code':execution_identity(),
            'runtime':runtime_identity(),'cases':prior['cases'],'smoke_indices':[1298,1455],
            'output':str(output/'results'),'seed':0,'max_steps':600,'replan_every':10,
            'max_polish_iterations':60,'max_evaluations':80,'for_training':False,
            'automatic_expansion':False,'automatic_training':False,'automatic_deployment':False}
    output.mkdir(parents=True);path=atomic_write_json(output/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path),'inputs':len(inputs),'code':len(config['code'])}))
    return config


def load_config(path,sha):
    if file_digest(path)!=sha:raise ValueError('配置摘要变化')
    c=read_json(path)
    if c['kind']!='cem_polishing_six_source_ab' or c['code']!=execution_identity() or c['runtime']!=runtime_identity():raise ValueError('执行身份变化')
    for value in c['inputs'].values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('输入身份变化')
    if c['max_polish_iterations']!=60 or c['max_evaluations']!=80:raise ValueError('精修预算变化')
    return c


def verify_baseline(result,archive,name,*,prefix_only):
    for key,actual in zip(('states','commands'),record_arrays(result)):
        expected=archive[name+'_'+key]
        if prefix_only:expected=expected[:len(actual)]
        if expected.shape!=actual.shape or not np.allclose(expected,actual,atol=1e-9,rtol=0):raise ValueError('原臂未复现LT09轨迹')
    return {'matched':True,'prefix_only':prefix_only,'steps':result.steps}


def fixed_window_probe(config,destination):
    data_path=Path(config['inputs']['train']['path']);data=DatasetGenerator.load(data_path)
    manifest=vm.load_dataset_manifest(data_path);vehicle=VehicleConfig(**manifest['vehicle_model']);rows=[]
    for index,step in [(1298,46),(1455,48)]:
        a=read_json(config['inputs'][f'lt09_{index}_original_expert_continuous-audit.json']['path'])[step]
        state=VehicleState.from_array(np.array(a['state']));window=np.array(a['window']);original=np.array(a['controls'])
        mpc=CEMPolishedMPC(dt=.1,horizon=10,**vehicle.mpc_kwargs());selected=mpc.polish_controls(original,window,state)
        restored=vm.reconstruct_dataset_task(data['task_meta'][index],root_seed=manifest['seed'],vehicle=vehicle)
        planner,_=vm._build_pipeline(vm.EXPERIMENT_SPECS['E5'],restored.task,vehicle)
        variants={}
        for name,controls in [('original',original),('polished',selected)]:
            rollout=rollout_controls(state,controls,.1)
            variants[name]={'controls':controls.tolist(),'cost':float(mpc._batch_cost(controls[None,:],window,state)[0]),
                           'first_transition_safe':bool(planner._collision_checker.swept_segment_free(state.to_array()[:3],rollout[0])),
                           'rollout_geometry':window_geometry(state,rollout,planner._collision_checker)}
        rows.append({'dataset_index':index,'relative_step':step,'state':a['state'],'window':a['window'],
                     'variants':variants,'polish':mpc.polish_audits[-1],'for_training':False})
    atomic_write_json(destination/'fixed-windows.json',rows)


def run_case(config,case,destination,*,smoke=False):
    destination.mkdir(parents=True,exist_ok=False);index=case['index'];data_path=Path(config['inputs']['train']['path'])
    data=DatasetGenerator.load(data_path);metadata=data['task_meta'][index]
    if metadata['task_id']!=case['task_id']:raise ValueError('训练任务身份变化')
    manifest=vm.load_dataset_manifest(data_path);vehicle=VehicleConfig(**manifest['vehicle_model'])
    restored=vm.reconstruct_dataset_task(metadata,root_seed=manifest['seed'],vehicle=vehicle)
    planner,pipeline=vm._build_pipeline(vm.EXPERIMENT_SPECS['E5'],restored.task,vehicle)
    model=load_model_checkpoint(config['inputs']['model']['path']).model;state=VehicleState.from_array(data['states'][index])
    initial=deepcopy(pipeline);initial.set_target_goals([restored.goal])
    if not np.array_equal(initial.capture_bev(state.x,state.y,state.yaw).data,data['bevs'][index]):raise ValueError('初态BEV不符')
    contact=RectangleFootprintCollisionChecker(restored.task.scene.env,vehicle_length=vehicle.length,vehicle_width=vehicle.width,
                  collision_margin=0.,resolution=vehicle.collision_check_resolution)
    kwargs={'vehicle_model':DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),'terminal':TerminalChecker(restored.tol_pos,restored.tol_yaw),
            'env':restored.task.scene.env,'collision_checker':contact,**vehicle.collision_kwargs()}
    engine=ContextCollector(mpc=MPCController(dt=.1,horizon=10,seed=index+config['seed'],**vehicle.mpc_kwargs()),
                 source=NetworkSource(pipeline,model),planning_checker=planner._collision_checker,
                 max_steps=case['step']+30 if smoke else 600,replan_every=10,**kwargs)
    print(f'#{index} SOURCE BEGIN',flush=True);original=engine.run(state,restored.goal)
    archive=np.load(config['inputs'][f'lt09_{index}_trajectories.npz']['path'])
    source_check=verify_archived_source(original,archive,prefix_only=smoke)
    context=next((c for c in engine.contexts if c.step==case['step']),None)
    if context is None:raise ValueError('固定上下文缺失')
    row={'dataset_index':index,'task_id':case['task_id'],'role':case['role'],'context_step':case['step'],
         'source_verification':source_check,'smoke':smoke,'for_training':False,'initial_bev_exactly_matches_training_archive':True,
         'actual_pipeline_target_override':pipeline.target_override,'original':result_summary(original,engine.margin_violations,smoke=smoke),'branches':{}}
    arrays=dict(zip(('original_states','original_commands'),record_arrays(original)));arrays['context_state']=context.state.to_array()
    originals={};budget=20 if smoke else 600-case['step']
    for solver,source in [('original','network'),('original','expert_continuous'),('polished','network'),('polished','expert_continuous')]:
        name=solver+'_'+source;print(f'#{index} {name} BEGIN budget={budget}',flush=True)
        if solver=='original':
            result,margins,audit,details=run_compatibility(context,source,'original',model,planner,restored.goal,kwargs,
                vehicle_length=vehicle.length,vehicle_width=vehicle.width,max_steps=budget,replan_every=10)
            row[name+'_lt09_verification']=verify_baseline(result,archive,name,prefix_only=smoke)
            originals[source]=audit
        else:
            result,margins,audit,details=run_polished_branch(context,source,model,planner,restored.goal,kwargs,max_steps=budget,replan_every=10)
            if audit:
                np.testing.assert_allclose(audit[0]['window'],originals[source][0]['window'],atol=1e-9,rtol=0)
                np.testing.assert_allclose(audit[0]['polish']['original_controls'],originals[source][0]['controls'],atol=1e-9,rtol=0)
        refs=details.pop('teacher_references')
        if source=='expert_continuous':
            if solver=='original':
                expected=read_json(config['inputs'][f'lt09_{index}_teacher-references.json']['path'])
                if smoke:expected=expected[:len(refs)]
                if refs!=expected:raise ValueError('原教师参考未复现LT09')
            path=atomic_write_json(destination/(name+'-references.json'),refs);details['teacher_references_sha256']=file_digest(path)
        row['branches'][name]={**result_summary(result,margins,smoke=smoke),**details}
        arrays[name+'_states'],arrays[name+'_commands']=record_arrays(result)
        path=atomic_write_json(destination/(name+'-audit.json'),audit);row['branches'][name]['audit_sha256']=file_digest(path)
    np.savez_compressed(destination/'trajectories.npz',**arrays)
    row['trajectories_sha256']=file_digest(destination/'trajectories.npz');atomic_write_json(destination/'report.json',row)
    print(f'#{index} COMPLETE smoke={smoke}',flush=True);return row


def run(path,sha,*,smoke=False,check_only=False):
    config=load_config(path,sha);torch.set_num_threads(1)
    if check_only:print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True);return
    output=Path(path).resolve().parent/'preflight' if smoke else Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/'identity.json',{'config_sha256':sha,'config':config,'formal':not smoke});started=time.perf_counter();rows=[]
    try:
        if smoke:fixed_window_probe(config,output)
        for case in config['cases']:
            if smoke and case['index'] not in config['smoke_indices']:continue
            rows.append(run_case(config,case,output/f'task-{case["index"]:04d}',smoke=smoke))
        load_config(path,sha)
        atomic_write_json(output/'report.json',{'status':'completed','smoke':smoke,'config_sha256':sha,'elapsed_sec':time.perf_counter()-started,
                  'summary':summarize(rows,smoke=smoke),'tasks':rows})
    except BaseException as exc:
        atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_sources':len(rows)});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--config',type=Path);p.add_argument('--config-sha256')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--smoke',action='store_true');modes.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.prepare:
        if a.config or a.config_sha256 or a.smoke or a.check_only:p.error('prepare不可混用运行参数')
        prepare(a.prepare)
    else:
        if not a.config or not a.config_sha256:p.error('要求配置与冻结摘要')
        run(a.config,a.config_sha256,smoke=a.smoke,check_only=a.check_only)


if __name__=='__main__':main()
