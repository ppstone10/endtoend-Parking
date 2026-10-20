"""冻结并人工运行训练域参考更新验证；不训练、不部署。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller import MPCController
from dataset import DatasetGenerator
from experiments import validation_matrix
from experiments.reference_update_stability import ReferenceUpdateCollector, choose_context, continue_update, result_summary, summarize
from experiments.recovery_execution import record_arrays, verify_network_suffix
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import NetworkSource, TerminalChecker
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest, runtime_identity
from scripts.probe_recovery_execution import select_training_sources
from sim import DifferentialDriveModel, VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PRIOR = ROOT/'runs/diagnostics/recovery-execution-gate-20261019-v2/config.json'
PRIOR_SHA = 'f20a17905febf3db49922adc8960d165521e4240b21ce598263ca77d7122d37f'


def execution_identity():
    paths = [Path(__file__), ROOT/'scripts/probe_recovery_execution.py']
    return {**code_identity(), **{str(p.resolve()): file_digest(p) for p in paths}}


def select_sources(metadata, excluded):
    selected = select_training_sources(metadata, excluded, per_scene=12)
    selected = [i for i in selected if metadata[i]['scene_name']=='S4_dump_area'] + [
        i for i in selected if metadata[i]['scene_name']=='S8_weigh_station'][:6]
    rng = np.random.default_rng(20261020)
    for scene in ('S1_parking_lot', 'S2_diagonal_lot'):
        pools = defaultdict(list)
        for index, meta in enumerate(metadata):
            if meta['scene_name']==scene:
                pools[(meta['difficulty']['maneuver'], meta['difficulty']['noise_level'])].append(index)
        pools = {k: list(rng.permutation(v)) for k,v in sorted(pools.items())}
        chosen = []
        while len(chosen)<3:
            before = len(chosen)
            for pool in pools.values():
                if pool and len(chosen)<3:chosen.append(int(pool.pop()))
            if len(chosen)==before:raise ValueError('成功对照候选来源不足，不自动替代')
        selected.extend(chosen)
    return selected


def prepare(output):
    output = Path(output).resolve()
    if output.exists():raise FileExistsError('必须使用全新目录')
    if file_digest(PRIOR)!=PRIOR_SHA:raise ValueError('先前诊断配置漂移')
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    for path,digest in prior['code'].items():
        if file_digest(path)!=digest:raise ValueError(f'原代码漂移: {path}')
    inputs = deepcopy(prior['inputs'])
    for value in inputs.values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('原输入漂移')
    for name,path in [('prior_config',PRIOR),('prior_report',PRIOR.parent/'results/report.json')]:
        inputs[name] = {'path':str(path),'sha256':file_digest(path)}
    data = DatasetGenerator.load(inputs['train']['path'])
    excluded = {m['task_id'] for name in ('validation','selection')
                for m in DatasetGenerator.load(inputs[name]['path'])['task_meta']}
    indices = select_sources(data['task_meta'],excluded)
    smoke_indices = [next(i for i in indices if data['task_meta'][i]['scene_name']==scene and
                         data['task_meta'][i]['difficulty']['noise_level']!='clean')
                     for scene in ('S4_dump_area','S8_weigh_station')]
    config = {'kind':'training_domain_reference_update_validation','status':'prepared',
              'inputs':inputs,'code':execution_identity(),'runtime':runtime_identity(),
              'output':str(output/'results'),'indices':indices,'smoke_indices':smoke_indices,
              'task_ids':[data['task_meta'][i]['task_id'] for i in indices],
              'seed':0,'max_steps':600,'replan_every':10,
              'initial_input_contract':'unchanged_actual_E5_matches_archived_training_input',
              'declared_target_metadata_mismatch_not_modified':True,
              'for_training':False,'automatic_training':False,'automatic_deployment':False}
    output.mkdir(parents=True)
    path = atomic_write_json(output/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path),'indices':indices},ensure_ascii=False))
    return config


def load_config(path, expected_sha):
    path = Path(path).resolve()
    if file_digest(path)!=expected_sha:raise ValueError('配置摘要变化')
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['kind']!='training_domain_reference_update_validation' or config['code']!=execution_identity() or config['runtime']!=runtime_identity():
        raise ValueError('执行代码/环境/实验身份变化')
    for value in config['inputs'].values():
        if file_digest(value['path'])!=value['sha256']:raise ValueError('输入身份变化')
    return config


def verify_prior_source(config,index,summary):
    """重叠来源对齐旧完整回合；新字段不反向伪造旧报告。"""
    report = json.loads(Path(config['inputs']['prior_report']['path']).read_text(encoding='utf-8'))
    previous = next((r for r in report['tasks'] if r['dataset_index']==index),None)
    if previous is None:return {'applicable':False}
    expected = previous['original']
    for key,value in expected.items():
        actual = summary[key]
        if isinstance(value,float):
            if not np.isclose(actual,value,rtol=0,atol=1e-9):raise ValueError(f'原回合漂移 {index}: {key}')
        elif actual!=value:raise ValueError(f'原回合漂移 {index}: {key}')
    return {'applicable':True,'matched':True}


def run_task(config,index,destination,*,smoke=False):
    destination.mkdir(parents=True,exist_ok=False)
    data_path = Path(config['inputs']['train']['path'])
    data = DatasetGenerator.load(data_path)
    metadata = data['task_meta'][index]
    manifest = validation_matrix.load_dataset_manifest(data_path)
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    restored = validation_matrix.reconstruct_dataset_task(metadata,root_seed=manifest['seed'],vehicle=vehicle)
    model = load_model_checkpoint(config['inputs']['model']['path']).model
    planner,pipeline = validation_matrix._build_pipeline(validation_matrix.EXPERIMENT_SPECS['E5'],restored.task,vehicle)
    state = VehicleState.from_array(data['states'][index])
    initial = deepcopy(pipeline)
    initial.set_target_goals([restored.goal])
    if not np.array_equal(initial.capture_bev(state.x,state.y,state.yaw).data,data['bevs'][index]):
        raise ValueError('实际E5初态输入与训练归档不一致，不能切换通道')
    contact = RectangleFootprintCollisionChecker(restored.task.scene.env,vehicle_length=vehicle.length,
                  vehicle_width=vehicle.width,collision_margin=0.,resolution=vehicle.collision_check_resolution)
    kwargs = {'vehicle_model':DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
              'terminal':TerminalChecker(restored.tol_pos,restored.tol_yaw),'env':restored.task.scene.env,
              'collision_checker':contact,**vehicle.collision_kwargs()}
    engine = ReferenceUpdateCollector(mpc=MPCController(dt=.1,horizon=10,seed=config['seed']+index,**vehicle.mpc_kwargs()),
                 source=NetworkSource(pipeline,model),planning_checker=planner._collision_checker,
                 replan_every=config['replan_every'],max_steps=60 if smoke else config['max_steps'],**kwargs)
    print(f'#{index} {metadata["scene_name"]} COLLECT BEGIN',flush=True)
    original = engine.run(state,restored.goal)
    context,role = choose_context(engine.contexts,engine.margin_violations,original.success)
    if smoke:
        candidates = [c for c in engine.contexts if 10<=c.step<=30]
        if not candidates:raise ValueError('短预检没有安全非零上下文')
        context,role = candidates[0],'bounded_preflight_context'
    row = {'dataset_index':index,'task_id':metadata['task_id'],'scene_name':metadata['scene_name'],
           'context_role':role,'context_step':None if context is None else context.step,
           'original':result_summary(original,engine.margin_violations,smoke=smoke),
           'updates':engine.updates,'actual_pipeline_target_override':pipeline.target_override,
           'initial_bev_exactly_matches_training_archive':True,'for_training':False,'smoke':smoke}
    row['prior_source_verification'] = {'applicable':False,'reason':'truncated_smoke'} if smoke else verify_prior_source(config,index,row['original'])
    arrays = dict(zip(('original_states','original_commands'),record_arrays(original)))
    arrays.update({f'reference_{n:03d}':points for n,points in enumerate(engine.reference_plans)})
    if context is not None:
        budget = min(20,original.steps-context.step) if smoke else config['max_steps']-context.step
        print(f'#{index} step={context.step} FRESH CONTINUATION budget={budget}',flush=True)
        fresh,margins = continue_update(context,model,restored.goal,kwargs,planner._collision_checker,
                            delay_once=False,max_steps=budget,replan_every=config['replan_every'])
        row['fresh_suffix_verification'] = verify_network_suffix(original,fresh,context.step,prefix_only=smoke)
        row['fresh'] = result_summary(fresh,margins,smoke=smoke)
        arrays['fresh_states'],arrays['fresh_commands'] = record_arrays(fresh)
        print(f'#{index} FRESH SUFFIX MATCHED; SINGLE DELAY BEGIN',flush=True)
        delayed,margins = continue_update(context,model,restored.goal,kwargs,planner._collision_checker,
                             delay_once=True,max_steps=budget,replan_every=config['replan_every'])
        row['delay_once'] = result_summary(delayed,margins,smoke=smoke)
        arrays['delay_once_states'],arrays['delay_once_commands'] = record_arrays(delayed)
        arrays['context_state'] = context.state.to_array()
    np.savez_compressed(destination/'trajectories.npz',**arrays)
    row['trajectories_sha256'] = file_digest(destination/'trajectories.npz')
    atomic_write_json(destination/'report.json',row)
    print(f'#{index} COMPLETE role={role} smoke={smoke}',flush=True)
    return row


def run(config_path,expected_sha,*,smoke=False,check_only=False):
    config = load_config(config_path,expected_sha)
    torch.set_num_threads(1)
    if check_only:
        print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True)
        return
    output = Path(config_path).resolve().parent/'preflight' if smoke else Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/'identity.json',{'config_sha256':expected_sha,'config':config,'formal':not smoke})
    started = time.perf_counter()
    rows = []
    try:
        for index in config['smoke_indices'] if smoke else config['indices']:
            rows.append(run_task(config,index,output/f'task-{index:04d}',smoke=smoke))
        load_config(config_path,expected_sha)
        atomic_write_json(output/'report.json',{'status':'completed','kind':'diagnostic_not_training_data',
                   'smoke':smoke,'config_sha256':expected_sha,'elapsed_sec':time.perf_counter()-started,
                   'summary':summarize(rows,smoke=smoke),'tasks':rows})
    except BaseException as exc:
        atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_sources':len(rows)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--config-sha256')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--smoke',action='store_true')
    modes.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.config or args.config_sha256 or args.smoke or args.check_only:parser.error('prepare不能混用运行参数')
        prepare(args.prepare)
    else:
        if not args.config or not args.config_sha256:parser.error('必须指定配置及冻结摘要')
        run(args.config,args.config_sha256,smoke=args.smoke,check_only=args.check_only)


if __name__=='__main__':main()
