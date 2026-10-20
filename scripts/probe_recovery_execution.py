"""准备/人工运行训练域配对执行诊断；不训练、不部署、不产训练集。"""
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
from experiments.recovery_execution import ContextCollector, ContinuationContext, execute_branch, execution_summary, record_arrays, select_context, verify_network_suffix
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import NetworkSource, TerminalChecker
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest, runtime_identity
from sim import DifferentialDriveModel, VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


def execution_identity():
    return {**code_identity(), str(Path(__file__).resolve()): file_digest(__file__)}


def select_training_sources(metadata, excluded_ids, *, per_scene=6):
    """按机动/噪声轮询抽样，种子固定；不读取闭环结果挑任务。"""
    ids = [m['task_id'] for m in metadata]
    if len(ids) != len(set(ids)) or set(ids) & set(excluded_ids) or any(m.get('recovery') for m in metadata):
        raise ValueError('训练来源重复、混入恢复或与保留集身份重叠')
    rng = np.random.default_rng(20261019)
    selected = []
    for scene in ('S4_dump_area', 'S8_weigh_station'):
        pools = defaultdict(list)
        for index, meta in enumerate(metadata):
            if meta['scene_name'] == scene:
                pools[(meta['difficulty']['maneuver'], meta['difficulty']['noise_level'])].append(index)
        pools = {k: list(rng.permutation(v)) for k, v in sorted(pools.items())}
        chosen = []
        while len(chosen) < per_scene:
            previous = len(chosen)
            for pool in pools.values():
                if pool and len(chosen) < per_scene:
                    chosen.append(int(pool.pop()))
            if len(chosen) == previous:
                raise ValueError('场景训练任务不足，不自动替换来源')
        selected.extend(chosen)
    return selected


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('必须使用新目录，不能覆盖旧准备/运行')
    inputs = {'train': ROOT/'data/task_dataset/tracked_pivot_v8_3000/train.npz',
              'train_manifest': ROOT/'data/task_dataset/tracked_pivot_v8_3000/manifest.json',
              'validation': ROOT/'data/task_dataset/tracked_pivot_v8_3000/val.npz',
              'selection': ROOT/'data/task_dataset/independent_selection_v1/selection.npz',
              'model': ROOT/'runs/validation/arch-ab/dep-epoch0007.pt'}
    data = DatasetGenerator.load(inputs['train'])
    excluded = {m['task_id'] for name in ('validation', 'selection') for m in DatasetGenerator.load(inputs[name])['task_meta']}
    indices = select_training_sources(data['task_meta'], excluded)
    smoke_indices = [next(i for i in indices if data['task_meta'][i]['scene_name']==scene and data['task_meta'][i]['difficulty']['noise_level']!='clean') for scene in ('S4_dump_area','S8_weigh_station')]
    if len(data['task_meta']) != 2400 or file_digest(inputs['model']) != 'd7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8':
        raise ValueError('不是冻结的原训练来源/原父模型')
    config = {'kind': 'training_domain_recovery_execution_probe', 'status': 'prepared_for_preflight',
              'inputs': {k: {'path': str(p), 'sha256': file_digest(p)} for k,p in inputs.items()},
              'code': execution_identity(), 'runtime': runtime_identity(), 'output': str(output/'results'),
              'indices': indices, 'smoke_indices': smoke_indices, 'task_ids': [data['task_meta'][i]['task_id'] for i in indices],
              'seed': 0, 'max_steps': 600, 'replan_every': 10, 'per_scene': 6,
              'initial_input_contract': 'same_actual_pipeline_as_frozen_E5_and_stored_training_inputs',
              'declared_target_metadata_mismatch_not_modified': True,
              'for_training': False, 'automatic_training': False, 'automatic_deployment': False}
    output.mkdir(parents=True)
    path = atomic_write_json(output/'config.json',config)
    print(json.dumps({'config': str(path), 'sha256': file_digest(path), 'indices': indices},ensure_ascii=False))
    return config


def load_config(path, expected_sha):
    path = Path(path).resolve()
    if file_digest(path) != expected_sha:
        raise ValueError('配置摘要变化')
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['kind'] != 'training_domain_recovery_execution_probe' or config['code'] != execution_identity() or config['runtime'] != runtime_identity():
        raise ValueError('实验种类/执行代码/环境身份变化')
    for value in config['inputs'].values():
        if file_digest(value['path']) != value['sha256']:
            raise ValueError('输入身份变化')
    return config


def run_task(config, index, destination, *, smoke=False):
    destination.mkdir(parents=True, exist_ok=False)
    data_path = Path(config['inputs']['train']['path'])
    data = DatasetGenerator.load(data_path)
    metadata = data['task_meta'][index]
    manifest = validation_matrix.load_dataset_manifest(data_path)
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    restored = validation_matrix.reconstruct_dataset_task(metadata,root_seed=manifest['seed'],vehicle=vehicle)
    model = load_model_checkpoint(config['inputs']['model']['path']).model
    planner,pipeline = validation_matrix._build_pipeline(validation_matrix.EXPERIMENT_SPECS['E5'],restored.task,vehicle)
    state = VehicleState.from_array(data['states'][index])
    initial_pipeline = deepcopy(pipeline)
    initial_pipeline.set_target_goals([restored.goal])
    initial_bev = initial_pipeline.capture_bev(state.x,state.y,state.yaw)
    if not np.array_equal(initial_bev.data,data['bevs'][index]):
        raise ValueError('实际E5初态输入与训练归档不一致，停止而非切换通道')
    contact = RectangleFootprintCollisionChecker(restored.task.scene.env,vehicle_length=vehicle.length,vehicle_width=vehicle.width,collision_margin=0.,resolution=vehicle.collision_check_resolution)
    kwargs = {'vehicle_model': DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
              'terminal': TerminalChecker(restored.tol_pos,restored.tol_yaw), 'env': restored.task.scene.env,
              'collision_checker': contact, **vehicle.collision_kwargs()}
    engine = ContextCollector(mpc=MPCController(dt=.1,horizon=10,seed=config['seed']+index,**vehicle.mpc_kwargs()),
                              source=NetworkSource(pipeline,model),planning_checker=planner._collision_checker,
                              replan_every=config['replan_every'],max_steps=60 if smoke else config['max_steps'],**kwargs)
    print(f'#{index} {metadata["scene_name"]} COLLECT BEGIN',flush=True)
    original = engine.run(state,restored.goal)
    context,reason = select_context(engine.contexts,engine.margin_violations,original.success)
    if smoke:
        # 验证非零热启动并跨两次推理；不把截断结果当标签通过。
        candidates = [c for c in engine.contexts if 10 <= c.step <= 30]
        if not candidates:
            raise ValueError('短预检没有安全的非零时刻上下文，不能声称热启动验证通过')
        context,reason = candidates[0],'bounded_nonzero_context_preflight'
    row = {'dataset_index':index,'task_id':metadata['task_id'],'scene_name':metadata['scene_name'],
           'selection_reason':reason,'original':execution_summary(original,engine.margin_violations,truncated=smoke),
           'actual_pipeline_target_override':pipeline.target_override,'initial_bev_exactly_matches_training_archive':True,
           'context_step':None if context is None else context.step,'for_training':False,'smoke':smoke}
    original_states,original_commands = record_arrays(original)
    arrays = {'original_states':original_states,'original_commands':original_commands,'initial_state':state.to_array()}
    if context is not None:
        remaining = config['max_steps']-context.step
        budget = min(20,original.steps-context.step) if smoke else remaining
        print(f'#{index} step={context.step} NETWORK CONTINUATION BEGIN budget={budget}',flush=True)
        network,network_margin = execute_branch(context,'network',model,planner,restored.goal,kwargs,max_steps=budget,replan_every=config['replan_every'])
        row['network_suffix_verification'] = verify_network_suffix(original,network,context.step,prefix_only=smoke)
        row['network'] = execution_summary(network,network_margin,truncated=smoke)
        arrays['network_states'],arrays['network_commands'] = record_arrays(network)
        print(f'#{index} NETWORK SUFFIX MATCHED; EXPERT CONTINUATION BEGIN',flush=True)
        expert,expert_margin = execute_branch(context,'expert',model,planner,restored.goal,kwargs,max_steps=budget,replan_every=config['replan_every'],
                                              progress=lambda n,phase,elapsed: print(f'#{index} EXPERT REPLAN {n} {phase} elapsed={elapsed}',flush=True))
        row['expert'] = execution_summary(expert,expert_margin,truncated=smoke)
        arrays['expert_states'],arrays['expert_commands'] = record_arrays(expert)
        arrays['context_state'] = context.state.to_array()
        row['execution_label_passed'] = bool(not smoke and row['expert']['reached_with_margin'])
    else:
        row['execution_label_passed'] = False
    np.savez_compressed(destination/'trajectories.npz',**arrays)
    row['trajectories_sha256'] = file_digest(destination/'trajectories.npz')
    atomic_write_json(destination/'report.json',row)
    print(f'#{index} COMPLETE label_passed={row["execution_label_passed"]} smoke={smoke}',flush=True)
    return row


def summarize(rows):
    pairs = [r for r in rows if 'network' in r and 'expert' in r]
    wins = [r for r in pairs if not r['network']['success'] and r['execution_label_passed']]
    controls = [r for r in pairs if r['selection_reason']=='success_control']
    coverage = {scene: {'pairs':sum(r['scene_name']==scene for r in pairs),
                        'recovery':sum(r['scene_name']==scene and r['selection_reason']=='before_first_margin_loss' for r in pairs),
                        'controls':sum(r['scene_name']==scene for r in controls),
                        'expert_safe_recovery_wins':sum(r['scene_name']==scene for r in wins)} for scene in ('S4_dump_area','S8_weigh_station')}
    ready = not any(r.get('smoke',False) for r in rows) and len(pairs)>=8 and all(v['recovery'] and v['controls'] and v['expert_safe_recovery_wins'] for v in coverage.values()) and all(r['execution_label_passed'] for r in pairs)
    return {'selected_sources':len(rows),'paired_states':len(pairs),'missing_states':len(rows)-len(pairs),'coverage':coverage,
            'network_successes':sum(r['network']['success'] for r in pairs),'expert_successes':sum(r['expert']['success'] for r in pairs),
            'expert_execution_label_passed':sum(r['execution_label_passed'] for r in pairs),
            'decision':'candidate_for_separate_data_training_plan' if ready else 'insufficient_or_execution_gate_failed_no_training',
            'automatic_training':False}


def run(config_path, expected_sha, *, smoke=False, check_only=False):
    config = load_config(config_path,expected_sha)
    torch.set_num_threads(1)
    if check_only:
        print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True)
        return
    output = Path(config_path).resolve().parent/'preflight' if smoke else Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/'identity.json',{'config_sha256':expected_sha,'config':config,'formal':not smoke})
    started=time.perf_counter();rows=[]
    try:
        for index in config['smoke_indices'] if smoke else config['indices']:
            rows.append(run_task(config,index,output/f'task-{index:04d}',smoke=smoke))
        load_config(config_path,expected_sha)
        atomic_write_json(output/'report.json',{'status':'completed','kind':'diagnostic_not_training_data',
                          'smoke':smoke,'config_sha256':expected_sha,'elapsed_sec':time.perf_counter()-started,
                          'summary':summarize(rows),'tasks':rows})
    except BaseException as exc:
        atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_sources':len(rows)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--config-sha256')
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--smoke',action='store_true');modes.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        if args.config or args.config_sha256 or args.smoke or args.check_only:parser.error('prepare不能混用运行参数')
        prepare(args.prepare)
    else:
        if not args.config or not args.config_sha256:parser.error('运行必须指定config和冻结摘要')
        run(args.config,args.config_sha256,smoke=args.smoke,check_only=args.check_only)


if __name__=='__main__':main()
