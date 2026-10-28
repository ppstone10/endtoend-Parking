"""训练源感知经验参考包络；不伪装为绝对质量或统计安全界。"""
from collections import defaultdict
from pathlib import Path
import json
import time
import numpy as np

from experiments import validation_matrix as matrix
from interfaces import VehicleState
from interfaces.bev import CAMERA_PROJECTION_CURRENT, CAMERA_PROJECTION_GEOMETRY
from metrics.bev_fidelity import (compute_bev_fidelity_metrics, rasterize_ground_truth_target,
                                  rasterize_lidar_truth_occupancy)
from sim import VehicleConfig
from sim.sensor_sim import SimulatedLiDAR
from training.reporting import atomic_write_json
from scripts.diagnose_closed_loop_failures import _sha256
from verification.terminal_study import read

METRICS = ('occupancy_iou', 'occupancy_recall', 'target_iou', 'target_hit_rate')
RULE = {'min_fit': 20, 'min_holdout': 10, 'max_holdout_failure_rate': .05,
        'max_tasks_per_group': 96, 'split': 'seed_sorted_modulo_3_holdout_zero',
        'bounds': 'fit_minimum', 'scope': 'empirical_reference_conformance_only'}


def pipeline_semantics(pipeline):
    if not isinstance(pipeline.lidar_sensor, SimulatedLiDAR):
        raise ValueError('未验证的高度语义，禁止猜测')
    return {'target_channel': 'geometry' if pipeline.target_override else 'image',
            'height_mode': 'semantic', 'lidar_beams': pipeline.lidar_sensor.beams,
            'camera_projection': CAMERA_PROJECTION_GEOMETRY if pipeline.target_override else CAMERA_PROJECTION_CURRENT}


def fidelity_observation(task, vehicle, pipeline, bev, pose):
    truth = rasterize_lidar_truth_occupancy(task.task.scene.env, *pose, pipeline.bev_config,
                                           max_range=pipeline.lidar_sensor.max_range)
    target = rasterize_ground_truth_target(task.goal, vehicle.length, vehicle.width, *pose, pipeline.bev_config)
    metrics = compute_bev_fidelity_metrics(bev, truth, target).to_dict()
    metrics = {key: float(metrics[key]) for key in METRICS}
    if not np.any(truth):
        metrics.update(occupancy_iou=None, occupancy_recall=None)
    if not np.any(target):
        metrics.update(target_iou=None, target_hit_rate=None)
    noise = task.task.difficulty.noise_level
    noise = getattr(noise, 'value', str(noise))
    group = f'{task.task.scene.name}|{noise}|target={int(np.any(target))}'
    return {'group': group, 'metrics': metrics, 'target_observed': bool(np.any(target)),
            'occupancy_observed': bool(np.any(truth))}


def fit_profile(observations, *, preflight=False):
    groups = defaultdict(list)
    for row in observations:
        for value in row['metrics'].values():
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('感知指标必须有限且在[0,1]内')
        groups[row['group']].append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row['seed'])
        if len({r['seed'] for r in ordered}) != len(ordered):
            raise ValueError('同组校准seed重复，不得把相关帧算独立任务')
        required = list(METRICS if ordered[0]['target_observed'] else METRICS[:2])
        fit = [r for n, r in enumerate(ordered) if n % 3 != 0]
        hold = [r for n, r in enumerate(ordered) if n % 3 == 0]
        usable_fit = [r for r in fit if all(r['metrics'][k] is not None for k in required)]
        usable_hold = [r for r in hold if all(r['metrics'][k] is not None for k in required)]
        enough = len(usable_fit) >= RULE['min_fit'] and len(usable_hold) >= RULE['min_holdout']
        bounds = {k: min(r['metrics'][k] for r in usable_fit) for k in required} if usable_fit else {}
        bad = sum(any(r['metrics'][k] < bounds[k] for k in required) for r in usable_hold) if bounds else 0
        rate = bad / len(usable_hold) if usable_hold else None
        ready = bool(enough and not preflight and rate <= RULE['max_holdout_failure_rate'])
        result[name] = {'status': 'ready' if ready else 'blocked' if not enough or preflight else 'failed',
                        'bounds': bounds, 'required_metrics': required,
                        'fit_seeds': [r['seed'] for r in usable_fit], 'holdout_seeds': [r['seed'] for r in usable_hold],
                        'holdout_failure_rate': rate, 'unobserved_tasks': len(rows)-len(usable_fit)-len(usable_hold)}
    return {'schema': 'perception_reference_profile_v1', 'status': 'completed', 'rule': RULE,
            'preflight': preflight, 'groups': result, 'absolute_quality_status': 'not_established'}


def evaluate_profile(profile, observation):
    if profile is None or profile.get('preflight') or profile.get('schema') != 'perception_reference_profile_v1':
        return 'blocked'
    group = profile.get('groups', {}).get(observation['group'])
    if group is None or group['status'] != 'ready':
        return 'blocked'
    metrics = observation['metrics']
    if any(metrics.get(k) is None or not np.isfinite(metrics[k]) for k in group['required_metrics']):
        return 'blocked'
    return 'failed' if any(metrics[k] < group['bounds'][k] for k in group['required_metrics']) else 'passed'


def calibration_selection(train, validation):
    """只选原训练任务起点；排除恢复、重复seed及与val相交的任务。"""
    val_ids = {m['task_id'] for m in validation}
    val_seeds = {int(m['seed']) for m in validation}
    groups, seen = defaultdict(list), set()
    for index, meta in enumerate(train):
        seed = int(meta['seed'])
        if meta.get('recovery') or meta['task_id'] in val_ids or seed in val_seeds or seed in seen:
            continue
        seen.add(seed)
        group = (meta['scene_name'], meta['difficulty']['noise_level'])
        groups[group].append((seed, index))
    return [index for group in sorted(groups) for seed, index in sorted(groups[group])[:RULE['max_tasks_per_group']]]


def run_calibration(config, output, *, preflight=False):
    from verification.candidate_layer_study import precheck
    c = precheck(config);output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output/'config.json', c)
    report = {'status': 'running', 'config_sha256': _sha256(Path(config)), 'preflight': preflight}
    atomic_write_json(output/'report.json', report)
    started = time.perf_counter()
    try:
        with np.load(c['train'], allow_pickle=False) as tape:
            train_meta = [json.loads(str(x)) for x in tape['task_meta']]
        with np.load(c['data'], allow_pickle=False) as tape:
            val_meta = [json.loads(str(x)) for x in tape['task_meta']]
        indices = calibration_selection(train_meta, val_meta)
        if preflight:
            indices = indices[:2]
        manifest = matrix.load_dataset_manifest(Path(c['data']))
        vehicle = VehicleConfig(**manifest['vehicle_model']);observations = []
        for ordinal, index in enumerate(indices):
            if time.perf_counter()-started > (45 if preflight else 7200):
                raise TimeoutError('感知参考校准预算耗尽')
            task = matrix.reconstruct_dataset_task(train_meta[index], root_seed=manifest['seed'], vehicle=vehicle)
            _, pipeline = matrix._build_pipeline(matrix.EXPERIMENT_SPECS['E5'], task.task, vehicle)
            pipeline.set_target_goals([task.goal]);start = task.task.start
            bev = pipeline.capture_bev(start.x, start.y, start.yaw)
            observed = fidelity_observation(task, vehicle, pipeline, bev, [start.x, start.y, start.yaw])
            observed.update(seed=int(train_meta[index]['seed']), index=index, task_id=train_meta[index]['task_id'])
            observations.append(observed)
            if ordinal % 25 == 0:
                print(f'[perception-reference {ordinal+1}/{len(indices)}]', flush=True)
        if not observations:
            raise ValueError('无可用训练源校准任务')
        profile = fit_profile(observations, preflight=preflight)
        profile.update(config_sha256=_sha256(Path(config)), code=c['code'], inputs=c['inputs'],
                       sensor_semantics=pipeline_semantics(pipeline), vehicle=vehicle.to_metadata(),
                       bev_metadata=bev.to_metadata(), observations=observations)
        atomic_write_json(output/'profile.json', profile)
        report.update(status='completed', tasks=len(observations), groups=len(profile['groups']),
                      ready_groups=sum(g['status']=='ready' for g in profile['groups'].values()),
                      absolute_quality_status='not_established', elapsed_seconds=time.perf_counter()-started)
        atomic_write_json(output/'report.json', report)
        atomic_write_json(output/'artifacts.json', {p.name: _sha256(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(output/'report.json', report)
        raise


def load_profile(folder, config):
    if folder is None:
        return None
    folder = Path(folder).resolve();manifest = read(folder/'artifacts.json')
    required = {'profile.json', 'report.json', 'config.json'}
    if not required <= set(manifest):
        raise ValueError('校准证据缺失')
    for name, digest in manifest.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or _sha256(path) != digest:
            raise ValueError('校准证据漂移')
    profile = read(folder/'profile.json');report = read(folder/'report.json')
    if (report['status'] != 'completed' or profile['code'] != config['code']
            or profile['rule'] != RULE or profile['config_sha256'] != _sha256(folder/'config.json')):
        raise ValueError('校准版本/规则/配置身份不符')
    expected = read(folder/'config.json')
    if expected['inputs'] != config['inputs'] or profile['inputs'] != config['inputs']:
        raise ValueError('校准输入身份不符')
    derived = fit_profile(profile['observations'], preflight=profile['preflight'])
    if (any(profile.get(k) != value for k, value in derived.items())
            or report['config_sha256'] != profile['config_sha256']
            or report['tasks'] != len(profile['observations'])
            or report['ready_groups'] != sum(g['status']=='ready' for g in derived['groups'].values())):
        raise ValueError('经验包络拟合/留出统计不能独立复现')
    return profile
