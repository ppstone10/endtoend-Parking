"""用真实归档输入独立重放五层，不从结果标签猜测模块健康。"""
from pathlib import Path
import numpy as np

from experiments import validation_matrix as matrix
from experiments.component_analysis import local_goal, transform_matches
from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState, ControlCmd
from planner import RectangleFootprintCollisionChecker
from runtime import NetworkSource, TerminalChecker
from sim import DifferentialDriveModel
from verification.candidate_execution import connected_safe
from verification.candidate_layer_quality import LAYERS, episode_verdict
from verification.feasible_reference_candidate import FeasibleReferenceCandidateMPC
from verification.perception_calibration import fidelity_observation, evaluate_profile, pipeline_semantics
from verification.planning_audit import sequence_safe
from verification.reference_retention import array_digest
from verification.terminal_study import read
from verification.typed_replay import scalar_type


def analyze_episode(folder, index, data, manifest, vehicle, loaded, profile, budget):
    folder = Path(folder);archived = read(folder/f'episode-{index:04d}.json')
    inputs = read(folder/f'inputs-{index:04d}.json');ledger = read(folder/f'retention-{index:04d}.json')
    task = matrix.reconstruct_dataset_task(data['task_meta'][index], root_seed=manifest['seed'], vehicle=vehicle)
    _, pipeline = matrix._build_pipeline(matrix.EXPERIMENT_SPECS['E5'], task.task, vehicle)
    pipeline.set_target_goals([task.goal]);semantics = pipeline_semantics(pipeline)
    terminal = TerminalChecker(task.tol_pos, task.tol_yaw)
    params = dict(vehicle_length=vehicle.length, vehicle_width=vehicle.width, resolution=vehicle.collision_check_resolution)
    physical = RectangleFootprintCollisionChecker(task.task.scene.env, collision_margin=0., **params)
    margin = RectangleFootprintCollisionChecker(task.task.scene.env, collision_margin=.2, **params)
    model = DifferentialDriveModel(**vehicle.vehicle_model_kwargs())
    controller = FeasibleReferenceCandidateMPC(contact_checker=physical, dt=.1, horizon=10, seed=index, **vehicle.mpc_kwargs())
    controller.bind_terminal(task.goal, terminal)
    errors = {name: [] for name in LAYERS};observations = [];profile_results = []
    facts = {'inputs_complete': False, 'physical_unsafe_references': 0, 'margin_unsafe_references': 0,
             'near_goal_queries': 0, 'near_goal_unsupported': 0, 'unsafe_actual_steps': 0,
             'unsafe_horizons': 0, 'success': archived['summary']['success'], 'steps': archived['summary']['steps']}

    def fail(layer, error, **details):
        errors[layer].append({'error': error, **details})

    with np.load(folder/f'episode-{index:04d}.npz', allow_pickle=False) as tape, \
            np.load(folder/f'inputs-{index:04d}.npz', allow_pickle=False) as packets, \
            np.load(folder/f'retention-{index:04d}.npz', allow_pickle=False) as raw:
        entries = inputs['packets'];n = len(entries)
        if inputs['status'] != 'completed' or not n or n != len(ledger['queries']):
            raise ValueError('完整输入记录/预测分母缺失')
        expected_keys = {f'{stem}_{q:04d}' for q in range(n) for stem in ('bev', 'local', 'model_state', 'raw')}
        if set(packets.files) != expected_keys or [p['step'] for p in entries] != tape['plan_issue_steps'].tolist():
            raise ValueError('输入张量/预测时刻缺失、重复或多余')
        if [p['ordinal'] for p in entries] != list(range(n)):
            raise ValueError('预测序号缺失或重复')
        if [p['step'] for p in entries] != list(range(0, facts['steps'], 10)):
            raise ValueError('实际重规划调度不符')
        frames = archived['frames']
        if len(frames) != facts['steps']:
            raise ValueError('实际控制分母不完整')
        current = VehicleState.from_array(tape['initial_state']);q = 0;trajectory = None;certified = {}
        for step, (before, after, command, selected, window, frame) in enumerate(zip(
                tape['states_before'], tape['states_after'], tape['commands'],
                tape['selected_control_sequences'], tape['control_windows'], frames, strict=True)):
            budget.check()
            if not np.array_equal(current.to_array(), before):
                raise ValueError('真实状态链不连续')
            if q < n and step == entries[q]['step']:
                entry = entries[q];query = ledger['queries'][q];metadata = entry['bev_metadata']
                state_values = packets[f'model_state_{q:04d}']
                types = [scalar_type(v) for v in (current.x, current.y, current.yaw, current.v, current.omega)]
                if (not np.array_equal(state_values, before) or types != entry['state_types']
                        or types[:3] != entry['capture_types'] or not np.array_equal(entry['capture_pose'], before[:3])):
                    fail('L1_L2', 'actual_input_state_or_type', query=q)
                bev = BEVTensor(packets[f'bev_{q:04d}'], metadata['resolution'], metadata['extent'], metadata['channels'],
                                camera_projection=metadata.get('camera_projection'))
                replay = pipeline.capture_bev(current.x, current.y, current.yaw)
                if (not np.isfinite(bev.data).all() or not np.array_equal(replay.data, bev.data)
                        or replay.to_metadata() != metadata):
                    fail('L1', 'actual_archived_bev_replay', query=q)
                if (semantics != entry['semantics'] or semantics['target_channel'] != loaded.model_config['target_channel']
                        or semantics['height_mode'] != loaded.model_config['height_mode']
                        or len(bev.channels) != loaded.model_config['bev_channels']):
                    fail('L1_L2', 'input_semantics', query=q)
                expected_goal = local_goal([task.goal.x, task.goal.y, task.goal.yaw], before)
                if not np.allclose(entry['goal_local'], expected_goal, atol=1e-12, rtol=0):
                    fail('L1_L2', 'local_goal_transform', query=q)
                local = loaded.model.predict(bev, GoalPose(*entry['goal_local']), current)
                expected_local = packets[f'local_{q:04d}'];world = packets[f'raw_{q:04d}']
                if (not np.array_equal(local.points, expected_local) or local.dt != entry['dt']
                        or entry['dt'] != loaded.model_config['dt']):
                    fail('L2', 'actual_archived_local_prediction', query=q)
                if not np.isfinite(world).all() or not np.isfinite(expected_local).all():
                    raise ValueError('参考包含非有限数')
                if (not transform_matches(expected_local, before, world)
                        or not np.array_equal(NetworkSource._to_global(expected_local, current), world)):
                    fail('L2_L3', 'local_world_transform', query=q)
                if (query['step'] != step or entry['raw_sha256'] != query['raw_sha256']
                        or array_digest(world) != query['raw_sha256'] or not np.array_equal(world, raw[query['raw_key']])
                        or query['raw_dt'] != entry['dt']):
                    raise ValueError('归档模型输出与真实原始参考不一致')
                used = tape[f'plan_{q:04d}'];dt = float(tape[f'plan_dt_{q:04d}'])
                if entry['used_sha256'] != query['used_sha256'] or array_digest(used) != query['used_sha256'] or dt != entry['dt']:
                    raise ValueError('原始/使用参考交接漂移')
                certificate = query['certificate']
                if certificate is None:
                    if not np.array_equal(used, world):
                        raise ValueError('无证书却替换原始参考')
                    trajectory = Trajectory(used.copy(), dt)
                else:
                    cert = ledger['certificates'][certificate]
                    if not np.array_equal(used, np.array(cert['reference'], dtype=cert['reference_dtype'])):
                        raise ValueError('使用参考不是证书参考')
                    if certificate not in certified:
                        certified[certificate] = Trajectory(used.copy(), dt)
                    trajectory = certified[certificate]
                safe = bool(connected_safe(physical, before, world));margin_safe = bool(connected_safe(margin, before, world))
                pos = np.hypot(world[:, 0]-task.goal.x, world[:, 1]-task.goal.y)
                yaw = np.abs(np.arctan2(np.sin(world[:, 2]-task.goal.yaw), np.cos(world[:, 2]-task.goal.yaw)))
                joint = bool(np.any((pos < terminal.tol_pos) & (yaw < terminal.tol_yaw)))
                near = bool(terminal.pos_err(current, task.goal) <= 3*terminal.tol_pos)
                facts['physical_unsafe_references'] += not safe;facts['margin_unsafe_references'] += not margin_safe
                facts['near_goal_queries'] += near;facts['near_goal_unsupported'] += near and not joint
                observed = fidelity_observation(task, vehicle, pipeline, bev, [current.x, current.y, current.yaw])
                conformance = evaluate_profile(profile, observed)
                if profile is not None and (profile['bev_metadata'] != metadata or profile['sensor_semantics'] != semantics):
                    conformance = 'blocked'
                profile_results.append(conformance)
                observations.append({'query': q, 'step': step, 'physical_safe': safe, 'margin_safe': margin_safe,
                                     'near_goal': near, 'has_joint_goal_point': joint,
                                     'fidelity': observed, 'perception_reference_conformance': conformance})
                q += 1
            if trajectory is None:
                raise ValueError('控制时缺少已发出参考')
            independent = controller.compute(trajectory, current)
            replay_window = controller._reference_points(controller._aligned_cache, controller._progress_idx)
            if (not np.array_equal([independent.v, independent.omega], command)
                    or not np.array_equal(controller._last_u.reshape(10, 2), selected)):
                fail('L3', 'candidate_control_replay', step=step)
            if controller._progress_idx != frame['progress'] or not np.array_equal(replay_window, window):
                fail('L2_L3', 'actual_window_or_progress', step=step)
            proposed = model.step(current, ControlCmd(*map(float, command)), .1)
            if not np.array_equal(proposed.to_array(), after):
                raise ValueError('真实动作不能重建后态')
            facts['unsafe_actual_steps'] += not (physical.swept_segment_free(before[:3], after[:3]) and
                                                  margin.swept_segment_free(before[:3], after[:3]))
            facts['unsafe_horizons'] += not sequence_safe(selected, current, model, margin)
            if terminal.reached(proposed, task.goal) != (facts['success'] and step == facts['steps']-1):
                fail('L2_L3', 'original_first_joint_terminal', step=step)
            current = proposed
        if q != n or facts['steps'] != len(tape['commands']):
            raise ValueError('全部预测/控制覆盖不完整')
        facts['inputs_complete'] = True
    return {'dataset_index': index, 'facts': facts, 'layers': episode_verdict(facts, errors, profile_results),
            'queries': observations, 'terminal': {'tol_pos': terminal.tol_pos, 'tol_yaw': terminal.tol_yaw},
            'absolute_perception_quality': 'not_established', 'production_qualified': False}
