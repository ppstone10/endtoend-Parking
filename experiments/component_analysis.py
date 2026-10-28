"""在原生 E5 访问状态上分别重放生产感知、网络和 MPC，并核验接口。"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from controller import MPCController
from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from experiments.component_verification import layer_status, locate_failure, pose_errors, trajectory_demands
from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState
from metrics.bev_fidelity import (compute_bev_fidelity_metrics, rasterize_ground_truth_target,
                                  rasterize_lidar_truth_occupancy)
from runtime import TerminalChecker
from scripts.diagnose_closed_loop_failures import first_blocked_index
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint


def global_prediction(local, state):
    """独立展开局部→世界变换，保留网络输出精度的舍入语义。"""
    c, s = np.cos(state[2]), np.sin(state[2])
    result = np.empty_like(local)
    result[:, 0] = state[0] + c * local[:, 0] - s * local[:, 1]
    result[:, 1] = state[1] + s * local[:, 0] + c * local[:, 1]
    result[:, 2] = state[2] + local[:, 2]
    return result


def local_goal(goal, state):
    dx, dy, angle = np.asarray(goal) - np.asarray(state)[:3]
    c, s = np.cos(state[2]), np.sin(state[2])
    return [c * dx + s * dy, -s * dx + c * dy, np.arctan2(np.sin(angle), np.cos(angle))]


def transform_matches(local, state, world):
    # 原接口float32计算会在标量/数组运算处舍入；独立展开的计算精度不同。
    # 按输入精度和坐标尺度取8 epsilon，禁止以逐位不同制造边界故障。
    expected = global_prediction(local, state)
    scale = np.maximum(1., np.maximum(np.abs(world), np.abs(np.asarray(state)[:3])))
    tolerance = 8 * np.finfo(local.dtype).eps * scale
    return bool(np.all(np.abs(expected.astype(np.float64) - world) <= tolerance))


def analyze_episode(capture_dir, index, baseline, data, manifest, vehicle, loaded, budget):
    source = Path(capture_dir) / f'episode-{index:04d}'
    row = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    from scripts.diagnose_closed_loop_failures import _sha256
    if not row['baseline_match'] or _sha256(source.with_suffix('.npz')) != row['npz_sha256']:
        raise ValueError('采集复现或NPZ身份无效')
    restored = matrix.reconstruct_dataset_task(data['task_meta'][index], root_seed=manifest['seed'], vehicle=vehicle)
    planner, pipeline = matrix._build_pipeline(matrix.EXPERIMENT_SPECS['E5'], restored.task, vehicle)
    pipeline.set_target_goals([restored.goal])
    errors = {name: [] for name in ('L1', 'L2', 'L3', 'L1_L2', 'L2_L3')}
    errors['L2_L3'].extend(row['boundary_errors'])
    planning = planner._collision_checker
    mpc = MPCController(dt=.1, horizon=10, seed=baseline['protocol']['control_seed'] + index, **vehicle.mpc_kwargs())
    mpc.reset()
    terminal = TerminalChecker(restored.tol_pos, restored.tol_yaw)
    plan_rows, tracking_rows, plans = [], [], []

    def fail(layer, kind, **details):
        errors[layer].append({'error': kind, **details})

    with np.load(source.with_suffix('.npz'), allow_pickle=False) as tape:
        steps = np.asarray(tape['plan_issue_steps'])
        states = np.asarray(tape['plan_issue_states'])
        before, after = np.asarray(tape['states_before']), np.asarray(tape['states_after'])
        bevs, goal = np.asarray(tape['issue_bevs']), np.asarray(tape['goal'])
        if len(steps) != len(row['inputs']) or len(before) != len(row['frames']) or not len(before):
            raise ValueError('计划/输入/控制采集不完整，不能作通过证据')
        if not np.array_equal(steps, np.arange(0, len(before), baseline['protocol']['replan_every'])):
            fail('L2_L3', 'replan_schedule')
        if not np.array_equal(before[0], tape['initial_state']) or not np.array_equal(before[1:], after[:-1]):
            fail('L2_L3', 'state_continuity')
        for ordinal, (step, state, bev_data, model_input) in enumerate(zip(steps, states, bevs, row['inputs'], strict=True)):
            budget.check()
            step = int(step)
            metadata = model_input['bev_meta']
            bev = BEVTensor(bev_data, metadata['resolution'], metadata['extent'], metadata['channels'],
                            camera_projection=metadata.get('camera_projection'))
            independent = pipeline.capture_bev(*state[:3])
            if not np.isfinite(bev_data).all() or not np.array_equal(independent.data, bev_data) or independent.to_metadata() != metadata:
                fail('L1', 'production_sensor_replay', plan=ordinal)
            if len(bev.channels) != loaded.model_config['bev_channels']:
                fail('L1_L2', 'channel_count', plan=ordinal)
            for key in ('target_channel', 'height_mode'):
                if model_input[key] != loaded.model_config.get(key, 'image' if key == 'target_channel' else 'installation'):
                    fail('L1_L2', 'model_input_contract', field=key, observed=model_input[key],
                         expected=loaded.model_config.get(key), plan=ordinal)
            if not model_input['capture_to_predict_equal']:
                fail('L1_L2', 'bev_changed_at_handoff', plan=ordinal)
            if not np.array_equal(model_input['capture_pose'], state[:3]) or not np.array_equal(model_input['state'], state):
                fail('L1_L2', 'pose_changed_at_handoff', plan=ordinal)
            expected_goal = local_goal(goal, state)
            if not np.allclose(model_input['goal'], expected_goal, atol=1e-12, rtol=0):
                fail('L1_L2', 'goal_frame', plan=ordinal)
            if not np.array_equal(state, before[step]):
                fail('L2_L3', 'plan_issued_from_stale_state', plan=ordinal)
            lidar_truth = rasterize_lidar_truth_occupancy(restored.task.scene.env, *state[:3],
                                                        pipeline.bev_config, max_range=pipeline.lidar_sensor.max_range)
            truth_target = rasterize_ground_truth_target(restored.goal, vehicle.length, vehicle.width,
                                                         *state[:3], pipeline.bev_config)
            fidelity = compute_bev_fidelity_metrics(bev, lidar_truth, truth_target).to_dict()
            # 旧MAE把语义高度与二值occupancy相比，不适用于语义height验收。
            fidelity.pop('height_mae')
            fidelity.pop('density_mae')
            local = np.asarray(tape[f'local_prediction_{ordinal:04d}'])
            world = np.asarray(tape[f'global_plan_{ordinal:04d}'])
            dt = float(tape[f'plan_dt_{ordinal:04d}'])
            prediction = loaded.model.predict(bev, GoalPose(*model_input['goal']), VehicleState.from_array(state))
            if not np.array_equal(prediction.points, local) or prediction.dt != dt:
                fail('L2', 'production_network_replay', plan=ordinal)
            if not np.isfinite(local).all() or not np.isfinite(world).all():
                fail('L2', 'nonfinite_reference', plan=ordinal)
            if not transform_matches(local, state, world):
                fail('L2_L3', 'local_global_transform', plan=ordinal)
            if dt != loaded.model_config['dt'] or model_input['dt'] != dt or float(tape['control_dt']) != .1:
                fail('L2_L3', 'reference_control_dt', plan=ordinal)
            if not planning.pose_free(*world[0, :3]):
                internal_blocked = 0
            else:
                blocked_offset = first_blocked_index(planning, world[0], world[1:])
                internal_blocked = None if blocked_offset is None else blocked_offset + 1
            endpoint_pos, endpoint_yaw = pose_errors(world[-1], goal)
            endpoint_reached = endpoint_pos < terminal.tol_pos and endpoint_yaw < terminal.tol_yaw
            first_gap, first_yaw_gap = pose_errors(world[0], state)
            previous_offset = None
            if plans:
                previous = plans[-1].points
                nearest = int(np.argmin(np.linalg.norm(previous[:, :2] - state[:2], axis=1)))
                previous_offset = list(pose_errors(world[0], previous[nearest]))
            plan_rows.append({'ordinal': ordinal, 'step': step, 'input_fidelity': fidelity,
                              'points': len(world), 'dt': dt, 'endpoint_position_m': endpoint_pos,
                              'endpoint_yaw_rad': endpoint_yaw, 'endpoint_reached': endpoint_reached,
                              'full_reference_planning_blocked_index': internal_blocked,
                              'reference_motion_demand': trajectory_demands(world, dt, max_v=vehicle.max_v, max_omega=vehicle.max_omega),
                              'first_reference_gap_position_m': first_gap, 'first_reference_gap_yaw_rad': first_yaw_gap,
                              'new_first_to_previous_nearest_xy_m_yaw_rad': previous_offset,
                              'transform_max_absolute_error': float(np.max(np.abs(global_prediction(local, state).astype(np.float64) - world))),
                              'goal_distance_m': pose_errors(state, goal)[0]})
            plans.append(Trajectory(world.copy(), dt=dt))
        for step, frame in enumerate(row['frames']):
            budget.check()
            state = VehicleState.from_array(before[step])
            trajectory = plans[frame['plan_ordinal']]
            command = mpc.compute(trajectory, state)
            window = mpc._reference_points(mpc._aligned_cache, mpc._progress_idx)
            if not np.array_equal([command.v, command.omega], frame['command']):
                fail('L3', 'production_mpc_replay', step=step)
            if (mpc._progress_idx != frame['progress_index'] or not np.array_equal(window, tape['control_windows'][step])
                    or not np.array_equal(mpc._last_u.reshape(-1, 2), tape['selected_control_sequences'][step])):
                fail('L2_L3', 'mpc_window_or_context', step=step)
            # 首预测点是下一控制时刻；最近点误差另外保留，禁止用小中心误差替代姿态与车体检查。
            aligned_xy, aligned_yaw = pose_errors(after[step], window[0])
            nearest = int(np.argmin(np.linalg.norm(window[:, :2] - after[step, :2], axis=1)))
            near_xy, near_yaw = pose_errors(after[step], window[nearest])
            tracking_rows.append({'step': step, 'aligned_xy_m': aligned_xy, 'aligned_yaw_rad': aligned_yaw,
                                  'nearest_xy_m': near_xy, 'nearest_yaw_rad': near_yaw})
        final = VehicleState.from_array(after[-1])
        reached = terminal.reached(final, restored.goal)
        actual = row['actual_result']
        if bool(actual['success']) != (reached and not actual['collision'] and actual['failure'] is None):
            fail('L2_L3', 'terminal_outcome')
        if not np.isclose(terminal.pos_err(final, restored.goal), actual['final_pos_err'], atol=1e-9, rtol=0):
            fail('L2_L3', 'terminal_position')
        if not np.isclose(terminal.yaw_err(final, restored.goal), actual['final_yaw_err'], atol=1e-9, rtol=0):
            fail('L2_L3', 'terminal_yaw')
    contact = row['collision_frame']
    reference_unsafe = any(f['contact']['reference_internal_blocked_index'] is not None for f in row['frames'])
    bridge_unsafe = any(not f['contact']['connection_safe'] for f in row['frames'])
    execution_unsafe = any(not f['contact']['actual_transition_safe'] for f in row['frames'])
    contact_signature = None if contact is None else locate_failure(
        success=False, collision=True, boundary_errors=[],
        reference_unsafe=contact['contact']['reference_internal_blocked_index'] is not None,
        connection_unsafe=not contact['contact']['connection_safe'],
        execution_unsafe=not contact['contact']['actual_transition_safe'])
    boundary = errors['L1_L2'] + errors['L2_L3']
    layers = {name: layer_status(items) for name, items in errors.items()}
    layers['L2']['quality_status'] = 'observed_reference_risk' if reference_unsafe or bridge_unsafe else 'no_contact_risk_in_consumed_windows'
    layers['L3']['quality_status'] = 'failed_contact_safety' if execution_unsafe else 'no_observed_contact'
    layers['L3']['unsafe_selected_rollout_steps'] = sum(f['contact']['mpc_rollout_blocked_index'] is not None for f in row['frames'])
    layers['L2']['planning_margin_risk_plans'] = sum(p['full_reference_planning_blocked_index'] is not None for p in plan_rows)
    layers['L1_L2']['quality_status'] = 'failed_input_contract' if errors['L1_L2'] else 'same_input_replayed'
    layers['L2_L3']['quality_status'] = 'failed_contact_or_nonarrival' if not actual['success'] else 'task_reached'
    return {'dataset_index': index, 'scene_name': row['scene_name'], **actual,
            'initial_dataset_bev_equal': bool(np.array_equal(bevs[0], data['bevs'][index])),
            'baseline_match': row['baseline_match'], 'terminal': row['terminal'], 'layers': layers,
            'plans': plan_rows, 'controls': tracking_rows,
            'reference_contact_risk': reference_unsafe, 'state_reference_bridge_risk': bridge_unsafe,
            'actual_contact_risk': execution_unsafe, 'contact_signature': contact_signature,
            'first_events': row['first_events'], 'collision_frame': contact,
            'nonarrival': None if actual['success'] else {
                'native_failure': actual['failure'], 'initial_goal_distance_m': plan_rows[0]['goal_distance_m'],
                'final_goal_distance_m': actual['final_pos_err'], 'final_yaw_rad': actual['final_yaw_err'],
                'endpoint_misses': sum(not p['endpoint_reached'] for p in plan_rows),
                'replans': len(plan_rows), 'last_plan_issue_step': plan_rows[-1]['step']},
            'observed_signature': locate_failure(success=actual['success'], collision=actual['collision'],
                                                 boundary_errors=boundary, reference_unsafe=reference_unsafe,
                                                 connection_unsafe=bridge_unsafe, execution_unsafe=execution_unsafe),
            'limits': '独立重放检验实现与接口一致；checkpoint声明与生产语义不符不单独证明训练/部署分布错配，应同时查看initial_dataset_bev_equal和数据生成身份。有限访问状态及几何事件不能证明唯一根因、可达性或全局安全。'}


def analyze_captures(capture_dir, baseline, indices, budget):
    data = DatasetGenerator.load(baseline['protocol']['data'])
    manifest = matrix.load_dataset_manifest(Path(baseline['protocol']['data']))
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    if vehicle.to_metadata() != baseline['vehicle_model']:
        raise ValueError('车辆与基线不一致')
    loaded = load_model_checkpoint(baseline['protocol']['checkpoint'])
    if loaded.model_config != baseline['protocol']['model_config'] or loaded.model_name != baseline['protocol']['model_name']:
        raise ValueError('模型配置与基线不一致')
    rows = []
    for index in indices:
        rows.append(analyze_episode(capture_dir, index, baseline, data, manifest, vehicle, loaded, budget))
        print(f'analyze [{len(rows)}/{len(indices)}] #{index} {rows[-1]["observed_signature"]}', flush=True)
    return rows
