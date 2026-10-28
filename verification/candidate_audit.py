"""独立重算候选选择：使用普通MPC原采样，不调用候选控制器求解。"""
from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np

from controller import MPCController
from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from interfaces import ControlCmd, Trajectory, VehicleState
from planner import RectangleFootprintCollisionChecker
from scripts.diagnose_closed_loop_failures import _sha256
from sim import VehicleConfig, DifferentialDriveModel
from training.reporting import atomic_write_json
from verification.candidate_study import precheck, FILES
from verification.identity import execution_identity


def audit_control(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('独立审计使用新目录')
    config = precheck(source / 'config.json')
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed' or report['mode'] != 'smoke':
        raise ValueError('要求完整固定短检')
    artifacts = json.loads((source / 'artifacts.json').read_text(encoding='utf-8'))
    for name, digest in artifacts.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or _sha256(path) != digest:
            raise ValueError('短检产物身份不一致')
    data = DatasetGenerator.load(config['data'])
    manifest = matrix.load_dataset_manifest(Path(config['data']))
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    vehicle_model = DifferentialDriveModel(**vehicle.vehicle_model_kwargs())
    code = execution_identity([*FILES, 'verification/candidate_audit.py', 'scripts/audit_footprint_execution.py'])
    started = time.perf_counter()
    summaries = []
    for index in report['indices']:
        if time.perf_counter() - started > 90:
            raise TimeoutError('独立短检审计超过预算')
        restored = matrix.reconstruct_dataset_task(data['task_meta'][index], root_seed=manifest['seed'], vehicle=vehicle)
        checker = RectangleFootprintCollisionChecker(restored.task.scene.env, vehicle_length=vehicle.length,
                                                     vehicle_width=vehicle.width, collision_margin=0.,
                                                     resolution=vehicle.collision_check_resolution)
        margin = RectangleFootprintCollisionChecker(restored.task.scene.env, vehicle_length=vehicle.length,
                                                    vehicle_width=vehicle.width, collision_margin=.2,
                                                    resolution=vehicle.collision_check_resolution)
        native = MPCController(dt=.1, horizon=10, seed=index, **vehicle.mpc_kwargs())
        bank = []
        cost_function = native._batch_cost

        def collect(samples, ref, state):
            costs = cost_function(samples, ref, state)
            bank.append((samples.copy(), costs.copy()))
            return costs

        native._batch_cost = collect
        record = json.loads((source / 'control' / f'episode-{index:04d}.json').read_text(encoding='utf-8'))
        with np.load(source / 'control' / f'episode-{index:04d}.npz', allow_pickle=False) as tape:
            state = VehicleState.from_array(tape['initial_state'])
            issue_steps = list(map(int, tape['plan_issue_steps']))
            plans = {step: Trajectory(tape[f'plan_{ordinal:04d}'], dt=float(tape[f'plan_dt_{ordinal:04d}']))
                     for ordinal, step in enumerate(issue_steps)}

            def safe(sequence, initial):
                if not checker.pose_free(initial.x, initial.y, initial.yaw):
                    return False
                previous = initial
                for v, omega in np.asarray(sequence).reshape(-1, 2):
                    after = vehicle_model.step(previous, ControlCmd(float(v), float(omega)), .1)
                    if not checker.swept_segment_free(previous.to_array()[:3], after.to_array()[:3]):
                        return False
                    previous = after
                return True

            selected_counts = {}
            margin_violations = 0
            for step, (before, after, command, selected, frame) in enumerate(zip(
                    tape['states_before'], tape['states_after'], tape['commands'],
                    tape['selected_control_sequences'], record['frames'], strict=True)):
                if step % 50 == 0 and time.perf_counter() - started > 90:
                    raise TimeoutError('独立短检审计超过预算')
                if not np.array_equal(before, state.to_array()):
                    raise ValueError('归档实际状态不连续')
                if step in plans:
                    active = plans[step]
                shifted = native._shift(native._last_u.copy())
                bank.clear()
                native.compute(active, state)
                mean = native._last_u.copy()
                ref = native._reference_points(native._aligned_cache, native._progress_idx)
                if not np.array_equal(ref, tape['control_windows'][step]) or native._progress_idx != frame['progress']:
                    raise ValueError('普通MPC消费上下文未精确复现')
                mean_safe = safe(mean, state)
                if mean_safe:
                    label, chosen = 'legacy_mean', mean
                else:
                    choices = []
                    samples = np.concatenate([s for s, _ in bank])
                    costs = np.concatenate([c for _, c in bank])
                    for ordinal in np.argsort(costs, kind='stable')[:16]:
                        if safe(samples[ordinal], state):
                            choices.append(('safe_sample', samples[ordinal], float(costs[ordinal])))
                            break
                    for candidate_label, sequence in [('shifted_warm_start', shifted), ('zero_hold', np.zeros_like(shifted))]:
                        if safe(sequence, state):
                            choices.append((candidate_label, sequence, float(cost_function(sequence[None, :], ref, state)[0])))
                    if not choices:
                        raise ValueError('归档声称有安全选择，但独立未找到')
                    label, chosen, _ = min(choices, key=lambda item: item[2])
                selected_flat = np.asarray(selected).reshape(-1)
                if not np.array_equal(selected_flat, chosen) or label != frame['selected'] or mean_safe != frame['legacy_safe']:
                    raise ValueError('独立安全候选选择与归档不一致')
                if not safe(selected_flat, state):
                    raise ValueError('实际选择的完整预测车体不安全')
                selected_cost = float(cost_function(selected_flat[None, :], ref, state)[0])
                mean_cost = float(cost_function(mean[None, :], ref, state)[0])
                if selected_cost != frame['selected_cost'] or mean_cost != frame['legacy_cost']:
                    raise ValueError('独立成本不一致')
                native._last_u = selected_flat.copy()
                selected_counts[label] = selected_counts.get(label, 0) + 1
                predicted = vehicle_model.step(state, ControlCmd(*map(float, command)), .1)
                if not np.array_equal(predicted.to_array(), after) or not checker.swept_segment_free(before[:3], after[:3]):
                    raise ValueError('真实动作/车体安全未复现')
                margin_violations += not margin.swept_segment_free(before[:3], after[:3])
                state = predicted
            if margin_violations != record['summary']['planning_margin_violations']:
                raise ValueError('规划余量统计不一致')
            summaries.append({'dataset_index': index, 'steps': len(record['frames']),
                              'selected_counts': selected_counts, 'all_choices_costs_windows_and_states_exact': True,
                              'all_selected_horizons_contact_safe': True, 'planning_margin_violations': margin_violations})
    output.mkdir(parents=True, exist_ok=False)
    payload = {'status': 'completed', 'source_report_sha256': _sha256(source / 'report.json'),
               'code': code, 'episodes': summaries, 'elapsed_seconds': time.perf_counter() - started,
               'method': '普通MPC重新采样/拟合，独立几何约束及原成本排序，未调用FootprintConstrainedMPC求解',
               'limits': '只核验固定四回合，不代表全300或0.2m规划余量全程安全。'}
    atomic_write_json(output / 'report.json', payload)
    return payload
