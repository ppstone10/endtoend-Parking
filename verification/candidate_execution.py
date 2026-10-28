"""显式四臂消融：原网络/参考过滤 × 原MPC/完整车体约束MPC。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import numpy as np

from experiments import validation_matrix as matrix
from interfaces import ControlCmd
from runtime import GeometricFilterSource, MpcVehicleExecutor
from training.reporting import atomic_write_json
from verification.safety_candidate import FootprintConstrainedMPC

ARMS = ('native', 'control', 'reference', 'combined')


def connected_safe(checker, start, points):
    before = np.asarray(start)[:3]
    if not checker.pose_free(*before):
        return False
    for point in points:
        if not checker.swept_segment_free(before, np.asarray(point)[:3]):
            return False
        before = np.asarray(point)[:3]
    return True


@contextmanager
def candidate_scope(arm, output, vehicle, expected, budget):
    """替换只限本进程本次显式运行，任何异常都恢复原工厂。"""
    if arm not in ARMS:
        raise ValueError('未知候选臂')
    original_engine = matrix.ClosedLoopEngine
    stats = []

    class AuditEngine(original_engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if arm in ('control', 'combined'):
                self.mpc = FootprintConstrainedMPC(contact_checker=self.collision_checker,
                                                  dt=self.mpc.dt, horizon=self.mpc.horizon,
                                                  seed=self.meta['dataset_index'], **vehicle.mpc_kwargs())
                self.executor = MpcVehicleExecutor(self.mpc, self.vehicle_model)
            if arm in ('reference', 'combined'):
                projector = matrix.build_filter_projector(self.env, vehicle,
                                                          required_margin_m=vehicle.collision_margin,
                                                          max_offset_m=1., max_rounds=10)
                self.source = GeometricFilterSource(self.source, projector)
            from planner import RectangleFootprintCollisionChecker
            self.margin_checker = RectangleFootprintCollisionChecker(
                self.env, vehicle_length=vehicle.length, vehicle_width=vehicle.width,
                collision_margin=vehicle.collision_margin, resolution=vehicle.collision_check_resolution)

        def run(self, start, goal):
            frames, plans = [], []
            compute, predict = self.mpc.compute, self.source.next_trajectory

            def next_plan(state):
                budget.check()
                trajectory, elapsed = predict(state)
                budget.check()
                plans.append({'step': len(frames), 'points': trajectory.points.copy(), 'dt': trajectory.dt})
                return trajectory, elapsed

            def command(trajectory, state):
                budget.check()
                cmd = compute(trajectory, state)
                sequence = self.mpc._last_u.copy().reshape(-1, 2)
                predicted, current = [], state
                for v, omega in sequence:
                    current = self.vehicle_model.step(current, ControlCmd(float(v), float(omega)), self.mpc.dt)
                    predicted.append(current.to_array())
                window = self.mpc._reference_points(self.mpc._aligned_cache, self.mpc._progress_idx).copy()
                audit = dict(self.mpc.last_safety_audit) if isinstance(self.mpc, FootprintConstrainedMPC) else {
                    'selected': 'legacy_mean', 'intervened': False,
                    'selected_horizon_safe': connected_safe(self.collision_checker, state.to_array(), predicted)}
                if arm in ('control', 'combined') and not audit['selected_horizon_safe']:
                    raise ValueError('约束控制器输出未通过完整时域扫掠')
                frames.append({'before': state.to_array(), 'window': window, 'sequence': sequence,
                               'rollout': np.asarray(predicted), 'command': [cmd.v, cmd.omega],
                               'progress': self.mpc._progress_idx, 'plan': len(plans) - 1, 'audit': audit})
                return cmd

            self.mpc.compute, self.source.next_trajectory = command, next_plan
            try:
                result = super().run(start, goal)
            finally:
                self.mpc.compute, self.source.next_trajectory = compute, predict
            if len(frames) != result.steps:
                raise ValueError('真实控制步分母不完整')
            details, previous = [], start.to_array()
            for step, (frame, state, collision) in enumerate(zip(frames, result.record.states, result.record.collisions, strict=True)):
                after = state.to_array()
                if not np.array_equal(previous, frame['before']) or not np.array_equal(after, frame['rollout'][0]):
                    raise ValueError('控制预测首步和实际推进不精确一致')
                actual_safe = connected_safe(self.collision_checker, previous, [after])
                if bool(collision) != (not actual_safe):
                    raise ValueError('原生碰撞与独立实际扫掠不一致')
                margin_safe = connected_safe(self.margin_checker, previous, [after])
                window_safe = connected_safe(self.collision_checker, previous, frame['window'])
                details.append({'step': step, 'plan': frame['plan'], 'progress': frame['progress'],
                                'actual_contact_safe': actual_safe, 'planning_margin_safe': margin_safe,
                                'reference_contact_safe': window_safe, **frame['audit']})
                previous = after
            index = self.meta['dataset_index']
            row = {'dataset_index': index, 'scene_name': self.meta['scene_name'],
                   **{k: getattr(result, k) for k in ('success', 'failure', 'collision', 'steps', 'final_pos_err', 'final_yaw_err')},
                   'intervened_steps': sum(d['intervened'] for d in details),
                   'zero_hold_steps': sum(d['selected'] == 'zero_hold' for d in details),
                   'unsafe_selected_horizons': sum(not d['selected_horizon_safe'] for d in details),
                   'planning_margin_violations': sum(not d['planning_margin_safe'] for d in details),
                   'unsafe_reference_windows': sum(not d['reference_contact_safe'] for d in details),
                   'reference_filter': self.source.filter_stats() if hasattr(self.source, 'filter_stats') else None}
            if arm == 'native':
                baseline = expected[index]
                if any(row[k] != baseline[k] for k in ('success', 'failure', 'collision', 'steps', 'final_pos_err', 'final_yaw_err')):
                    raise ValueError(f'#{index}原生基线未精确复现')
                row['baseline_exact'] = True
            output_path = Path(output) / f'episode-{index:04d}'
            arrays = {'initial_state': start.to_array(), 'goal': np.array([goal.x, goal.y, goal.yaw]),
                      'states_before': np.asarray([f['before'] for f in frames]),
                      'states_after': np.asarray([s.to_array() for s in result.record.states]),
                      'commands': np.asarray([f['command'] for f in frames]),
                      'control_windows': np.asarray([f['window'] for f in frames]),
                      'selected_control_sequences': np.asarray([f['sequence'] for f in frames]),
                      'plan_issue_steps': np.asarray([p['step'] for p in plans]), 'control_dt': np.asarray(self.mpc.dt)}
            for ordinal, plan in enumerate(plans):
                arrays[f'plan_{ordinal:04d}'] = plan['points']
                arrays[f'plan_dt_{ordinal:04d}'] = np.array(plan['dt'])
            np.savez_compressed(output_path.with_suffix('.npz'), **arrays)
            atomic_write_json(output_path.with_suffix('.json'), {'summary': row, 'frames': details})
            stats.append(row)
            return result

    matrix.ClosedLoopEngine = AuditEngine
    try:
        yield stats
    finally:
        matrix.ClosedLoopEngine = original_engine


def compare_rows(native, candidate):
    expected = {r['dataset_index']: r for r in native}
    if len(expected) != len(native) or set(expected) != {r['dataset_index'] for r in candidate} or len(native) != len(candidate):
        raise ValueError('配对分母缺失或重复')
    old_lost = [r['dataset_index'] for r in candidate if expected[r['dataset_index']]['success'] and not r['success']]
    gained = [r['dataset_index'] for r in candidate if not expected[r['dataset_index']]['success'] and r['success']]
    return {'samples': len(candidate), 'successes': sum(r['success'] for r in candidate),
            'collisions': sum(r['collision'] for r in candidate), 'old_success_lost': old_lost, 'new_success_gained': gained,
            'zero_hold_steps': sum(r['zero_hold_steps'] for r in candidate),
            'planning_margin_violations': sum(r['planning_margin_violations'] for r in candidate),
            'intervened_steps': sum(r['intervened_steps'] for r in candidate),
            'net_success_change': len(gained) - len(old_lost),
            'pilot_benefit': bool(gained and not old_lost and sum(r['collision'] for r in candidate) < sum(r['collision'] for r in native))}
