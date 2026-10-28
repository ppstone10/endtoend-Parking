"""原生 E5 的只读观察：同一实际运行生成所有分层验证输入。"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np

from experiments import validation_matrix as matrix
from interfaces import ControlCmd
from scripts.diagnose_reference_execution import save_episode
from training.reporting import atomic_write_json


class VerificationBudget:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.seconds = seconds

    def check(self):
        if time.perf_counter() - self.started > self.seconds:
            raise TimeoutError('验证预算耗尽；保留部分产物，不自动续跑')


def capture_native_e5(baseline, output, indices, budget):
    """进程内串行观察；finally 恢复工厂。不改控制器、参考或传感器随机流。"""
    from experiments.component_verification import validate_baseline
    expected = validate_baseline(baseline, indices)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original_engine, original_builder = matrix.ClosedLoopEngine, matrix._build_source
    checkers, rows = [], []

    def build_source(*args, **kwargs):
        checkers.append(args[1]._collision_checker)
        return original_builder(*args, **kwargs)

    class ObservedEngine(original_engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.planning_checker = checkers.pop()

        def run(self, start, goal):
            frames, plans, inputs, latest = [], [], [], {}
            pipeline, model = self.source.sensor_pipeline, self.source.model
            compute, predict = self.mpc.compute, self.source.next_trajectory
            capture, local_predict = pipeline.capture_bev, model.predict
            boundary_errors = []

            def capture_bev(x, y, yaw):
                budget.check()
                bev = capture(x, y, yaw)
                latest.update(bev=np.array(bev.data, copy=True), bev_meta=bev.to_metadata(), pose=[x, y, yaw])
                return bev

            def predict_local(bev, target, state):
                trajectory = local_predict(bev, target, state)
                inputs.append({'state': state.to_array().tolist(), 'goal': [target.x, target.y, target.yaw],
                               'capture_pose': latest['pose'], 'bev_meta': latest['bev_meta'],
                               'capture_to_predict_equal': bool(np.array_equal(bev.data, latest['bev'])),
                               'target_channel': 'geometry' if pipeline.target_override else 'image',
                               'height_mode': 'semantic', 'dt': float(trajectory.dt)})
                latest['local'] = np.array(trajectory.points, copy=True)
                return trajectory

            def predict_global(state):
                trajectory, elapsed = predict(state)
                plans.append({'step': len(frames), 'state': state.to_array(),
                              'global': np.array(trajectory.points, copy=True), 'local': latest['local'],
                              'bev': latest['bev'], 'dt': trajectory.dt, 'object': trajectory})
                return trajectory, elapsed

            def compute_command(trajectory, state):
                budget.check()
                if not plans or trajectory is not plans[-1]['object']:
                    boundary_errors.append({'step': len(frames), 'error': 'source_to_mpc_reference_object'})
                command = compute(trajectory, state)
                window = self.mpc._reference_points(self.mpc._aligned_cache, self.mpc._progress_idx)
                controls = self.mpc._last_u.copy().reshape(-1, 2)
                rollout, predicted = [], state
                for v, omega in controls:
                    predicted = self.vehicle_model.step(predicted, ControlCmd(float(v), float(omega)), self.mpc.dt)
                    rollout.append(predicted.to_array())
                frames.append({'before': state.to_array(), 'window': window.copy(), 'u': controls,
                               'rollout': np.asarray(rollout), 'progress': self.mpc._progress_idx,
                               'aligned_length': len(self.mpc._aligned_cache), 'plan_ordinal': len(plans) - 1})
                return command

            pipeline.capture_bev, model.predict = capture_bev, predict_local
            self.source.next_trajectory, self.mpc.compute = predict_global, compute_command
            try:
                result = super().run(start, goal)
            finally:
                pipeline.capture_bev, model.predict = capture, local_predict
                self.source.next_trajectory, self.mpc.compute = predict, compute
            index = self.meta['dataset_index']
            path = output / f'episode-{index:04d}'
            row = save_episode(path, self, start, goal, result, frames, plans)
            row.update(inputs=inputs, boundary_errors=boundary_errors,
                       terminal={'tol_pos': self.terminal.tol_pos, 'tol_yaw': self.terminal.tol_yaw},
                       actual_result={k: getattr(result, k) for k in ('success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')})
            original = expected[index]
            row['baseline_match'] = (all(original[k] == getattr(result, k) for k in ('success', 'collision', 'failure', 'steps'))
                                     and all(np.isclose(original[k], getattr(result, k), atol=1e-9, rtol=0)
                                             for k in ('final_pos_err', 'final_yaw_err')))
            atomic_write_json(path.with_suffix('.json'), row)
            if not row['baseline_match']:
                raise ValueError(f'#{index}未复现基线，停止归因')
            rows.append(row)
            return result

    matrix.ClosedLoopEngine, matrix._build_source = ObservedEngine, build_source
    protocol = baseline['protocol']
    try:
        replay = matrix.run_validation_experiment(
            'E5', data_path=protocol['data'], checkpoint_path=protocol['checkpoint'],
            indices=indices, samples=0, max_steps=protocol['max_steps'], replan_every=protocol['replan_every'],
            control_seed=protocol['control_seed'], output_path=output / 'replayed-report.json',
            progress=lambda n, total, bundle: print(f'capture [{n}/{total}] #{bundle.result.meta["dataset_index"]} matched', flush=True))
    finally:
        matrix.ClosedLoopEngine, matrix._build_source = original_engine, original_builder
    if replay['reconstruct_failures'] or len(rows) != len(indices):
        raise ValueError('采集分母不完整')
    return rows
