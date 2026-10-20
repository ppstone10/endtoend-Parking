"""只观察原E5：分开参考内部、状态桥接与真实执行风险。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import validation_matrix
from interfaces import ControlCmd
from scripts.diagnose_closed_loop_failures import _sha256, first_blocked_index
from scripts.prepare_recovery_balance_ab import runtime_identity
from training.reporting import atomic_write_json


def select_indices(report, indices):
    if report.get('status') != 'completed' or report['experiment']['name'] != 'E5' or report['experiment']['safety_mode'] != 'none':
        raise ValueError('要求已完成无干预E5报告')
    available = [e['meta']['dataset_index'] for e in report['episodes']]
    if len(available) != len(set(available)) or not indices or len(indices) != len(set(indices)) or not set(indices) <= set(available):
        raise ValueError('必须显式指定非空、唯一且属于报告的任务索引')
    return list(indices)


def window_risks(before, window, after, rollout, checker):
    """内部风险不包含当前车体到参考首点的桥接；索引以实际窗口为基准。"""
    if not checker.pose_free(*window[0, :3]):
        blocked = 0
    else:
        offset = first_blocked_index(checker, window[0], window[1:])
        blocked = None if offset is None else offset + 1
    return {'state_free': bool(checker.pose_free(*before[:3])),
            'reference_internal_blocked_index': blocked,
            'connection_safe': bool(checker.swept_segment_free(before[:3], window[0, :3])),
            'actual_transition_safe': bool(checker.swept_segment_free(before[:3], after[:3])),
            'mpc_rollout_blocked_index': first_blocked_index(checker, before, rollout)}


def classify_contact(risk):
    if not risk['state_free']:
        return 'state_already_outside_contact_free_set'
    if risk['reference_internal_blocked_index'] is not None:
        return 'reference_internal_unsafe'
    if not risk['connection_safe']:
        return 'state_to_reference_connection_unsafe'
    if not risk['actual_transition_safe']:
        return 'execution_left_contact_safe_reference'
    if risk['mpc_rollout_blocked_index'] is not None:
        return 'selected_control_rollout_unsafe_with_safe_reference'
    return 'no_contact_risk_in_checked_window'


def code_identity():
    paths = []
    for folder in ('controller', 'sim', 'runtime', 'planner', 'experiments', 'sensor2bev', 'perception', 'sensors', 'model', 'training', 'dataset', 'interfaces', 'metrics', 'bev'):
        paths.extend((ROOT / folder).rglob('*.py'))
    paths.extend([Path(__file__), ROOT / 'scripts/diagnose_closed_loop_failures.py', ROOT / 'scripts/prepare_recovery_balance_ab.py'])
    return {str(p.resolve()): _sha256(p) for p in sorted(set(paths))}


def summarize_frames(rows):
    events = {}
    for margin in ('contact', 'planning'):
        predicates = {'reference_internal': lambda r: r['reference_internal_blocked_index'] is not None,
                      'connection': lambda r: not r['connection_safe'],
                      'actual_transition': lambda r: not r['actual_transition_safe'],
                      'selected_control_rollout': lambda r: r['mpc_rollout_blocked_index'] is not None}
        for name, predicate in predicates.items():
            event = next((r for r in rows if predicate(r[margin])), None)
            events[f'{margin}_{name}'] = None if event is None else {
                'step': event['step'], 'previous_step': event['step'] - 1 if event['step'] else None,
                'plan_issue_step': event['plan_issue_step'], 'plan_ordinal': event['plan_ordinal']}
    return events


def save_episode(output, engine, start, goal, result, frames, plans):
    record = result.record
    if len(record.states) != len(frames):
        raise ValueError('实际控制与推进记录未一一对齐')
    rows = []
    before = start.to_array()
    for step, (state, frame) in enumerate(zip(record.states, frames, strict=True)):
        after = state.to_array()
        if not np.allclose(frame['before'], before, atol=1e-12, rtol=0):
            raise ValueError('compute前态与真实前态不一致')
        if not np.allclose(frame['rollout'][0], after, atol=1e-12, rtol=0):
            raise ValueError('选中控制rollout首步与真实推进不一致')
        risks = {name: window_risks(before, frame['window'], after, frame['rollout'], checker)
                 for name, checker in [('contact', engine.collision_checker), ('planning', engine.planning_checker)]}
        contact = risks['contact']
        if record.collisions[step] != (not contact['actual_transition_safe']):
            raise ValueError('真实碰撞与转移重算不一致')
        plan = plans[frame['plan_ordinal']]
        # 对齐数组的等时间网格与原网络点时间一一映射；末点补齐时索引也钳制。
        aligned_indices = np.minimum(frame['progress'] + np.arange(len(frame['window'])), frame['aligned_length'] - 1)
        source_indices = (aligned_indices * (len(plan['global']) - 1) / max(1, frame['aligned_length'] - 1)).tolist()
        rows.append({'step': step, 'plan_ordinal': frame['plan_ordinal'], 'plan_issue_step': plan['step'],
                     'progress_index': frame['progress'], 'source_fractional_indices': source_indices,
                     'before': before.tolist(), 'after': after.tolist(),
                     'command': [record.cmds[step].v, record.cmds[step].omega],
                     'distance_to_window_first_m': float(np.linalg.norm(before[:2] - frame['window'][0, :2])),
                     'distance_to_nearest_window_point_m': float(np.linalg.norm(before[:2] - frame['window'][:, :2], axis=1).min()),
                     'contact_signature': classify_contact(contact), **risks})
        before = after
    arrays = {'initial_state': start.to_array(), 'goal': np.array([goal.x, goal.y, goal.yaw]),
              'states_before': np.asarray([f['before'] for f in frames]),
              'states_after': np.asarray([s.to_array() for s in record.states]),
              'control_windows': np.asarray([f['window'] for f in frames]),
              'selected_control_sequences': np.asarray([f['u'] for f in frames]),
              'selected_control_rollouts': np.asarray([f['rollout'] for f in frames]),
              'plan_issue_states': np.asarray([p['state'] for p in plans]),
              'plan_issue_steps': np.asarray([p['step'] for p in plans]),
              'issue_bevs': np.asarray([p['bev'] for p in plans]), 'control_dt': np.array(engine.mpc.dt)}
    for ordinal, plan in enumerate(plans):
        arrays[f'global_plan_{ordinal:04d}'] = plan['global']
        arrays[f'local_prediction_{ordinal:04d}'] = plan['local']
        arrays[f'plan_dt_{ordinal:04d}'] = np.array(plan['dt'])
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    return {'dataset_index': engine.meta['dataset_index'], 'scene_name': engine.meta['scene_name'],
            'first_events': summarize_frames(rows), 'frames': rows,
            'collision_frame': rows[-1] if result.collision else None,
            'npz_sha256': _sha256(output.with_suffix('.npz')),
            'limits': '几何事件签名不是充分因果；rollout是选中序列的开环预测，不是未来重规划后的真实轨迹；安全几何不保证动力学可跟踪。'}


def run_diagnosis(report_path, output, indices):
    started = time.perf_counter()
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    indices = select_indices(report, indices)
    protocol = report['protocol']
    expected = {e['meta']['dataset_index']: e for e in report['episodes']}
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    identity = {'report': str(report_path), 'report_sha256': _sha256(report_path),
                'data': protocol['data'], 'data_sha256': _sha256(Path(protocol['data'])),
                'checkpoint': protocol['checkpoint'], 'checkpoint_sha256': _sha256(Path(protocol['checkpoint'])),
                'protocol': protocol, 'indices': indices, 'code': code_identity(), 'runtime': runtime_identity()}
    atomic_write_json(output / 'identity.json', identity)
    original_engine = validation_matrix.ClosedLoopEngine
    original_builder = validation_matrix._build_source
    planning_checkers, diagnostics = [], []

    def build_source(*args, **kwargs):
        planning_checkers.append(args[1]._collision_checker)
        return original_builder(*args, **kwargs)

    class CaptureEngine(original_engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.planning_checker = planning_checkers.pop()

        def run(self, start, goal):
            frames, plans, latest = [], [], {}
            compute, predict = self.mpc.compute, self.source.next_trajectory
            capture, local_predict = self.source.sensor_pipeline.capture_bev, self.source.model.predict

            def capture_bev(*args, **kwargs):
                bev = capture(*args, **kwargs)
                latest['bev'] = np.array(bev.data, copy=True)
                return bev

            def predict_local(*args, **kwargs):
                trajectory = local_predict(*args, **kwargs)
                latest['local'] = np.array(trajectory.points, copy=True)
                return trajectory

            def predict_global(state):
                trajectory, elapsed = predict(state)
                plans.append({'step': len(frames), 'state': state.to_array(), 'global': np.array(trajectory.points, copy=True),
                              'local': latest['local'], 'bev': latest['bev'], 'dt': trajectory.dt})
                return trajectory, elapsed

            def compute_command(trajectory, state):
                command = compute(trajectory, state)
                window = self.mpc._reference_points(self.mpc._aligned_cache, self.mpc._progress_idx)
                u = self.mpc._last_u.copy().reshape(-1, 2)
                rollout, predicted_state = [], state
                for v, omega in u:
                    predicted_state = self.vehicle_model.step(predicted_state, ControlCmd(float(v), float(omega)), self.mpc.dt)
                    rollout.append(predicted_state.to_array())
                frames.append({'before': state.to_array(), 'window': window.copy(), 'u': u,
                               'rollout': np.asarray(rollout), 'progress': self.mpc._progress_idx,
                               'aligned_length': len(self.mpc._aligned_cache), 'plan_ordinal': len(plans) - 1})
                return command

            self.mpc.compute, self.source.next_trajectory = compute_command, predict_global
            self.source.sensor_pipeline.capture_bev, self.source.model.predict = capture_bev, predict_local
            try:
                result = super().run(start, goal)
            finally:
                self.mpc.compute, self.source.next_trajectory = compute, predict
                self.source.sensor_pipeline.capture_bev, self.source.model.predict = capture, local_predict
            index = self.meta['dataset_index']
            row = save_episode(output / f'episode-{index:04d}', self, start, goal, result, frames, plans)
            baseline = expected[index]
            row['baseline_match'] = all(baseline[k] == getattr(result, k) for k in ('success', 'collision', 'failure', 'steps')) and all(np.isclose(baseline[k], getattr(result, k), atol=1e-9, rtol=0) for k in ('final_pos_err', 'final_yaw_err'))
            row['actual_result'] = {k: getattr(result, k) for k in ('success', 'collision', 'failure', 'steps', 'final_pos_err', 'final_yaw_err')}
            atomic_write_json(output / f'episode-{index:04d}.json', row)
            if not row['baseline_match']:
                raise ValueError(f'#{index}未复现基线，保留原始证据，停止归因')
            diagnostics.append(row)
            return result

    validation_matrix.ClosedLoopEngine, validation_matrix._build_source = CaptureEngine, build_source
    try:
        replay = validation_matrix.run_validation_experiment(
            'E5', data_path=protocol['data'], checkpoint_path=protocol['checkpoint'], indices=indices, samples=0,
            max_steps=protocol['max_steps'], replan_every=protocol['replan_every'], control_seed=protocol['control_seed'],
            output_path=output / 'replayed-report.json',
            progress=lambda n, total, bundle: print(f'[{n}/{total}] #{bundle.result.meta["dataset_index"]} matched', flush=True))
    finally:
        validation_matrix.ClosedLoopEngine, validation_matrix._build_source = original_engine, original_builder
    if replay['overall']['reconstruct_failures'] or len(diagnostics) != len(indices) or code_identity() != identity['code']:
        raise ValueError('诊断不完整或执行代码已变化')
    payload = {'status': 'completed', 'kind': 'diagnostic_reference_execution_not_acceptance', 'identity': identity,
               'elapsed_sec': time.perf_counter() - started,
               'episodes': [{k: v for k, v in row.items() if k != 'frames'} for row in diagnostics]}
    atomic_write_json(output / 'report.json', payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--indices', required=True, help='逗号分隔的明确任务索引；允许成功对照')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run_diagnosis(args.report, args.output, [int(s) for s in args.indices.split(',')])
    print(json.dumps({'status': result['status'], 'elapsed_sec': result['elapsed_sec']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
