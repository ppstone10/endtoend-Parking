"""单碰撞回合有界回放与同输入预测比较；不产生候选闭环分数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import validation_matrix
from runtime.sources import NetworkSource
from scripts.analyze_prefix_kinematics import prefix_metrics
from scripts.diagnose_closed_loop_failures import _sha256, first_blocked_index
from scripts.prepare_recovery_balance_ab import execution_code_identity, runtime_identity
from training.checkpoint import load_model_checkpoint
from training.data import to_local
from training.reporting import atomic_write_json


def select_collision(report, index):
    if report.get('status') != 'completed' or report['experiment']['name'] != 'E5' or report['experiment']['safety_mode'] != 'none':
        raise ValueError('要求已完成无干预 E5 报告')
    rows = [row for row in report['episodes'] if row['meta']['dataset_index'] == index]
    if len(rows) != 1 or not rows[0]['collision'] or rows[0]['success']:
        raise ValueError('索引必须唯一对应一条碰撞任务')
    if not 0 < rows[0]['steps'] <= 600:
        raise ValueError('单次诊断仅支持 1..600 步')
    return rows[0]


def compare_models(samples, checkpoints, config, goal, checker):
    bev = torch.as_tensor(np.stack([s['bev'] for s in samples]), dtype=torch.float32)
    local_goals = np.stack([to_local(np.array([[goal.x, goal.y, goal.yaw]]), s['state'].x, s['state'].y, s['state'].yaw)[0] for s in samples])
    motion = torch.tensor([[s['state'].v, s['state'].omega] for s in samples], dtype=torch.float32)
    comparisons = {}
    for name, path in checkpoints.items():
        loaded = load_model_checkpoint(path)
        if loaded.model_config != config:
            raise ValueError('比较模型配置/阈值必须一致')
        loaded.model.eval()
        with torch.no_grad():
            prediction = loaded.model.forward_with_stop(bev, torch.as_tensor(local_goals, dtype=torch.float32), motion)
        points = prediction.points.numpy()
        stop = prediction.stop_logits.sigmoid().numpy()
        rows = []
        for j, sample in enumerate(samples):
            crossing = np.flatnonzero(stop[j] >= loaded.model.stop_threshold)
            length = int(crossing[0]) + 1 if len(crossing) else points.shape[1]
            prefix = points[j, :min(20, length)]
            global_prefix = NetworkSource._to_global(prefix, sample['state'])
            rows.append({'step': sample['step'], 'state': sample['state'].to_array().tolist(),
                         'local_goal': local_goals[j].tolist(), 'predicted_length': length,
                         'prefix': prefix.tolist(),
                         'prefix_contact_blocked_index': first_blocked_index(checker, sample['state'].to_array(), global_prefix),
                         'kinematics': prefix_metrics(prefix, dt=loaded.model.dt) if len(prefix) >= 2 else None})
        comparisons[name] = {'checkpoint': str(Path(path).resolve()), 'sha256': _sha256(Path(path)), 'rows': rows}
    return comparisons


def run_audit(report_path, index, checkpoints, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    expected = select_collision(report, index)
    protocol = report['protocol']
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    original = validation_matrix.ClosedLoopEngine
    samples, captured = [], {}

    class CaptureEngine(original):
        def run(self, start, goal):
            capture = self.source.sensor_pipeline.capture_bev
            predict = self.source.next_trajectory
            compute = self.mpc.compute
            latest, steps = {}, [0]

            def capture_bev(*args, **kwargs):
                bev = capture(*args, **kwargs)
                latest['bev'] = np.array(bev.data, copy=True)
                return bev

            def next_trajectory(state):
                result = predict(state)
                samples.append({'step': steps[0], 'state': state, 'bev': latest['bev']})
                return result

            def compute_command(*args, **kwargs):
                command = compute(*args, **kwargs)
                steps[0] += 1
                return command

            self.source.sensor_pipeline.capture_bev = capture_bev
            self.source.next_trajectory = next_trajectory
            self.mpc.compute = compute_command
            try:
                result = super().run(start, goal)
            finally:
                self.source.sensor_pipeline.capture_bev = capture
                self.source.next_trajectory = predict
                self.mpc.compute = compute
            captured.update(result=result, goal=goal, checker=self.collision_checker)
            return result

    validation_matrix.ClosedLoopEngine = CaptureEngine
    try:
        replay = validation_matrix.run_validation_experiment(
            'E5', data_path=protocol['data'], checkpoint_path=protocol['checkpoint'], indices=[index],
            max_steps=expected['steps'], replan_every=protocol['replan_every'], control_seed=protocol['control_seed'])
    finally:
        validation_matrix.ClosedLoopEngine = original
    actual = replay['episodes'][0]
    matched = all(actual[key] == expected[key] for key in ('success', 'collision', 'failure', 'steps')) and all(np.isclose(actual[key], expected[key], rtol=0, atol=1e-9) for key in ('final_pos_err', 'final_yaw_err'))
    if not matched:
        atomic_write_json(output / 'mismatch.json', {'expected': expected, 'actual': actual})
        raise ValueError('回放未复现，禁止继续归因')
    np.savez_compressed(output / 'issued-inputs.npz', states=np.stack([s['state'].to_array() for s in samples]),
                        bevs=np.stack([s['bev'] for s in samples]), steps=np.array([s['step'] for s in samples]))
    result = {'kind': 'diagnostic_same_state_not_acceptance', 'dataset_index': index, 'baseline_match': matched,
              'code': execution_code_identity(), 'audit_script_sha256': _sha256(Path(__file__)), 'runtime': runtime_identity(),
              'issued_inputs_sha256': _sha256(output / 'issued-inputs.npz'),
              'report': str(report_path), 'report_sha256': _sha256(report_path),
              'data_sha256': _sha256(Path(protocol['data'])), 'replay_checkpoint_sha256': _sha256(Path(protocol['checkpoint'])),
              'replay_protocol': replay['protocol'], 'actual': actual,
              'comparisons': compare_models(samples, checkpoints, protocol['model_config'], captured['goal'], captured['checker']),
              'limits': '单失败回合的同状态预测，不是反事实闭环结果；20点前缀不等于实际控制窗口。'}
    atomic_write_json(output / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--index', required=True, type=int)
    parser.add_argument('--checkpoint', action='append', required=True, help='名称=路径（名称不能重复）')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    checkpoints = dict(item.split('=', 1) for item in args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        parser.error('模型名称不能重复')
    result = run_audit(args.report, args.index, checkpoints, args.output)
    print(json.dumps({'baseline_match': result['baseline_match'], 'states': len(next(iter(result['comparisons'].values()))['rows'])}))


if __name__ == '__main__':
    main()
