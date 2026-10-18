"""隔离 MPC 求解器反事实；不改变默认控制器，不作为主协议准入结果。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller import MPCController
from experiments import validation_matrix
from scripts.diagnose_closed_loop_failures import _sha256
from training.reporting import atomic_write_json


class IncumbentMPC(MPCController):
    """只改变候选保留/选择：不丢弃热启动、均值和已经找到的低代价解。"""

    def _solve(self, u_mean, ref, state):
        best = np.array(u_mean, copy=True)
        best[0::2] = np.clip(best[0::2], -self.max_v, self.max_v)
        best[1::2] = np.clip(best[1::2], -self.max_omega, self.max_omega)
        best_cost = float(self._batch_cost(best[None], ref, state)[0])
        std = np.tile([self.max_v * .5, self.max_omega * .8], self.horizon)
        n_elite = max(2, int(round(self.population * self.elite_frac)))
        for _ in range(self.iterations):
            samples = self._rng.normal(size=(self.population, 2 * self.horizon)) * std + u_mean
            samples[:, 0::2] = np.clip(samples[:, 0::2], -self.max_v, self.max_v)
            samples[:, 1::2] = np.clip(samples[:, 1::2], -self.max_omega, self.max_omega)
            # 不增加 population；随机数抽取量与原 CEM 相同。
            samples[-2:] = np.stack([u_mean, best])
            costs = self._batch_cost(samples, ref, state)
            elite_indices = np.argsort(costs)[:n_elite]
            if costs[elite_indices[0]] < best_cost:
                best = samples[elite_indices[0]].copy()
                best_cost = float(costs[elite_indices[0]])
            elites = samples[elite_indices]
            u_mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-3
            mean_cost = float(self._batch_cost(u_mean[None], ref, state)[0])
            if mean_cost < best_cost:
                best, best_cost = u_mean.copy(), mean_cost
        return best


class CorrelatedMPC(MPCController):
    """只改变采样噪声：每个控制通道使用 rho=.8 的平稳 AR(1)。"""

    correlation = .8

    def _sample_noise(self, population):
        noise = self._rng.normal(size=(population, self.horizon, 2))
        scale = np.sqrt(1. - self.correlation ** 2)
        for step in range(1, self.horizon):
            noise[:, step] = self.correlation * noise[:, step - 1] + scale * noise[:, step]
        return noise.reshape(population, 2 * self.horizon)

    def _solve(self, u_mean, ref, state):
        std = np.tile([self.max_v * .5, self.max_omega * .8], self.horizon)
        n_elite = max(2, int(round(self.population * self.elite_frac)))
        for _ in range(self.iterations):
            samples = self._sample_noise(self.population) * std + u_mean
            samples[:, 0::2] = np.clip(samples[:, 0::2], -self.max_v, self.max_v)
            samples[:, 1::2] = np.clip(samples[:, 1::2], -self.max_omega, self.max_omega)
            costs = self._batch_cost(samples, ref, state)
            elites = samples[np.argsort(costs)[:n_elite]]
            u_mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-3
        return u_mean


CONTROLLERS = {'baseline': MPCController, 'incumbent': IncumbentMPC, 'correlated': CorrelatedMPC}


@contextmanager
def isolated_controller(candidate):
    """仅供单进程实验；无论评测是否抛异常均恢复工厂。"""
    replacement = CONTROLLERS[candidate]
    original = validation_matrix.MPCController
    original_engine = validation_matrix.ClosedLoopEngine

    class ProbeEngine(original_engine):
        def run(self, start, goal):
            result = super().run(start, goal)
            # 矩阵评测在回调前释放 record，故在此只提取小型统计。
            result.meta['solver_probe_commands'] = command_metrics(result.record)
            return result

    validation_matrix.MPCController = replacement
    validation_matrix.ClosedLoopEngine = ProbeEngine
    try:
        yield
    finally:
        validation_matrix.MPCController = original
        validation_matrix.ClosedLoopEngine = original_engine


def validate_probe_indices(baseline, indices):
    allowed = set(baseline['protocol']['evaluated_indices'])
    if not indices or len(indices) != len(set(indices)) or not set(indices) <= allowed:
        raise ValueError('实验索引必须唯一、非空且属于基线')
    return indices


def command_metrics(record):
    commands = np.asarray([[cmd.v, cmd.omega] for cmd in record.cmds]).reshape(-1, 2)
    delta = np.diff(commands, axis=0)
    moving = commands[np.abs(commands[:, 0]) > .05, 0]
    return {
        'mean_abs_delta_v': float(np.abs(delta[:, 0]).mean()) if len(delta) else 0.,
        'mean_abs_delta_omega': float(np.abs(delta[:, 1]).mean()) if len(delta) else 0.,
        'velocity_sign_flips': int(np.count_nonzero(moving[1:] * moving[:-1] < 0.)),
    }


def run_probe(baseline_path: Path, output: Path, *, candidate: str, indices=None):
    baseline_path, output = baseline_path.resolve(), output.resolve()
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if baseline.get('status') != 'completed' or baseline['experiment']['name'] != 'E5' or baseline['experiment']['safety_mode'] != 'none':
        raise ValueError('反事实要求已完成无干预 E5 报告')
    if baseline['overall']['reconstruct_failures']:
        raise ValueError('基线任务复原失败，不能缩小分母')
    if candidate not in CONTROLLERS:
        raise ValueError('未知候选控制器')
    protocol = baseline['protocol']
    if protocol['mpc_dt'] != .1:
        raise ValueError('当前矩阵固定控制周期为0.1，不能复用不同控制周期的基线')
    selected = validate_probe_indices(baseline, protocol['evaluated_indices'] if indices is None else indices)
    if output.exists():
        raise FileExistsError('实验目录已存在，请使用新目录')
    output.mkdir(parents=True)
    identity = {
        'kind': 'diagnostic_controller_counterfactual', 'candidate': candidate,
        'baseline': str(baseline_path), 'data_sha256': _sha256(Path(protocol['data'])),
        'model_sha256': _sha256(Path(protocol['checkpoint'])), 'indices': selected,
        'obstacle_objective': False, 'expert_fallback': False, 'main_acceptance': False,
        'deployment_modified': False, 'correlation': .8 if candidate == 'correlated' else None,
        'probe_source_sha256': _sha256(Path(__file__)),
        'controller_source_sha256': _sha256(Path('controller/mpc.py')),
        'unchanged': ['network', 'reference', 'cost_function', 'limits', 'horizon', 'dt', 'progress', 'population', 'iterations'],
    }
    atomic_write_json(output / 'identity.json', identity)
    old_rows = {row['meta']['dataset_index']: row for row in baseline['episodes']}
    paired = []

    def progress(ordinal, total, bundle):
        result = bundle.result
        old = old_rows[result.meta['dataset_index']]
        if old['meta']['task_id'] != result.meta['task_id']:
            raise ValueError('任务身份与基线不一致')
        paired.append({
            'dataset_index': result.meta['dataset_index'], 'task_id': result.meta['task_id'],
            'scene_name': result.meta['scene_name'], 'task_identity_match': old['meta']['task_id'] == result.meta['task_id'],
            'baseline_success': old['success'], 'baseline_collision': old['collision'], 'baseline_failure': old['failure'],
            'success': result.success, 'collision': result.collision, 'failure': result.failure,
            'baseline_exact_match': all(old[key] == getattr(result, key) for key in ('success', 'collision', 'failure', 'steps'))
                and bool(np.isclose(old['final_pos_err'], result.final_pos_err, atol=1e-9, rtol=0)),
            **result.meta['solver_probe_commands'],
        })
        print(f'[{ordinal}/{total}] #{result.meta["dataset_index"]} {old["failure"] or "success"} -> {result.failure or "success"}', flush=True)

    with isolated_controller(candidate):
        report = validation_matrix.run_validation_experiment(
            'E5', data_path=protocol['data'], checkpoint_path=protocol['checkpoint'],
            indices=selected, samples=0, max_steps=protocol['max_steps'], replan_every=protocol['replan_every'],
            control_seed=protocol['control_seed'], progress=progress,
        )
    report['probe'] = identity
    report['protocol']['controller_candidate'] = candidate
    atomic_write_json(output / 'report.json', report)
    summary = {
        'identity': identity, 'overall': report['overall'], 'pairs': paired,
        'recovered_successes': [row['dataset_index'] for row in paired if not row['baseline_success'] and row['success']],
        'lost_successes': [row['dataset_index'] for row in paired if row['baseline_success'] and not row['success']],
        'baseline_exact_mismatches': [row['dataset_index'] for row in paired if not row['baseline_exact_match']],
    }
    atomic_write_json(output / 'paired.json', summary)
    print(json.dumps(report['overall'], ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=CONTROLLERS, required=True)
    parser.add_argument('--indices-file', type=Path)
    args = parser.parse_args()
    indices = None
    if args.indices_file:
        payload = json.loads(args.indices_file.read_text(encoding='utf-8'))
        indices = payload['indices'] if isinstance(payload, dict) else payload
    run_probe(args.baseline_report, args.output, candidate=args.candidate, indices=indices)


if __name__ == '__main__':
    main()
