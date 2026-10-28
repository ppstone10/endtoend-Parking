"""60秒验证唯一早段脉冲从任务起点的完整效果；无搜索/训练/控制器改动。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.behavior_exploration_data import run_case, save_record
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.recovery_execution import record_arrays
from interfaces import Trajectory
from runtime.sources import NetworkSource
from scripts import search_s8_start_references as search
from scripts.diagnose_s8_start_tracking import FORMAL, require
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.prepare_recovery_balance_ab import file_digest
from training.reporting import atomic_write_json

DIAGNOSTIC = ROOT / 'runs/diagnostics/s8-start-tracking-diagnostic-20261023-v1'
HASHES = {'report.json': '79babfb7973d29548d99c07912d9e5a9907d177ba6f59b26d8065f2c3ce79de1',
          'config.json': 'dfa544a41c640b12c80db1f948f621779f8f1df24aed461c081111d3d49fbe6c'}
EXPECTED = (2351, 'candidate', 2, 11, .01)


def select_candidate(report):
    eligible = []
    for prefix in report['prefixes']:
        event = prefix['event_diagnostic']
        if event is None:
            continue
        for response in event['responses']:
            if response['amplitude'] and all(response[key] for key in
                    ('reference_internal_safe', 'event_window_connected_safe', 'actual_step_safe')):
                eligible.append((prefix['index'], prefix['strategy'], prefix['control_seed'],
                                 prefix['event'], response['amplitude']))
    require(eligible == [EXPECTED], '唯一局部安全见证身份/数量漂移')
    return eligible[0]


def prepare_config(budget):
    for name, digest in HASHES.items():
        require(file_digest(DIAGNOSTIC / name) == digest, '起步诊断身份不符')
    report = json.loads((DIAGNOSTIC / 'report.json').read_text(encoding='utf-8'))
    require(report['status'] == 'completed' and report['complete_prefixes'] == 6
            and report['complete_events'] == 3 and not report['search_performed'], '起步诊断未完整')
    select_candidate(report)
    config = json.loads((DIAGNOSTIC / 'config.json').read_text(encoding='utf-8'))
    verify_config(config, budget)
    config = deepcopy(config)
    config['frozen_files'].update({str(DIAGNOSTIC / name): digest for name, digest in HASHES.items()})
    config['frozen_files'].update({str(artifact_path(DIAGNOSTIC, name)): digest
                                   for name, digest in report['artifacts_sha256'].items()})
    config['code'][str(Path(__file__).resolve())] = file_digest(__file__)
    config.update(kind='s8_single_start_pulse_validation_v1', candidate=list(EXPECTED),
                  validation_seconds=60, maximum_complete_episodes=7)
    verify_config(config, budget)
    return config


def validate_reference(case, base, original, reference):
    require(reference.dt == base.dt == .2 and reference.points.shape == base.points.shape
            and np.isfinite(reference.points).all()
            and np.array_equal(reference.points[:, :2], base.points[:, :2])
            and np.array_equal(reference.points[[0, -1]], base.points[[0, -1]]), '脉冲破坏固定参考边界')
    delta = reference.points[:, 2].astype(np.float64) - original.points[:, 2].astype(np.float64)
    maximum = float(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta)))))
    require(maximum <= .1 + 1e-7, '总航向偏移超限')
    safe, reason = case['planner']._collision_checker.check_trajectory(case['start'], reference)
    require(safe, f'脉冲参考起点连接/几何不安全：{reason}')
    return dict(max_total_yaw_offset=maximum, original_start_and_reference_safe=True)


def first_difference(left, right):
    size = min(len(left), len(right))
    changed = np.flatnonzero(np.any(left[:size] != right[:size], axis=1))
    return int(changed[0]) if len(changed) else size if len(left) != len(right) else None


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(60); torch.set_num_threads(1)
    report = dict(status='running', paired=[], training_updates=0, teacher_bank_created=False,
        learning_benefit_tested=False, allowed_to_prepare_training_plan=False, automatic_deployment=False,
        main_protocol_passed=False, search_performed=False, source_qualified=False, executed_episodes=0)
    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_write_json(output / 'report.json', report)
    with exploration_lock(output):
        save()
        try:
            config = prepare_config(budget); atomic_write_json(output / 'config.json', config)
            report['config_sha256'] = file_digest(output / 'config.json')
            data, sources = search.load_sources()
            _, case, _, base, original = next(s for s in sources if s[0]['index'] == 2351)
            source_dir = FORMAL / 'source-2351'
            with np.load(source_dir / 'eval-0001/execution.npz') as stored:
                old_reference = Trajectory(stored['reference'].copy(), float(stored['dt']))
            probe = DIAGNOSTIC / '2351-candidate-control2/probe-+0.01.npz'
            with np.load(probe) as stored:
                reference = Trajectory(stored['reference'].copy(), old_reference.dt)
            report['reference_check'] = validate_reference(case, base, original, reference)

            def execute(name, trajectory, seed, archive=None):
                budget.check()
                result, engine, frames = run_case(case, None, budget, trajectory=trajectory, control_seed=seed)
                states, commands = record_arrays(result)
                if archive is not None:
                    with np.load(archive) as stored:
                        for key, value in (('reference', trajectory.points), ('states', states), ('commands', commands),
                                           ('margin_violations', np.asarray(engine.margin_violations))):
                            require(np.array_equal(value, stored[key]), f'原候选精确重放失败：{name}/{key}')
                save_record(output, name, result, reference=trajectory.points, dt=np.asarray(trajectory.dt),
                    margin_violations=np.asarray(engine.margin_violations), progress_trace=np.asarray(engine.mpc.execution_trace))
                report['executed_episodes'] += 1
                row = dict(name=name, control_seed=seed, **search.result_row(case, result, engine))
                print('EPISODE', name, row['steps'], row['margin_violation_steps'], flush=True)
                return row, states, commands, frames

            def pair(seed):
                archive = source_dir / ('eval-0001/execution.npz' if seed == 0 else f'candidate-control{seed}.npz')
                baseline, old_states, old_commands, _ = execute(f'baseline-control{seed}', old_reference, seed, archive)
                candidate, states, commands, frames = execute(f'pulse-control{seed}', reference, seed)
                row = dict(control_seed=seed, baseline_exact=True, baseline=baseline, candidate=candidate,
                    first_command_difference=first_difference(old_commands, commands),
                    first_state_difference=first_difference(old_states, states),
                    original_event_step_safe=bool(case['planner']._collision_checker.swept_segment_free(
                        states[10, :3], states[11, :3])) if candidate['steps'] > 11 else None)
                report['paired'].append(row); save()
                return candidate, states, frames

            candidate2, states2, frames2 = pair(2)
            report['decision'] = 'local_safe_step_failed_full_start_transfer'
            if candidate2['reached_with_margin']:
                candidate0, states0, frames0 = pair(0)
                candidate1, _, _ = pair(1)
                report['decision'] = 'single_start_pulse_failed_other_control_streams'
                if candidate0['reached_with_margin'] and candidate1['reached_with_margin']:
                    sample, window0 = search.check_first_window(case, data, frames0, states0, budget)
                    _, window2 = search.check_first_window(case, data, frames2, states2, budget)
                    report['first_windows'] = [window0, window2]
                    exported = Trajectory(NetworkSource._to_global(np.asarray(sample['points'])[:frames0[0]['length']],
                                                                    case['start']), reference.dt)
                    export, _, _, _ = execute('float32-export-control0', exported, 0)
                    report['float32_export'] = export
                    report['source_qualified'] = bool(window0['clean'] and window2['clean'] and export['reached_with_margin'])
                    report['decision'] = ('single_source_witness_not_two_source_or_learning_qualification'
                        if report['source_qualified'] else 'single_start_pulse_failed_window_or_export')
                    torch.save(dict(purpose='diagnostic_not_teacher_bank', frame=frames0[0]), output / 'first-frame.pt')
            require(report['executed_episodes'] <= 7, '固定完整回合预算超限')
            verify_config(config, budget); budget.check()
            report.update(status='completed', stage='decision')
            report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                if p.is_file() and p != output / 'report.json' and p.name != 'run.lock'}
            save(); print('FINISHED', report['decision'], report['elapsed_seconds'], flush=True)
            return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else 'failed',
                          error=str(exc), decision='incomplete_do_not_infer'); save(); raise


def precheck():
    config = prepare_config(PreflightBudget(45))
    print('START PULSE INPUTS VERIFIED; ZERO ROLLOUT/SEARCH/TRAINING', flush=True)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precheck', action='store_true'); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.precheck:
        if args.output: parser.error('precheck不写output')
        precheck()
    elif args.output is None: parser.error('需要独立新output目录')
    else: run(args.output)
