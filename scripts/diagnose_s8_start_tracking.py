"""90秒固定六前缀/三首次侵入事件诊断，不搜索、训练或改变原控制器。"""
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

from controller import MPCController
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.reference_compensation import reference_internal_safe
from experiments.s8_start_diagnostic import predict_sequences, swept_flags, summarize_iteration
from experiments.s8_yaw_response import probe_command
from interfaces import Trajectory, VehicleState
from scripts import search_s8_start_references as search
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.prepare_recovery_balance_ab import file_digest
from training.reporting import atomic_write_json

FORMAL = ROOT / 'runs/diagnostics/s8-start-reference-search-20261023-v1'
CONFIG_SHA = '8077359c541ae34f3f9f258e6052b2998e13db189e8a42efb0736b783b49dcb8'
REPORT_SHA = 'da4f876ee9395a85b4be5c0e746bdf835a0671775bd40e6a532354742ea42af1'
PREFIXES = (
    (2338, 'baseline', 0, 'source-2338/eval-0000/execution.npz', 5),
    (2338, 'best', 0, 'source-2338/eval-0129/execution.npz', 8),
    (2351, 'baseline', 0, 'source-2351/eval-0000/execution.npz', None),
    (2351, 'candidate', 0, 'source-2351/eval-0001/execution.npz', None),
    (2351, 'baseline', 2, 'source-2351/baseline-control2.npz', None),
    (2351, 'candidate', 2, 'source-2351/candidate-control2.npz', 11))
STEPS = 24


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_config(budget):
    require(file_digest(FORMAL / 'config.json') == CONFIG_SHA
            and file_digest(FORMAL / 'report.json') == REPORT_SHA, 'LT20正式身份不符')
    config = deepcopy(search.checked_config(FORMAL / 'config.json', budget))
    report = json.loads((FORMAL / 'report.json').read_text(encoding='utf-8'))
    require(report['status'] == 'completed' and not report['qualification_passed'], '不是已完成的LT20负结果')
    for name, digest in report['artifacts_sha256'].items():
        config['frozen_files'][str(artifact_path(FORMAL, name))] = digest
    config['frozen_files'].update({str(FORMAL / 'config.json'): CONFIG_SHA, str(FORMAL / 'report.json'): REPORT_SHA})
    from experiments import s8_start_diagnostic
    for path in (Path(__file__).resolve(), Path(s8_start_diagnostic.__file__).resolve()):
        config['code'][str(path)] = file_digest(path)
    config.update(kind='s8_start_tracking_diagnostic_v1', diagnostic=dict(prefixes=list(PREFIXES), steps=STEPS,
        seconds=90, amplitudes=[0., -.01, .01], purpose='diagnostic_not_new_teacher_or_controller'))
    verify_config(config, budget)
    return config


def event_diagnostic(case, snapshot, sequence, captures, command, expected_next, budget, directory):
    mpc, trajectory, state = snapshot
    checker = case['planner']._collision_checker
    zero = probe_command(mpc, trajectory, state, 0.)
    require(np.array_equal([zero['command'].v, zero['command'].omega], command), '原事件命令不能精确复现')
    next_state = case['engine_kwargs']['vehicle_model'].step(state, zero['command'], mpc.dt)
    require(np.array_equal(next_state.to_array(), expected_next), '原事件下一状态不能精确复现')
    predicted = predict_sequences(mpc, state, sequence[None])
    require(np.array_equal(predicted[0, 0], expected_next[:3]), '预测首步与原模型不一致')
    mean_flags = swept_flags(state.to_array(), predicted, checker, budget.check)[0]
    iterations = []
    for number, (samples, costs) in enumerate(captures):
        points = predict_sequences(mpc, state, samples)
        flags = swept_flags(state.to_array(), points, checker, budget.check)
        iterations.append(dict(iteration=number, **summarize_iteration(samples, costs, flags, mpc.elite_frac)))
        np.savez_compressed(directory / f'cem-{number}.npz', samples=samples, costs=costs, points=points, unsafe=flags)
    require(len(captures) == mpc.iterations, 'CEM迭代分母不符')
    samples, costs = captures[-1]
    elite = np.argsort(costs)[:max(2, int(round(len(samples) * mpc.elite_frac)))]
    require(np.array_equal(samples[elite].mean(axis=0), sequence), '原输出不是最后一轮精英均值')
    responses = []
    for amplitude in (0., -.01, .01):
        budget.check()
        response = zero if amplitude == 0. else probe_command(mpc, trajectory, state, amplitude)
        next_state = case['engine_kwargs']['vehicle_model'].step(state, response['command'], mpc.dt)
        local = mpc._reference_points(mpc._align_dt(response['trajectory']), response['progress_index'])
        risks = swept_flags(state.to_array(), local[None], checker, budget.check)[0]
        row = dict(amplitude=amplitude, command=[response['command'].v, response['command'].omega],
            next_state=next_state.to_array().tolist(), progress_index=response['progress_index'],
            reference_internal_safe=reference_internal_safe(response['trajectory'], checker),
            event_window_connected_safe=not bool(risks.any()),
            actual_step_safe=checker.swept_segment_free(state.to_array()[:3], next_state.to_array()[:3]))
        responses.append(row)
        np.savez_compressed(directory / f'probe-{amplitude:+.2f}.npz', reference=response['trajectory'].points,
                            basis=response['basis'], command=row['command'], next_state=next_state.to_array())
    fields = ('dt', 'horizon', 'max_v', 'max_omega', 'w_xy', 'w_yaw', 'w_terminal', 'w_v', 'w_omega',
              'population', 'elite_frac', 'iterations')
    atomic_write_json(directory / 'snapshot.json', dict(parameters={key: getattr(mpc, key) for key in fields},
        catchup_window=mpc._catchup_window, rng=mpc._rng.bit_generator.state, progress_before=mpc._progress_idx))
    np.savez_compressed(directory / 'snapshot.npz', state=state.to_array(), reference=trajectory.points,
        dt=np.asarray(trajectory.dt), last_u=mpc._last_u, aligned_cache=mpc._aligned_cache,
        expected_command=command, expected_next_state=expected_next, output_sequence=sequence)
    return dict(original_replayed_exact=True, original_mean_prediction_unsafe=mean_flags.tolist(),
                cem_iterations=iterations, responses=responses)


def replay_prefix(case, archive, control_seed, event, budget, directory):
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    require(len(arrays['states']) >= STEPS, '固定前缀不足24步')
    if event is not None:
        require(int(np.flatnonzero(arrays['margin_violations'])[0]) == event, '首次侵入事件身份不符')
    trajectory = Trajectory(arrays['reference'], float(arrays['dt']))
    mpc = MPCController(dt=.1, horizon=10, seed=case['index'] + 10000 * control_seed, **case['vehicle'].mpc_kwargs())
    state = deepcopy(case['start']); states = []; commands = []; windows = []; sequences = []; rows = []; evidence = None
    checker = case['planner']._collision_checker
    for step in range(STEPS):
        budget.check()
        before = mpc._progress_idx
        captures = []
        snapshot = deepcopy((mpc, trajectory, state)) if step == event else None
        if snapshot is not None:
            native = mpc._batch_cost
            def capture(samples, reference, current):
                costs = native(samples, reference, current)
                captures.append((samples.copy(), costs.copy()))
                return costs
            mpc._batch_cost = capture
        try:
            command = mpc.compute(trajectory, state)
        finally:
            if snapshot is not None:
                del mpc._batch_cost
        next_state = case['engine_kwargs']['vehicle_model'].step(state, command, mpc.dt)
        cmd = np.asarray([command.v, command.omega]); point = next_state.to_array()
        require(np.array_equal(cmd, arrays['commands'][step]) and np.array_equal(point, arrays['states'][step]),
                f'固定前缀漂移：step{step}')
        progress = mpc._progress_idx
        if 'progress_trace' in arrays:
            require(np.array_equal([progress, len(mpc._aligned_cache)], arrays['progress_trace'][step]), '进度/缓存长度漂移')
        window = mpc._reference_points(mpc._aligned_cache, progress)
        current = state.to_array()[:3]
        actual_safe = checker.swept_segment_free(current, point[:3])
        require(actual_safe == (not bool(arrays['margin_violations'][step])), '实际扫掠与归档不一致')
        nearest = int(np.argmin(np.sum((mpc._aligned_cache[:, :2] - current[:2]) ** 2, axis=1)))
        yaw_error = current[2] - window[0, 2]
        window_flags = swept_flags(current, window[None], checker, budget.check)[0]
        mean_flags = swept_flags(current, predict_sequences(mpc, state, mpc._last_u[None]), checker, budget.check)[0]
        rows.append(dict(step=step, progress_before=before, progress=progress, nearest_xy_index=nearest,
            progress_ahead_of_nearest=progress-nearest, reference_xy_error=float(np.linalg.norm(current[:2]-window[0, :2])),
            reference_yaw_error=float(np.arctan2(np.sin(yaw_error), np.cos(yaw_error))),
            current_pose_safe=checker.pose_free(*current), actual_step_safe=actual_safe,
            connected_reference_window_safe=not bool(window_flags.any()),
            mean_horizon_safe=not bool(mean_flags.any()), command=cmd.tolist()))
        if snapshot is not None:
            evidence = event_diagnostic(case, snapshot, mpc._last_u.copy(), captures, cmd, point, budget, directory)
        states.append(point); commands.append(cmd); windows.append(window); sequences.append(mpc._last_u.copy()); state = next_state
    np.savez_compressed(directory / 'prefix.npz', states=np.asarray(states), commands=np.asarray(commands),
                        windows=np.asarray(windows), output_sequences=np.asarray(sequences), reference=trajectory.points)
    return dict(steps=STEPS, exact_archive_prefix=True, trace=rows, event=event, event_diagnostic=evidence)


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(90); torch.set_num_threads(1)
    report = dict(status='running', prefixes=[], training_updates=0, teacher_bank_created=False,
        learning_benefit_tested=False, allowed_to_prepare_training_plan=False, automatic_deployment=False,
        search_performed=False, full_candidate_episodes=0)
    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_write_json(output / 'report.json', report)
    with exploration_lock(output):
        save()
        try:
            config = prepare_config(budget); atomic_write_json(output / 'config.json', config)
            report['config_sha256'] = file_digest(output / 'config.json')
            _, sources = search.load_sources(); cases = {case['index']: case for _, case, *_ in sources}
            for index, strategy, seed, path, event in PREFIXES:
                directory = output / f'{index}-{strategy}-control{seed}'; directory.mkdir()
                row = dict(index=index, strategy=strategy, control_seed=seed, archive=path,
                    **replay_prefix(cases[index], FORMAL / path, seed, event, budget, directory))
                report['prefixes'].append(row); save()
                print('PREFIX', index, strategy, seed, 'EXACT', row['steps'], 'EVENT', event, flush=True)
            verify_config(config, budget); budget.check()
            report.update(status='completed', complete_prefixes=6, exact_control_steps=6*STEPS, complete_events=3,
                decision='fixed_prefix_mechanism_evidence_not_controller_or_teacher_qualification')
            report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                if p.is_file() and p != output / 'report.json' and p.name != 'run.lock'}
            save(); print('FINISHED', report['decision'], report['elapsed_seconds'], flush=True)
            return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else 'failed',
                          error=str(exc), decision='incomplete_do_not_infer'); save(); raise


def precheck():
    config = prepare_config(PreflightBudget(45))
    print('START TRACKING INPUTS VERIFIED; ZERO MPC/ROLLOUT/SEARCH/TRAINING', flush=True)
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
