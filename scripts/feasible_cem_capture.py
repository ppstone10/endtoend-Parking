"""FEAS-CEM-003: capture new controller execution and independently audit it."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from controller.constrained_mpc import ControlSafetyError
from experiments.recovery_execution import MarginAuditEngine, execution_summary
from interfaces import ControlCmd, Trajectory, VehicleState
from scripts.emitted_execution_capture import CapturedNetworkSource, reference_checks
from scripts.feasible_cem_mpc import FeasibleCEMController
from scripts.diagnose_reference_execution import save_episode
from training.reporting import atomic_write_json


class RecordedTeacherSource:
    """Reuse LT34's original world reference, without invoking a planner/network."""

    def __init__(self, archive):
        with np.load(archive, allow_pickle=False) as saved:
            self.trajectory = Trajectory(saved['reference'].copy(), float(saved['dt']))
            self.local = saved['frame_points'][0, :int(saved['frame_lengths'][0])].copy()
            self.bev = saved['frame_bevs'][0].copy()
        self.plans = []

    def begin(self, start, goal):
        self.plans = [dict(step=0, state=start.to_array(), global_=self.trajectory.points,
                           local=self.local, bev=self.bev, dt=self.trajectory.dt)]
        self.plans[0]['global'] = self.plans[0].pop('global_')

    def next_trajectory(self, state):
        return self.trajectory, 0.


def latency_summary(milliseconds):
    values = np.asarray(milliseconds, dtype=float)
    if not len(values):
        return dict(calls=0, p50_ms=None, p95_ms=None, max_ms=None, over_100ms=0,
                    over_100ms_fraction=None)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('invalid controller timing')
    return dict(calls=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(values.max()),
                over_100ms=int((values > 100.).sum()),
                over_100ms_fraction=float((values > 100.).mean()))


def record_feasible_branch(case, model, output, budget, *, branch, teacher_archive=None, max_steps=600):
    """Only this private instance uses the feasible-CEM controller; no factory patch."""
    if branch not in {'network_fixed', 'network_rolling', 'teacher_fixed'}:
        raise ValueError('unknown experimental branch')
    output = Path(output)
    frames, searches, failures, timings = [], [], [], []
    vehicle_model = case['engine_kwargs']['vehicle_model']
    mpc = FeasibleCEMController(footprint_checker=case['planner']._collision_checker,
        vehicle_model=vehicle_model, dt=.1, horizon=10, seed=case['index'],
        **case['vehicle'].mpc_kwargs())
    if branch == 'teacher_fixed':
        source = RecordedTeacherSource(teacher_archive)
    else:
        source = CapturedNetworkSource(deepcopy(case['pipeline']), model, budget,
            fixed=branch == 'network_fixed', step=lambda: len(frames))
    engine = MarginAuditEngine(mpc=mpc, source=source, max_steps=max_steps, replan_every=10,
        planning_checker=case['planner']._collision_checker,
        meta=dict(dataset_index=case['index'], scene_name=case['scene_name']), **case['engine_kwargs'])
    original_compute = mpc.compute

    def capture(trajectory, state):
        budget.check()
        started = time.perf_counter()
        try:
            command = original_compute(trajectory, state)
        except ControlSafetyError as error:
            elapsed = 1000. * (time.perf_counter() - started)
            timings.append(elapsed)
            failures.append(dict(step=len(frames), before=state.to_array().tolist(),
                                 reason=str(error), compute_ms=elapsed))
            raise
        elapsed = 1000. * (time.perf_counter() - started)
        timings.append(elapsed)
        budget.check()
        controls = mpc._last_u.copy().reshape(-1, 2)
        predicted, rollout = state, []
        for v, omega in controls:
            predicted = vehicle_model.step(predicted, ControlCmd(float(v), float(omega)), mpc.dt)
            rollout.append(predicted.to_array())
        frames.append(dict(before=state.to_array(),
            window=mpc._reference_points(mpc._aligned_cache, mpc._progress_idx).copy(),
            u=controls, rollout=np.asarray(rollout), progress=mpc._progress_idx,
            aligned_length=len(mpc._aligned_cache), plan_ordinal=len(source.plans) - 1))
        searches.append(dict(step=len(frames) - 1, compute_ms=elapsed, **mpc.safety_audit()))
        return command

    mpc.compute = capture
    try:
        result = engine.run(deepcopy(case['start']), case['goal'])
    finally:
        mpc.compute = original_compute
    budget.check()
    diagnostic = save_episode(output, engine, case['start'], case['goal'], result, frames, source.plans)
    checks = [reference_checks(plan, case) for plan in source.plans]
    if branch != 'teacher_fixed':
        np.savez_compressed(output.parent / f'{output.name}-predictions.npz',
            **{key: np.stack([r[key] for r in source.raw])
               for key in ('points', 'logits', 'bev', 'goal', 'motion', 'pose')},
            lengths=np.array([r['length'] for r in source.raw]),
            issue_steps=np.array([p['step'] for p in source.plans]))
    summary = dict(dataset_index=case['index'], task_id=case['task_id'], scene_name=case['scene_name'],
        branch=branch, **execution_summary(result, engine.margin_violations, truncated=max_steps != 600),
        safety_stop=result.failure == 'safety_stop', zero_hold_steps=sum(s['is_zero_hold'] for s in searches),
        predictions=0 if branch == 'teacher_fixed' else len(source.raw),
        first_prediction=checks[0], reference_checks=checks,
        control_timing=latency_summary(timings), controller_stats=mpc.safety_stats())
    detail = dict(**summary, frames=diagnostic['frames'], searches=searches,
        failed_calls=failures, first_events=diagnostic['first_events'],
        execution_npz_sha256=diagnostic['npz_sha256'], limits=diagnostic['limits'],
        geometry_source='simulator_ground_truth', required_margin_m=.2,
        candidate_qualified=False, training_performed=False)
    atomic_write_json(output.with_suffix('.json'), detail)
    audit_execution(output, detail, case)
    return summary


def audit_execution(output, detail, case):
    """Reconstruct every selected H10 and actual transition, without trusting flags."""
    output = Path(output)
    with np.load(output.with_suffix('.npz'), allow_pickle=False) as saved:
        before = saved['states_before']
        after = saved['states_after']
        controls = saved['selected_control_sequences']
        rollouts = saved['selected_control_rollouts']
        goal = saved['goal']
        initial = saved['initial_state']
    count = len(after)
    if not (len(before) == len(controls) == len(rollouts) == len(detail['searches'])
            == len(detail['frames']) == count == detail['steps']):
        raise ValueError('control evidence denominator mismatch')
    model = case['engine_kwargs']['vehicle_model']
    checkers = (case['engine_kwargs']['collision_checker'], case['planner']._collision_checker)
    terminal = case['engine_kwargs']['terminal']
    reached = []
    if count and (controls.shape != (count, 10, 2) or rollouts.shape != (count, 10, 5)):
        raise ValueError('frozen H10 output shape mismatch')
    for n in range(count):
        np.testing.assert_allclose(before[n], initial if n == 0 else after[n-1], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(controls[n], detail['searches'][n]['selected_sequence'])
        np.testing.assert_allclose(detail['frames'][n]['command'], controls[n, 0], atol=0, rtol=0)
        if not np.isfinite(controls[n]).all():
            raise ValueError('nonfinite output controls')
        if np.any(np.abs(controls[n, :, 0]) > 2.) or np.any(np.abs(controls[n, :, 1]) > 1.):
            raise ValueError('output exceeds frozen execution limits')
        predicted = VehicleState.from_array(before[n])
        for k, (v, omega) in enumerate(controls[n]):
            following = model.step(predicted, ControlCmd(float(v), float(omega)), .1)
            np.testing.assert_allclose(following.to_array(), rollouts[n, k], atol=1e-12, rtol=0)
            for checker in checkers:
                if not checker.pose_free(*predicted.to_array()[:3]) or not checker.swept_segment_free(
                        predicted.to_array()[:3], following.to_array()[:3]):
                    raise ValueError('selected H10 violates contact or required margin')
            predicted = following
        np.testing.assert_allclose(rollouts[n, 0], after[n], atol=1e-12, rtol=0)
        reached.append(terminal.reached(VehicleState.from_array(after[n]), case['goal']))
        generations = detail['searches'][n]['generations']
        if len(generations) != 5:
            raise ValueError('frozen CEM iteration count mismatch')
        for generation in generations:
            mask = np.asarray(generation['feasible_mask'], dtype=bool)
            elites = generation['elite_indices']
            if (len(mask) != 80 or int(mask.sum()) != generation['feasible_samples']
                    or int((~mask).sum()) != generation['rejected_samples']
                    or len(elites) != min(20, int(mask.sum())) or len(set(elites)) != len(elites)
                    or any(i < 0 or i >= 80 for i in elites) or not all(mask[i] for i in elites)):
                raise ValueError('unsafe sample recorded as eligible elite')
    expected_success = bool(count and reached[-1] and detail['failure'] is None)
    if (any(reached[:-1]) or bool(detail['success']) != expected_success
            or detail['collision'] or detail['margin_violation_steps'] != 0):
        raise ValueError('arrival/risk report does not match actual execution')
    np.testing.assert_array_equal(goal, [case['goal'].x, case['goal'].y, case['goal'].yaw])
    failures = detail['failed_calls']
    if bool(failures) != bool(detail['safety_stop']) or len(failures) > 1:
        raise ValueError('safety-stop call not separated from executed steps')
    timing = [s['compute_ms'] for s in detail['searches']] + [s['compute_ms'] for s in failures]
    if detail['control_timing'] != latency_summary(timing):
        raise ValueError('timing summary mismatch')
    if detail['zero_hold_steps'] != sum(bool(np.all(sequence == 0.)) for sequence in controls):
        raise ValueError('zero hold counted as movement')
    final = VehicleState.from_array(after[-1] if count else initial)
    if (abs(terminal.pos_err(final, case['goal']) - detail['final_pos_err']) > 1e-12
            or abs(terminal.yaw_err(final, case['goal']) - detail['final_yaw_err']) > 1e-12):
        raise ValueError('final pose errors do not match actual state')
    return dict(steps=count, selected_h10_transitions=count * 10,
                contact_violations=0, margin_violations=0)


def audit_predictions(output, detail, teacher_archive=None):
    """Validate every emitted prefix and unchanged native stop, not just frame zero."""
    from runtime.sources import NetworkSource
    from experiments.emitted_output_joint import native_lengths
    import torch

    output = Path(output)
    with np.load(output.with_suffix('.npz'), allow_pickle=False) as executed:
        if detail['branch'] == 'teacher_fixed':
            with np.load(teacher_archive, allow_pickle=False) as teacher:
                np.testing.assert_array_equal(executed['global_plan_0000'], teacher['reference'])
            if executed['plan_issue_steps'].tolist() != [0] or detail['predictions'] != 0:
                raise ValueError('teacher reference was replaced')
            return
        with np.load(output.parent / f'{output.name}-predictions.npz', allow_pickle=False) as raw:
            length = len(raw['lengths'])
            if (length != detail['predictions'] or raw['points'].shape != (length, 320, 3)
                    or raw['logits'].shape != (length, 320)):
                raise ValueError('full native prediction denominator/shape mismatch')
            lengths = native_lengths(torch.from_numpy(raw['logits'])).numpy()
            np.testing.assert_array_equal(lengths, raw['lengths'])
            np.testing.assert_array_equal(raw['issue_steps'], executed['plan_issue_steps'])
            expected_steps = [0] if detail['branch'] == 'network_fixed' else list(range(0, int(raw['issue_steps'][-1]) + 1, 10))
            if raw['issue_steps'].tolist() != expected_steps:
                raise ValueError('native prediction schedule changed')
            for ordinal, emitted in enumerate(lengths):
                np.testing.assert_array_equal(executed[f'local_prediction_{ordinal:04d}'], raw['points'][ordinal, :emitted])
                pose = executed['plan_issue_states'][ordinal]
                np.testing.assert_array_equal(pose, raw['pose'][ordinal])
                # Native differential-drive evolution uses Python scalars at start,
                # then numpy.float64 x/y and Python yaw. Preserve float32 export arithmetic.
                initial = int(raw['issue_steps'][ordinal]) == 0
                state = VehicleState(float(pose[0]) if initial else np.float64(pose[0]),
                    float(pose[1]) if initial else np.float64(pose[1]),
                    float(pose[2]), float(pose[3]), float(pose[4]))
                converted = NetworkSource._to_global(raw['points'][ordinal, :emitted], state)
                np.testing.assert_array_equal(converted, executed[f'global_plan_{ordinal:04d}'])
                if float(executed[f'plan_dt_{ordinal:04d}']) != .2:
                    raise ValueError('native trajectory timestep changed')
