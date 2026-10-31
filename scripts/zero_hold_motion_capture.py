"""MPC-RESCUE-003: actual execution, frozen parent and fixed-bank audit."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from controller.constrained_mpc import ControlSafetyError
from controller.mpc import MPCController
from experiments.recovery_execution import MarginAuditEngine, execution_summary
from interfaces import ControlCmd, VehicleState, Trajectory
from scripts.emitted_execution_capture import CapturedNetworkSource, reference_checks
from scripts.feasible_cem_capture import RecordedTeacherSource, latency_summary, audit_execution
from scripts.ungated_refinement_mpc import UngatedRefinedCEMController, OPTIONS
from scripts.zero_hold_motion_rescue import ZeroHoldMotionRescueController, PROFILE, SPEEDS, motion_bank
from scripts.diagnose_reference_execution import save_episode
from training.reporting import atomic_write_json

def record_rescue_branch(case, model, output, budget, *, branch, teacher_archive=None, max_steps=600):
    """Only this private instance uses the feasible-CEM controller; no factory patch."""
    if branch not in {'network_fixed', 'network_rolling', 'teacher_fixed'}:
        raise ValueError('unknown experimental branch')
    output = Path(output)
    frames, searches, failures, timings = [], [], [], []
    vehicle_model = case['engine_kwargs']['vehicle_model']
    mpc = ZeroHoldMotionRescueController(goal=case['goal'],
        terminal=case['engine_kwargs']['terminal'], footprint_checker=case['planner']._collision_checker,
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
        branch=branch, profile=PROFILE, **execution_summary(result, engine.margin_violations, truncated=max_steps != 600),
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
    audit_rescue_execution(output, detail, case)
    return summary


def audit_rescue_execution(output, detail, case):
    """Rebuild progress, replay the frozen CEM and independently check the bank."""
    if detail['profile'] != PROFILE:
        raise ValueError('rescue profile identity mismatch')
    reference = MPCController(dt=.1, horizon=10)
    parent = UngatedRefinedCEMController(goal=case['goal'], terminal=case['engine_kwargs']['terminal'],
        footprint_checker=case['planner']._collision_checker,
        vehicle_model=case['engine_kwargs']['vehicle_model'], dt=.1, horizon=10,
        seed=case['index'], **case['vehicle'].mpc_kwargs())
    warm, previous_plan, aligned = np.zeros(20), None, None
    bank = motion_bank(10)
    deltas = {}
    totals = dict(rescue_attempts=0, rescue_candidates_checked=0,
                  rescue_safe_candidates=0, rescue_interventions=0)
    with np.load(Path(output).with_suffix('.npz'), allow_pickle=False) as saved:
        for n, search in enumerate(detail['searches']):
            if search['profile'] != PROFILE or search['options'] != OPTIONS or search['progress_gate']:
                raise ValueError('parent progress/options changed')
            frame = detail['frames'][n]
            plan = frame['plan_ordinal']
            if plan != previous_plan:
                trajectory = Trajectory(saved[f'global_plan_{plan:04d}'], float(saved[f'plan_dt_{plan:04d}']))
                aligned = reference._align_dt(trajectory)
                reference._progress_idx = None
                previous_plan = plan
            state = VehicleState.from_array(saved['states_before'][n])
            progress = reference._advance_progress(aligned, state)
            if frame['progress_index'] != progress:
                raise ValueError('original clock progress changed')
            window = reference._reference_points(aligned, progress)
            np.testing.assert_array_equal(window, saved['control_windows'][n])
            base = parent._solve(warm, window, state)
            base_audit = parent.safety_audit()
            for key in ('generations', 'final_mean_safe', 'candidate_costs', 'terminal_rotation'):
                if search[key] != base_audit[key]:
                    raise ValueError('frozen parent CEM/terminal changed')
            attempted = bool(np.all(base == 0.))
            expected = dict(attempted=attempted, intervened=False, candidate_count=0,
                base_selected=base_audit['selected'], base_cost=base_audit['selected_cost'],
                base_sequence=base.reshape(-1, 2).tolist(), speeds=[], safe_mask=[], costs=[], selected_index=None)
            selected, label, cost = base, base_audit['selected'], base_audit['selected_cost']
            if attempted:
                safe = np.array([parent.sequence_safe(sequence, state) for sequence in bank], dtype=bool)
                costs = MPCController._batch_cost(parent, bank, window, state)
                expected.update(candidate_count=8, speeds=list(SPEEDS), safe_mask=safe.tolist(), costs=costs.tolist())
                eligible = np.flatnonzero(safe)
                if len(eligible):
                    winner = int(eligible[np.argmin(costs[eligible])])
                    if float(costs[winner]) < cost:
                        selected, label, cost = bank[winner], 'primitive_rescue', float(costs[winner])
                        expected.update(intervened=True, selected_index=winner)
                        for key, delta in [(base_audit['selected']+'_selected', -1),
                                           ('primitive_rescue_selected', 1), ('zero_hold', -1)]:
                            deltas[key] = deltas.get(key, 0)+delta
            if search['rescue'] != expected:
                raise ValueError('fixed-bank safety/cost/selection mismatch')
            np.testing.assert_array_equal(selected, saved['selected_control_sequences'][n].reshape(-1))
            if search['selected'] != label or search['selected_cost'] != cost or search['is_zero_hold'] != bool(np.all(selected == 0.)):
                raise ValueError('actual rescue output/report mismatch')
            totals['rescue_attempts'] += int(attempted)
            totals['rescue_candidates_checked'] += expected['candidate_count']
            totals['rescue_safe_candidates'] += sum(expected['safe_mask'])
            totals['rescue_interventions'] += int(expected['intervened'])
            warm = parent._shift(selected)
    expected_stats = parent.safety_stats()
    expected_stats.update(totals)
    for key, delta in deltas.items():
        expected_stats[key] = expected_stats.get(key, 0)+delta
    if detail['controller_stats'] != expected_stats:
        raise ValueError('rescue/frozen parent counters mismatch')
    return dict(controls=len(detail['searches']), frozen_parent_replays=len(detail['searches']), **totals)

