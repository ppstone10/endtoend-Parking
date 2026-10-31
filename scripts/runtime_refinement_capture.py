"""RUN-REFINE-006: private instance capture and independent refinement audit."""
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
from scripts.runtime_refinement_mpc import RefinedCEMController, PROFILES
from scripts.diagnose_reference_execution import save_episode
from training.reporting import atomic_write_json

def record_refined_branch(case, model, output, budget, *, profile, branch, teacher_archive=None, max_steps=600):
    """Only this private instance uses the feasible-CEM controller; no factory patch."""
    if branch not in {'network_fixed', 'network_rolling', 'teacher_fixed'}:
        raise ValueError('unknown experimental branch')
    output = Path(output)
    frames, searches, failures, timings = [], [], [], []
    vehicle_model = case['engine_kwargs']['vehicle_model']
    mpc = RefinedCEMController(profile=profile, goal=case['goal'],
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
        branch=branch, profile=profile, **execution_summary(result, engine.margin_violations, truncated=max_steps != 600),
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
    audit_refinement(output, detail, case)
    return summary


def audit_refinement(output, detail, case):
    """Rebuild RNG/cost/fits, progress gates and terminal certificates from arrays.

    Candidate feasibility masks are logged, not independently resampled in full;
    the legacy scalar audit separately checks every actual selected H10.
    """
    options = PROFILES[detail['profile']]
    rng = np.random.default_rng(case['index'])
    with np.load(Path(output).with_suffix('.npz'), allow_pickle=False) as saved:
        before = saved['states_before']
        windows = saved['control_windows']
        controls = saved['selected_control_sequences']
        initial_std = np.tile([1., .8], 10)
        warm = np.zeros(20)
        cost_model = MPCController()
        plan_ordinal, previous_progress, aligned = None, None, None
        for n, search in enumerate(detail['searches']):
            if search['profile'] != detail['profile'] or search['options'] != options:
                raise ValueError('refinement switch identity drift')
            state, ref = VehicleState.from_array(before[n]), windows[n]
            mean, std = warm.copy(), initial_std.copy()
            best, best_cost = None, np.inf
            for g in search['generations']:
                np.testing.assert_array_equal(g['mean_before'], mean)
                np.testing.assert_array_equal(g['std_before'], std)
                samples = rng.normal(size=(80, 20))*std+mean
                samples[:, 0::2] = np.clip(samples[:, 0::2], -2., 2.)
                samples[:, 1::2] = np.clip(samples[:, 1::2], -1., 1.)
                costs = cost_model._batch_cost(samples, ref, state)
                eligible = np.flatnonzero(g['feasible_mask'])
                ranked = eligible[np.argsort(costs[eligible])]
                elites = ranked[:20]
                np.testing.assert_array_equal(g['elite_indices'], elites)
                guarded = bool(options['variance'] and len(elites) < 2)
                if g['sparse_elite_guard'] != guarded:
                    raise ValueError('sparse elite protection report mismatch')
                if len(elites):
                    if costs[ranked[0]] < best_cost:
                        best, best_cost = samples[ranked[0]].copy(), float(costs[ranked[0]])
                    if not guarded:
                        fit_mean, fit_std = samples[elites].mean(axis=0), samples[elites].std(axis=0)+1e-3
                        if options['variance']:
                            mean = .1*mean+.9*fit_mean
                            std = np.maximum(.1*std+.9*fit_std, .1*initial_std)
                        else:
                            mean, std = fit_mean, fit_std
                np.testing.assert_array_equal(g['mean_after'], mean)
                np.testing.assert_array_equal(g['std_after'], std)
            # Validate all recorded competing costs, including an unexecuted mean.
            candidates = {'feasible_mean': mean, 'shifted_warm_start': warm, 'zero_hold': np.zeros(20)}
            if best is not None:
                candidates['safe_sample'] = best
            for candidate in search['candidate_costs']:
                cost = float(cost_model._batch_cost(candidates[candidate['label']][None], ref, state)[0])
                if cost != candidate['cost']:
                    raise ValueError('recorded candidate cost mismatch')
            audit = search['terminal_rotation']
            terminal = case['engine_kwargs']['terminal']
            pos, yaw = terminal.pos_err(state, case['goal']), terminal.yaw_err(state, case['goal'])
            attempted = bool(options['terminal_rotation'] and pos < terminal.tol_pos and yaw >= terminal.tol_yaw)
            if audit['enabled'] != options['terminal_rotation'] or audit['attempted'] != attempted:
                raise ValueError('terminal gate mismatch')
            if attempted:
                angle = float(np.arctan2(np.sin(case['goal'].yaw-state.yaw), np.cos(case['goal'].yaw-state.yaw)))
                end = np.array([state.x, state.y, state.yaw+angle])
                certified = case['planner']._collision_checker.swept_segment_free(before[n, :3], end)
                if bool(certified) != audit['full_rotation_margin_safe'] or angle != audit['signed_rotation']:
                    raise ValueError('full rotation certificate mismatch')
            if audit['applied']:
                if not attempted or not audit['full_rotation_margin_safe'] or search['selected'] != 'terminal_rotation':
                    raise ValueError('uncertified terminal intervention')
                np.testing.assert_array_equal(controls[n, :, 0], np.zeros(10))
            elif search['selected'] == 'terminal_rotation':
                raise ValueError('unreported terminal intervention')
            else:
                winner = min(search['candidate_costs'], key=lambda c: c['cost'])
                if winner['label'] != search['selected']:
                    raise ValueError('verified best candidate not selected')
                np.testing.assert_array_equal(controls[n].reshape(-1), candidates[winner['label']])
            selected_cost = float(cost_model._batch_cost(controls[n].reshape(1, -1), ref, state)[0])
            if selected_cost != search['selected_cost']:
                raise ValueError('selected cost mismatch')
            frame = detail['frames'][n]
            progress = frame['progress_index']
            if plan_ordinal != frame['plan_ordinal']:
                plan_ordinal = frame['plan_ordinal']
                trajectory = Trajectory(saved[f'global_plan_{plan_ordinal:04d}'], float(saved[f'plan_dt_{plan_ordinal:04d}']))
                aligned = cost_model._align_dt(trajectory)
                expected = int(np.argmin((aligned[:, 0]-state.x)**2+(aligned[:, 1]-state.y)**2))
                if progress != expected:
                    raise ValueError('initial reference alignment changed')
            elif options['progress']:
                if not previous_progress <= progress <= previous_progress+4:
                    raise ValueError('nonlocal or regressing reference progress')
                for skipped in aligned[previous_progress:progress]:
                    heading = abs(np.arctan2(np.sin(skipped[2]-state.yaw), np.cos(skipped[2]-state.yaw)))
                    if np.hypot(skipped[0]-state.x, skipped[1]-state.y) > .15 or heading > .1:
                        raise ValueError('reference advanced through unreached pose')
            previous_progress = progress
            np.testing.assert_array_equal(cost_model._reference_points(aligned, progress), ref)
            previous = controls[n].reshape(-1)
            warm[:-2], warm[-2:] = previous[2:], previous[-2:]
    return dict(controls=len(detail['searches']), candidate_masks='logged_not_exhaustively_resampled',
                all_actual_selected_h10='independently_scalar_checked')

