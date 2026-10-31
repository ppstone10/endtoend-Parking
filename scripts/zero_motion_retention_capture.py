"""RESCUE-KEEP-002/003: isolated rolling pair capture; frozen solvers stay unchanged."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from controller.constrained_mpc import ControlSafetyError
from experiments.recovery_execution import MarginAuditEngine, execution_summary
from interfaces import ControlCmd
from scripts.emitted_execution_capture import CapturedNetworkSource, reference_checks
from scripts.feasible_cem_capture import latency_summary, audit_execution
from scripts.ungated_refinement_mpc import UngatedRefinedCEMController, PROFILE as PARENT_PROFILE
from scripts.zero_hold_motion_rescue import ZeroHoldMotionRescueController, PROFILE as CANDIDATE_PROFILE
from scripts.zero_hold_motion_capture import audit_rescue_execution
from scripts.ungated_refinement_capture import audit_ungated_execution
from scripts.feasible_cem_capture import audit_predictions
from scripts.diagnose_reference_execution import save_episode
from training.reporting import atomic_write_json

ARMS = ('parent', 'candidate')


def record_pair_arm(case, model, output, budget, *, arm, max_steps=600):
    """Only this private instance uses the feasible-CEM controller; no factory patch."""
    if arm not in ARMS:
        raise ValueError('unknown paired arm')
    branch = 'network_rolling'
    output = Path(output)
    frames, searches, failures, timings = [], [], [], []
    vehicle_model = case['engine_kwargs']['vehicle_model']
    controller = UngatedRefinedCEMController if arm == 'parent' else ZeroHoldMotionRescueController
    mpc = controller(goal=case['goal'],
        terminal=case['engine_kwargs']['terminal'], footprint_checker=case['planner']._collision_checker,
        vehicle_model=vehicle_model, dt=.1, horizon=10, seed=case['index'],
        **case['vehicle'].mpc_kwargs())
    source = CapturedNetworkSource(deepcopy(case['pipeline']), model, budget, fixed=False, step=lambda: len(frames))
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
    if source.raw:
        np.savez_compressed(output.parent / f'{output.name}-predictions.npz',
            **{key: np.stack([r[key] for r in source.raw])
               for key in ('points', 'logits', 'bev', 'goal', 'motion', 'pose')},
            lengths=np.array([r['length'] for r in source.raw]),
            issue_steps=np.array([p['step'] for p in source.plans]))
    else:
        np.savez_compressed(output.parent / f'{output.name}-predictions.npz',
            points=np.empty((0,320,3)), logits=np.empty((0,320)), lengths=np.array([],dtype=int),
            issue_steps=np.array([],dtype=int))
    summary = dict(dataset_index=case['index'], task_id=case['task_id'], scene_name=case['scene_name'],
        branch=branch, arm=arm, profile=mpc.profile,
        initial_contact_safe=bool(case['engine_kwargs']['collision_checker'].pose_free(*case['start'].to_array()[:3])),
        initial_margin_safe=bool(case['planner']._collision_checker.pose_free(*case['start'].to_array()[:3])), **execution_summary(result, engine.margin_violations, truncated=max_steps != 600),
        safety_stop=result.failure == 'safety_stop', zero_hold_steps=sum(s['is_zero_hold'] for s in searches),
        predictions=len(source.raw),
        first_prediction=checks[0] if checks else None, reference_checks=checks,
        planning_failure=result.meta.get('planning_failure'),
        control_timing=latency_summary(timings), controller_stats=mpc.safety_stats())
    detail = dict(**summary, frames=diagnostic['frames'], searches=searches,
        failed_calls=failures, first_events=diagnostic['first_events'],
        execution_npz_sha256=diagnostic['npz_sha256'], limits=diagnostic['limits'],
        geometry_source='simulator_ground_truth', required_margin_m=.2,
        candidate_qualified=False, training_performed=False)
    atomic_write_json(output.with_suffix('.json'), detail)
    audit_pair_arm(output, detail, case)
    return summary


def audit_pair_arm(output, detail, case):
    """Audit real controls and retain zero-step unsafe-start negative outcomes."""
    arm = detail['arm']
    if arm not in ARMS:
        raise ValueError('unknown arm')
    expected_profile = PARENT_PROFILE if arm == 'parent' else CANDIDATE_PROFILE
    if detail['profile'] != expected_profile or detail['branch'] != 'network_rolling':
        raise ValueError('arm/controller identity mismatch')
    contact = bool(case['engine_kwargs']['collision_checker'].pose_free(*case['start'].to_array()[:3]))
    margin = bool(case['planner']._collision_checker.pose_free(*case['start'].to_array()[:3]))
    if detail['initial_contact_safe'] != contact or detail['initial_margin_safe'] != margin:
        raise ValueError('initial risk report mismatch')
    checked = audit_execution(output, detail, case)
    if detail['steps']:
        audit = audit_ungated_execution if arm == 'parent' else audit_rescue_execution
        audit(output, detail, case)
    elif detail['controller_stats']:
        raise ValueError('zero-step controller counters unexpected')
    for failure in detail['failed_calls']:
        with np.load(Path(output).with_suffix('.npz'), allow_pickle=False) as saved:
            state = saved['states_after'][-1] if detail['steps'] else saved['initial_state']
            np.testing.assert_array_equal(failure['before'], state)
        if (case['planner']._collision_checker.pose_free(*state[:3])
                or failure['reason'] != 'current footprint violates required clearance'):
            raise ValueError('unverified safe-state safety failure; retain incomplete evidence')
    with np.load(Path(output).parent / f'{Path(output).name}-predictions.npz', allow_pickle=False) as raw:
        if len(raw['lengths']):
            audit_predictions(output, detail)
        elif (detail['predictions'] or detail['steps'] or not detail['planning_failure']
              or detail['first_prediction'] is not None):
            raise ValueError('unexplained empty predictions')
    return checked


def audit_paired_prefix(parent, candidate):
    """Prove only the fixed-bank intervention can cause the first divergence."""
    parent, candidate = Path(parent), Path(candidate)
    import json
    original = json.loads(parent.with_suffix('.json').read_text(encoding='utf-8'))
    changed = json.loads(candidate.with_suffix('.json').read_text(encoding='utf-8'))
    interventions = [n for n,s in enumerate(changed['searches']) if s['rescue']['intervened']]
    shared = interventions[0] if interventions else changed['steps']
    if (not interventions and original['steps'] != changed['steps']
            or interventions and shared >= original['steps']):
        raise ValueError('pair terminated differently before intervention')
    with np.load(parent.with_suffix('.npz'), allow_pickle=False) as old:
        with np.load(candidate.with_suffix('.npz'), allow_pickle=False) as new:
            for key in ('initial_state','goal'):
                np.testing.assert_array_equal(old[key], new[key])
            for key in ('states_before','states_after','control_windows','selected_control_sequences'):
                np.testing.assert_array_equal(old[key][:shared], new[key][:shared])
            if interventions:
                for key in ('states_before','control_windows'):
                    np.testing.assert_array_equal(old[key][shared], new[key][shared])
    with np.load(parent.parent / f'{parent.name}-predictions.npz', allow_pickle=False) as old:
        with np.load(candidate.parent / f'{candidate.name}-predictions.npz', allow_pickle=False) as new:
            upto = shared if interventions else float('inf')
            old_n = int(np.count_nonzero(old['issue_steps'] <= upto))
            new_n = int(np.count_nonzero(new['issue_steps'] <= upto))
            if old_n != new_n:
                raise ValueError('prediction schedule changed before intervention')
            for key in old.files:
                np.testing.assert_array_equal(old[key][:old_n], new[key][:new_n])
    if not interventions:
        for key in ('success','failure','collision','margin_violation_steps','planning_failure'):
            if original[key] != changed[key]:
                raise ValueError('unmodified execution outcome differs')
    return dict(shared_control_steps=shared, rescue_interventions=len(interventions))


def audit_forward(output, model, budget):
    """Recompute all H320 points/logits for the exact stored native inputs."""
    import torch
    count = 0
    with np.load(Path(output).parent / f'{Path(output).name}-predictions.npz', allow_pickle=False) as raw:
        with torch.no_grad():
            for n in range(len(raw['lengths'])):
                budget.check()
                predicted = model.forward_with_stop(torch.from_numpy(raw['bev'][n:n+1]),
                    torch.from_numpy(raw['goal'][n:n+1]), torch.from_numpy(raw['motion'][n:n+1]))
                np.testing.assert_array_equal(predicted.points[0].cpu().numpy(), raw['points'][n])
                np.testing.assert_array_equal(predicted.stop_logits[0].cpu().numpy(), raw['logits'][n])
                count += 1
    return count

