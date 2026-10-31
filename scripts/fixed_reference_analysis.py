"""REF-DIAG-002..004: saved-output pairing and existing window continuity metrics."""
from pathlib import Path
import numpy as np
from controller.mpc import MPCController
from interfaces import Trajectory, VehicleState
from experiments.emitted_output_data import read_json
from experiments.reference_update_stability import compare_windows, window_geometry


def verify_fixed_prefix(fixed, rolling):
    """Require identical native first prediction and ten actually executed steps."""
    fixed,rolling=Path(fixed),Path(rolling)
    f,r=read_json(fixed.with_suffix('.json')),read_json(rolling.with_suffix('.json'))
    if (f['branch']!='network_fixed' or r['branch']!='network_rolling'
            or f['profile']!='zero_rescue' or r['profile']!='zero_rescue'
            or f['task_id']!=r['task_id'] or f['dataset_index']!=r['dataset_index']
            or f['predictions']!=1 or f['steps']<10 or r['steps']<10):
        raise ValueError('fixed/rolling identity, one native prediction and 10 actual steps required')
    with np.load(fixed.with_suffix('.npz'),allow_pickle=False) as a:
        with np.load(rolling.with_suffix('.npz'),allow_pickle=False) as b:
            if a['plan_issue_steps'].tolist()!=[0]:raise ValueError('fixed reference was replaced')
            for key in ('initial_state','goal','global_plan_0000','local_prediction_0000'):
                np.testing.assert_array_equal(a[key],b[key])
            for key in ('states_before','states_after','control_windows','selected_control_sequences'):
                np.testing.assert_array_equal(a[key][:10],b[key][:10])
    with np.load(fixed.parent/f'{fixed.name}-predictions.npz',allow_pickle=False) as a:
        with np.load(rolling.parent/f'{rolling.name}-predictions.npz',allow_pickle=False) as b:
            for key in ('points','logits','bev','goal','motion','pose','lengths','issue_steps'):
                if len(a[key])!=1:raise ValueError('fixed branch decoded more than once')
                np.testing.assert_array_equal(a[key][0],b[key][0])
    return dict(initial_native_prediction_exact=True,shared_actual_steps=10)


def execution_observation(output, detail, terminal):
    """Keep actual task convergence separate from raw reference endpoint quality."""
    with np.load(Path(output).with_suffix('.npz'),allow_pickle=False) as saved:
        states=saved['states_after'];controls=saved['selected_control_sequences'];goal=saved['goal']
    if not len(states):raise ValueError('frozen LT40 failure cohort requires executed evidence')
    position=np.linalg.norm(states[:,:2]-goal[:2],axis=1)
    yaw=np.abs(np.arctan2(np.sin(states[:,2]-goal[2]),np.cos(states[:,2]-goal[2])))
    joint=(position<terminal.tol_pos)&(yaw<terminal.tol_yaw)
    hits=np.flatnonzero(joint)
    if bool(len(hits))!=bool(detail['success']):raise ValueError('actual arrival mismatch')
    active=np.sign(controls[:,0,0][np.abs(controls[:,0,0])>.1])
    nearest=int(position.argmin())
    return dict(steps=len(states),success=detail['success'],failure=detail['failure'],
        final_position_m=float(position[-1]),final_yaw_rad=float(yaw[-1]),
        closest_position_m=float(position[nearest]),yaw_at_closest_position_rad=float(yaw[nearest]),
        position_threshold_visited=bool((position<terminal.tol_pos).any()),
        first_joint_arrival_step=None if not len(hits) else int(hits[0]),
        active_velocity_sign_flips=int(np.count_nonzero(active[1:]!=active[:-1])),
        full_zero_h10_steps=int(np.count_nonzero(np.all(controls==0,axis=(1,2)))),
        terminal_attempts=detail['controller_stats'].get('terminal_attempts',0),
        terminal_rotation_selected=detail['controller_stats'].get('terminal_rotation_selected',0),
        final_speed_mps=float(states[-1,3]),final_omega_radps=float(states[-1,4]),
        first_prediction=detail['first_prediction'],control_timing=detail['control_timing'])


def classify_fixed_result(fixed, rolling, *, formal):
    if rolling['success']:raise ValueError('frozen rolling branch must be a prior failure')
    if not formal:return 'truncated_unclassified'
    risk=bool(fixed['collision'] or fixed['margin_violation_steps'])
    if not fixed['success']:return 'fixed_reference_execution_also_failed'
    if risk or fixed['control_timing']['over_100ms']:
        return 'fixed_arrival_unqualified_risk_or_latency'
    return 'fixed_reference_recovers_rolling_failure'


def rolling_window_updates(output, detail, case):
    """Observe old continuation vs new initialization at the SAME recorded state.

    This constructs reference windows only; no control is optimized or executed.
    """
    output=Path(output)
    updates=[]
    reference=MPCController(dt=.1,horizon=10)
    with np.load(output.with_suffix('.npz'),allow_pickle=False) as saved:
        issue_steps=saved['plan_issue_steps'];states=saved['states_before'];actual=saved['control_windows']
        for ordinal,step_value in enumerate(issue_steps[1:],start=1):
            step=int(step_value)
            if step<=0 or step%10 or step>=len(states):raise ValueError('rolling replan schedule mismatch')
            if detail['frames'][step-1]['plan_ordinal']!=ordinal-1 or detail['frames'][step]['plan_ordinal']!=ordinal:
                raise ValueError('rolling reference issue mapping mismatch')
            state=VehicleState.from_array(states[step])
            old=Trajectory(saved[f'global_plan_{ordinal-1:04d}'],float(saved[f'plan_dt_{ordinal-1:04d}']))
            fresh=Trajectory(saved[f'global_plan_{ordinal:04d}'],float(saved[f'plan_dt_{ordinal:04d}']))
            old_aligned=reference._align_dt(old)
            reference._progress_idx=detail['frames'][step-1]['progress_index']
            old_index=reference._advance_progress(old_aligned,state)
            old_window=reference._reference_points(old_aligned,old_index)
            fresh_aligned=reference._align_dt(fresh)
            reference._progress_idx=None
            fresh_index=reference._advance_progress(fresh_aligned,state)
            fresh_window=reference._reference_points(fresh_aligned,fresh_index)
            np.testing.assert_array_equal(fresh_window,actual[step])
            if detail['frames'][step]['progress_index']!=fresh_index:
                raise ValueError('new reference initialization mismatch')
            previous_geometry={name:window_geometry(state,old_window,checker) for name,checker in
                (('contact',case['engine_kwargs']['collision_checker']),('planning',case['planner']._collision_checker))}
            fresh_geometry={name:window_geometry(state,fresh_window,checker) for name,checker in
                (('contact',case['engine_kwargs']['collision_checker']),('planning',case['planner']._collision_checker))}
            updates.append(dict(step=step,previous_progress_index=old_index,fresh_progress_index=fresh_index,
                **compare_windows(old_window,fresh_window),previous_geometry=previous_geometry,
                fresh_geometry=fresh_geometry,safety_regression={name:
                    all(previous_geometry[name].values()) and not all(fresh_geometry[name].values())
                    for name in ('contact','planning')}))
    return updates


def diagnose_pair(fixed, rolling, case, *, formal):
    fixed,rolling=Path(fixed),Path(rolling)
    f,r=read_json(fixed.with_suffix('.json')),read_json(rolling.with_suffix('.json'))
    prefix=verify_fixed_prefix(fixed,rolling)
    updates=rolling_window_updates(rolling,r,case)
    with np.load(fixed.with_suffix('.npz'),allow_pickle=False) as a:
        with np.load(rolling.with_suffix('.npz'),allow_pickle=False) as b:
            n=min(len(a['states_after']),len(b['states_after']))
            fa,ra=a['selected_control_sequences'][:n,0],b['selected_control_sequences'][:n,0]
            changed=np.flatnonzero(np.any(fa!=ra,axis=1))
    return dict(dataset_index=f['dataset_index'],task_id=f['task_id'],scene_name=f['scene_name'],
        classification=classify_fixed_result(f,r,formal=formal),prefix=prefix,
        first_executed_command_difference_step=None if not len(changed) else int(changed[0]),
        fixed=execution_observation(fixed,f,case['engine_kwargs']['terminal']),
        rolling=execution_observation(rolling,r,case['engine_kwargs']['terminal']),
        rolling_updates=updates,
        window_change_summary=dict(updates=len(updates),
            xy_rms_p50_m=None if not updates else float(np.median([u['xy_rms_m'] for u in updates])),
            xy_rms_max_m=None if not updates else max(u['xy_rms_m'] for u in updates),
            yaw_rms_max_rad=None if not updates else max(u['yaw_rms_rad'] for u in updates),
            planning_safety_regressions=sum(u['safety_regression']['planning'] for u in updates)),
        causal_limit='Window changes are observational; fixed failure does not identify a unique network/control cause.',
        training_unlocked=False,production_qualified=False)
