"""Observe native predictions and original-MPC execution for a paired diagnosis.

Kept in scripts so the existing LT34 execution source inventory stays unchanged.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from controller import MPCController
from experiments.emitted_output_joint import native_lengths
from experiments.recovery_execution import MarginAuditEngine, execution_summary, record_arrays
from experiments.reference_compensation import reference_internal_safe
from interfaces import ControlCmd, Trajectory
from runtime import NetworkSource
from scripts.diagnose_reference_execution import save_episode
from training.reporting import atomic_write_json


class NativePredictionRecorder:
    """Observe the same decoder call used by public predict; no second decoding."""
    def __init__(self, model, budget):
        self.model=model; self.budget=budget; self.raw=[]

    def predict(self, bev, goal, state):
        self.budget.check(); captured=[]
        def observe(module, args, output):
            captured.append((output.points[0].detach().cpu().numpy().copy(),
                             output.stop_logits[0].detach().cpu().numpy().copy()))
        hook=self.model.decoder.register_forward_hook(observe)
        try:
            trajectory=self.model.predict(bev,goal,state)
        finally:
            hook.remove()
        if len(captured)!=1:
            raise ValueError('public predict must produce exactly one decoder output')
        points,logits=captured[0]
        length=int(native_lengths(torch.from_numpy(logits)[None])[0])
        if trajectory.horizon!=length:
            raise ValueError('native stop length mismatch')
        np.testing.assert_array_equal(trajectory.points,points[:length])
        self.raw.append(dict(points=points,logits=logits,bev=np.array(bev.data,copy=True),
            goal=np.array([goal.x,goal.y,goal.yaw],np.float32),
            motion=np.array([state.v,state.omega],np.float32),pose=state.to_array(),length=length))
        return trajectory


class CapturedNetworkSource(NetworkSource):
    def __init__(self,pipeline,model,budget,*,fixed,step):
        super().__init__(pipeline,NativePredictionRecorder(model,budget))
        self.fixed=fixed; self.step=step; self.plans=[]; self.raw=self.model.raw; self.cached=None

    def begin(self,start,goal):
        self.plans.clear(); self.raw.clear(); self.cached=None
        super().begin(start,goal)

    def next_trajectory(self,state):
        self.model.budget.check()
        if self.fixed and self.cached is not None:
            return self.cached,0.
        trajectory,elapsed=super().next_trajectory(state)
        raw=self.raw[-1]
        self.plans.append(dict(step=self.step(),state=state.to_array(),
            global_points=trajectory.points.copy(),local=raw['points'][:raw['length']].copy(),
            bev=raw['bev'],dt=trajectory.dt))
        # Existing independent window exporter uses the key "global".
        self.plans[-1]['global']=self.plans[-1].pop('global_points')
        self.cached=trajectory
        return trajectory,elapsed


def paired_prefix(fixed,rolling):
    shared=min(10,len(fixed['states']),len(rolling['states']))
    if shared<1:
        raise ValueError('no shared executed prefix; attribution is incomplete')
    for name in ('states','commands'):
        if not np.allclose(fixed[name][:shared],rolling[name][:shared],atol=1e-12,rtol=0):
            raise ValueError('fixed/rolling prefix differs before first replan')
    return dict(matched=True,shared_control_steps=shared)


def reference_checks(plan,case):
    trajectory=Trajectory(plan['global'],plan['dt'])
    checks={}
    for name,checker in [('contact',case['engine_kwargs']['collision_checker']),
                         ('planning',case['planner']._collision_checker)]:
        checks[name]=dict(internal_safe=reference_internal_safe(trajectory,checker),
            connection_safe=bool(checker.swept_segment_free(plan['state'][:3],trajectory.points[0])))
    goal=np.array([case['goal'].x,case['goal'].y,case['goal'].yaw])
    terminal=case['engine_kwargs']['terminal']; delta=trajectory.points[-1]-goal
    position=float(np.linalg.norm(delta[:2]));yaw=float(abs(np.arctan2(np.sin(delta[2]),np.cos(delta[2]))))
    return dict(issue_step=plan['step'],length=len(trajectory.points),
        endpoint_position_m=position,endpoint_yaw_rad=yaw,
        endpoint_joint_passed=position<terminal.tol_pos and yaw<terminal.tol_yaw,**checks)


def record_branch(case,model,output,budget,*,fixed):
    """Run unmodified MPC.compute; instance wrappers record only after its return."""
    output=Path(output); frames=[]
    mpc=MPCController(dt=.1,horizon=10,seed=case['index'],**case['vehicle'].mpc_kwargs())
    source=CapturedNetworkSource(deepcopy(case['pipeline']),model,budget,fixed=fixed,step=lambda:len(frames))
    engine=MarginAuditEngine(mpc=mpc,source=source,max_steps=600,replan_every=10,
        planning_checker=case['planner']._collision_checker,
        meta=dict(dataset_index=case['index'],scene_name=case['scene_name']),**case['engine_kwargs'])
    compute=mpc.compute
    def capture(trajectory,state):
        budget.check(); command=compute(trajectory,state)
        window=mpc._reference_points(mpc._aligned_cache,mpc._progress_idx).copy()
        controls=mpc._last_u.copy().reshape(-1,2); predicted=state; rollout=[]
        for v,w in controls:
            predicted=engine.vehicle_model.step(predicted,ControlCmd(float(v),float(w)),mpc.dt)
            rollout.append(predicted.to_array())
        frames.append(dict(before=state.to_array(),window=window,u=controls,rollout=np.array(rollout),
            progress=mpc._progress_idx,aligned_length=len(mpc._aligned_cache),plan_ordinal=len(source.plans)-1))
        return command
    mpc.compute=capture
    try:
        result=engine.run(deepcopy(case['start']),case['goal'])
    finally:
        mpc.compute=compute
    budget.check()
    diagnostic=save_episode(output,engine,case['start'],case['goal'],result,frames,source.plans)
    checks=[reference_checks(p,case) for p in source.plans]
    np.savez_compressed(output.parent/f'{output.name}-predictions.npz',
        **{key:np.stack([r[key] for r in source.raw]) for key in ('points','logits','bev','goal','motion','pose')},
        lengths=np.array([r['length'] for r in source.raw]),
        issue_steps=np.array([p['step'] for p in source.plans]))
    summary=dict(dataset_index=case['index'],task_id=case['task_id'],scene_name=case['scene_name'],
        branch='network_fixed' if fixed else 'network_rolling',
        **execution_summary(result,engine.margin_violations),predictions=len(source.raw),
        first_prediction=checks[0],reference_checks=checks,first_events=diagnostic['first_events'])
    atomic_write_json(output.with_suffix('.json'),dict(**summary,frames=diagnostic['frames'],
        limits=diagnostic['limits'],execution_npz_sha256=diagnostic['npz_sha256']))
    states,commands=record_arrays(result)
    budget.check()
    return summary,dict(states=states,commands=commands,first_raw=source.raw[0],first_global=source.plans[0]['global'])
