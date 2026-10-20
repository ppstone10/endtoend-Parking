"""事件对齐的一次参考替换与原MPC实际窗口审计；仅供离线诊断。"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from experiments.recovery_execution import MarginAuditEngine
from experiments.reference_update_stability import SingleDelayedUpdateSource, window_geometry
from runtime import NetworkSource
from runtime.sources import ReplanningExpertSource


def validate_event(row,step,risk_type):
    loss=row['original']['first_margin_violation_step']
    update=next((u for u in row['updates'] if u['step']==step),None)
    if (step<=0 or risk_type not in {'planning','contact'} or loss is None or step>loss
            or update is None or not update['has_previous'] or not update['safety_regression'][risk_type]):
        raise ValueError('不是冻结的首次余量损失前参考安全回退事件')


class OneExpertUpdateSource:
    """首轮仍消费网络输入，仅替换一次；后续完全回到正常网络。"""
    def __init__(self,primary,expert):
        self.primary,self.expert=primary,expert
        self.calls=self.attempts=self.applied=0

    def begin(self,start,goal):
        self.calls=self.attempts=self.applied=0
        self.primary.begin(start,goal)
        self.expert.begin(start,goal)

    def next_trajectory(self,state):
        trajectory,elapsed=self.primary.next_trajectory(state)
        self.calls+=1
        if self.calls==1:
            self.attempts+=1
            trajectory,expert_elapsed=self.expert.next_trajectory(state)
            self.applied+=1
            elapsed+=expert_elapsed
        return trajectory,elapsed


def rollout_controls(state,controls,dt):
    controls=np.asarray(controls,dtype=float).reshape(-1,2)
    pose=np.array([state.x,state.y,state.yaw],dtype=float)
    points=[]
    for v,omega in controls:
        pose=pose+np.array([v*np.cos(pose[2])*dt,v*np.sin(pose[2])*dt,omega*dt])
        points.append(pose.copy())
    return np.asarray(points)


@contextmanager
def audit_private_solves(mpc,planning_checker,contact_checker):
    """不再捕获上下文后才包装私有MPC，避免闭包被复制到其他快照。"""
    solve=mpc._solve
    rows=[]
    def observe(initial,window,state):
        controls=solve(initial,window,state)
        rollout=rollout_controls(state,controls,mpc.dt)
        if not np.isfinite(controls).all() or not np.isfinite(rollout).all():
            raise ValueError('控制序列或rollout非有限值')
        checks={'planning':planning_checker,'contact':contact_checker}
        nearest=int(np.argmin(np.linalg.norm(window[:,:2]-rollout[0,:2],axis=1)))
        pose_probes={name:{'actual_pose_free':bool(checker.pose_free(*rollout[0])),
                           'same_xy_reference_yaw_free':bool(checker.pose_free(rollout[0,0],rollout[0,1],window[nearest,2])),
                           'reference_xy_same_yaw_free':bool(checker.pose_free(window[nearest,0],window[nearest,1],rollout[0,2]))}
                     for name,checker in checks.items()}
        rows.append({'step':len(rows),'state':state.to_array().tolist(),
                     'window':np.asarray(window).tolist(),'controls':np.asarray(controls).tolist(),
                     'rollout':rollout.tolist(),
                     'nearest_reference_index':nearest,'geometric_pose_probes':pose_probes,
                     'cost':float(mpc._batch_cost(np.asarray(controls)[None,:],window,state)[0]),
                     'window_geometry':{name:window_geometry(state,window,checker) for name,checker in checks.items()},
                     'rollout_geometry':{name:window_geometry(state,rollout,checker) for name,checker in checks.items()},
                     'first_transition_safe':{name:bool(checker.swept_segment_free(state.to_array()[:3],rollout[0])) for name,checker in checks.items()}})
        return controls
    mpc._solve=observe
    try:yield rows
    finally:mpc._solve=solve


def run_event_branch(context,mode,model,planner,goal,kwargs,*,max_steps,replan_every):
    if (mode not in {'fresh','delay_once','expert_once'} or replan_every<=0 or context.step<0
            or context.step%replan_every or max_steps<=0):
        raise ValueError('分支模式/预算/相位无效')
    if context.step==0 and mode=='delay_once':raise ValueError('step0没有旧参考，不得伪造延迟')
    mpc,pipeline=deepcopy(context.mpc),deepcopy(context.pipeline)
    source=NetworkSource(pipeline,model)
    if mode=='delay_once':source=SingleDelayedUpdateSource(source,mpc._progress_traj)
    elif mode=='expert_once':source=OneExpertUpdateSource(source,ReplanningExpertSource(deepcopy(planner)))
    engine=MarginAuditEngine(mpc=mpc,source=source,planning_checker=planner._collision_checker,
                            replan_every=replan_every,max_steps=max_steps,**kwargs)
    reset=mpc.reset
    mpc.reset=lambda:None
    try:
        with audit_private_solves(mpc,planner._collision_checker,kwargs['collision_checker']) as audit:
            result=engine.run(deepcopy(context.state),goal)
    finally:mpc.reset=reset
    counts={'expert_attempts':getattr(source,'attempts',0),'expert_applied':getattr(source,'applied',0),
            'delayed_updates':getattr(source,'delays',0)}
    if mode=='delay_once' and counts['delayed_updates']!=1:raise ValueError('未恰好延迟一次')
    if mode=='expert_once' and (counts['expert_attempts']>1 or counts['expert_applied']>1):raise ValueError('专家不应重复调用')
    if len(audit)!=result.steps:raise ValueError('实际控制账本长度不一致')
    for step,row in enumerate(audit):
        np.testing.assert_allclose(row['rollout'][0],result.record.states[step].to_array()[:3],rtol=0,atol=1e-9)
    return result,engine.margin_violations,audit,counts


def summarize_events(rows,*,smoke=False):
    full=[r for r in rows if not r['horizon_limited']]
    early=[r for r in rows if r['horizon_limited']]
    variants={}
    for mode in ('fresh','delay_once','expert_once'):
        pairs=[r for r in full if mode in r['branches']]
        variants[mode]={'sources':len(pairs),'successes':sum(r['branches'][mode]['success'] for r in pairs),
                        'collisions':sum(r['branches'][mode]['collision'] for r in pairs),
                        'strict_margin_recoveries':sum(not r['branches']['fresh']['reached_with_margin'] and r['branches'][mode]['reached_with_margin'] for r in pairs),
                        'lost_successes':sum(r['branches']['fresh']['success'] and not r['branches'][mode]['success'] for r in pairs)}
    return {'sources':len(rows),'event_cohort_sources':len(full),'early_ten_step_sources':len(early),
            'full_task_evaluated_sources':0 if smoke else len(full),'event_cohort_variants':variants,
            'early_planning_safe':{mode:sum(r['branches'][mode]['margin_violation_steps']==0 and r['branches'][mode]['steps']==10 for r in early) for mode in ('fresh','expert_once')},
            'early_expert_planning_failures':sum(r['branches']['expert_once']['planning_failure_details'] is not None for r in early),
            'decision':'bounded_preflight_only' if smoke else 'event_audit_only_no_training',
            'automatic_training':False,'automatic_deployment':False}
