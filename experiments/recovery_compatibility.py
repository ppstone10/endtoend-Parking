"""持续教师恢复与纯几何控制代价的隔离执行。"""
from copy import deepcopy

import numpy as np

from controller import MPCController
from experiments.footprint_yaw_mpc import FootprintYawMPC
from experiments.recovery_execution import MarginAuditEngine,record_arrays
from experiments.reference_event_probe import OneExpertUpdateSource,audit_private_solves
from interfaces import VehicleState
from runtime import NetworkSource
from runtime.sources import ReplanningExpertSource


class ContinuousExpertSource:
    """每次消费原感知/推理，但只执行同状态专家参考；仅记录不导出标签。"""
    def __init__(self,primary,expert):
        self.primary,self.expert=primary,expert
        self.attempts=self.applied=0
        self.references=[]

    def begin(self,start,goal):
        self.attempts=self.applied=0
        self.references=[]
        self.primary.begin(start,goal)
        self.expert.begin(start,goal)

    def next_trajectory(self,state):
        _,elapsed=self.primary.next_trajectory(state)
        self.attempts+=1
        trajectory,expert_elapsed=self.expert.next_trajectory(state)
        self.applied+=1
        self.references.append({'state':state.to_array().tolist(),'points':trajectory.points.tolist(),'dt':trajectory.dt})
        return trajectory,elapsed+expert_elapsed


def run_compatibility(context,source_mode,solver,model,planner,goal,kwargs,*,vehicle_length,
                      vehicle_width,max_steps,replan_every):
    if (source_mode not in {'network','expert_once','expert_continuous'} or solver not in {'original','footprint_yaw'}
            or replan_every<=0 or context.step<0 or context.step%replan_every or max_steps<=0):
        raise ValueError('来源/控制器/相位/预算无效')
    if solver=='footprint_yaw' and (context.step!=0 or max_steps>10 or source_mode=='expert_continuous'):
        raise ValueError('本轮几何cost只允许预声明初态10步双参考')
    mpc=deepcopy(context.mpc) if solver=='original' else FootprintYawMPC.from_controller(
        context.mpc,vehicle_length=vehicle_length,vehicle_width=vehicle_width)
    primary=NetworkSource(deepcopy(context.pipeline),model)
    expert=ReplanningExpertSource(deepcopy(planner))
    source=primary
    if source_mode=='expert_once':source=OneExpertUpdateSource(primary,expert)
    elif source_mode=='expert_continuous':source=ContinuousExpertSource(primary,expert)
    engine=MarginAuditEngine(mpc=mpc,source=source,planning_checker=planner._collision_checker,
                            max_steps=max_steps,replan_every=replan_every,**kwargs)
    reset=mpc.reset;mpc.reset=lambda:None
    try:
        with audit_private_solves(mpc,planner._collision_checker,kwargs['collision_checker']) as audit:
            result=engine.run(deepcopy(context.state),goal)
    finally:mpc.reset=reset
    for row in audit:
        legacy=float(MPCController._batch_cost(mpc,np.array(row['controls'])[None,:],
                         np.array(row['window']),VehicleState.from_array(np.array(row['state'])))[0])
        row['legacy_cost']=legacy
        row['orientation_cost_increment']=row['cost']-legacy
    states,commands=record_arrays(result)
    before=np.vstack([context.state.to_array(),states[:-1]]) if len(states) else states
    path=float(np.linalg.norm(states[:,:2]-before[:,:2],axis=1).sum())
    rotation=float(np.abs(states[:,2]-before[:,2]).sum())
    radius_squared=(vehicle_length**2+vehicle_width**2)/4
    def goal_error(pose):
        return float(np.sqrt((pose[0]-goal.x)**2+(pose[1]-goal.y)**2+4*radius_squared*np.sin((pose[2]-goal.yaw)/2)**2))
    progress=goal_error(context.state.to_array())-goal_error(states[-1] if len(states) else context.state.to_array())
    details={'solver_class':type(mpc).__name__,'source_mode':source_mode,
             'expert_attempts':getattr(source,'attempts',0),'expert_applied':getattr(source,'applied',0),
             'teacher_references':getattr(source,'references',[]),'path_length_m':path,
             'rotation_rad':rotation,'equivalent_motion_m':path+np.hypot(vehicle_length,vehicle_width)/2*rotation}
    details['goal_pose_progress_m']=progress
    if len(audit)!=len(states):raise ValueError('实际控制账本长度不符')
    for k,row in enumerate(audit):np.testing.assert_allclose(row['rollout'][0],states[k,:3],atol=1e-9,rtol=0)
    return result,engine.margin_violations,audit,details


def summarize_compatibility(rows,*,smoke=False):
    recovery=[r for r in rows if r['cohort']=='S4' and r['role']=='safety_regression_event']
    controls=[r for r in rows if r['cohort']=='S4' and r['role']=='success_control']
    early=[r for r in rows if r['cohort']=='S8']
    safety={}
    for source in ('network','expert_once'):
        baseline=[r['branches']['original_'+source] for r in early]
        trial=[r['branches']['footprint_yaw_'+source] for r in early]
        eligible=[b['steps']==10 and b['margin_violation_steps']==0 and b['equivalent_motion_m']>=1e-5
                  and b['goal_pose_progress_m']>=max(1e-5,.8*a['goal_pose_progress_m'])
                  for a,b in zip(baseline,trial)]
        safety[source]={'sources':len(early),'baseline_margin_safe':sum(b['steps']==10 and b['margin_violation_steps']==0 for b in baseline),
                        'trial_margin_safe':sum(b['steps']==10 and b['margin_violation_steps']==0 for b in trial),
                        'trial_safe_with_progress_floor':sum(eligible)}
    return {'sources':len(rows),'s4_recovery_sources':len(recovery),'s4_success_controls':len(controls),
            's4_teacher_strict_recoveries':sum(r['branches']['original_expert_continuous']['reached_with_margin'] for r in recovery),
            's4_control_teacher_successes':sum(r['branches']['original_expert_continuous']['success'] for r in controls),
            'original_success_trajectories_not_relabelled':True,'student_success_retention_not_evaluated':True,
            's8_ten_step_comparisons':safety,
            'decision':'bounded_preflight_only' if smoke else 'separate_mechanism_review_no_training',
            'automatic_training':False,'automatic_expansion':False,'automatic_deployment':False}
