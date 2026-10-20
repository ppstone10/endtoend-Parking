"""CEM精修四分支的私有快照执行与联合门槛。"""
from copy import deepcopy

import numpy as np

from experiments.cem_polishing import CEMPolishedMPC
from experiments.recovery_execution import MarginAuditEngine,record_arrays
from experiments.recovery_compatibility import ContinuousExpertSource
from experiments.reference_event_probe import audit_private_solves
from runtime import NetworkSource
from runtime.sources import ReplanningExpertSource


def run_polished_branch(context,source_mode,model,planner,goal,kwargs,*,max_steps,replan_every):
    if source_mode not in {'network','expert_continuous'} or max_steps<=0 or replan_every<=0 or context.step<0 or context.step%replan_every:
        raise ValueError('分支模式/预算/相位不符')
    mpc=CEMPolishedMPC.from_controller(context.mpc)
    source=NetworkSource(deepcopy(context.pipeline),model)
    if source_mode=='expert_continuous':source=ContinuousExpertSource(source,ReplanningExpertSource(deepcopy(planner)))
    engine=MarginAuditEngine(mpc=mpc,source=source,planning_checker=planner._collision_checker,
                            max_steps=max_steps,replan_every=replan_every,**kwargs)
    reset=mpc.reset;mpc.reset=lambda:None
    try:
        with audit_private_solves(mpc,planner._collision_checker,kwargs['collision_checker']) as audit:
            result=engine.run(deepcopy(context.state),goal)
    finally:mpc.reset=reset
    states,_=record_arrays(result)
    if len(audit)!=len(states) or len(mpc.polish_audits)!=len(states):raise ValueError('精修账本长度不符')
    for k,(row,polish) in enumerate(zip(audit,mpc.polish_audits)):
        row['polish']=polish
        np.testing.assert_allclose(row['rollout'][0],states[k,:3],atol=1e-9,rtol=0)
    before=np.vstack([context.state.to_array(),states[:-1]]) if len(states) else states
    details={'solver_class':type(mpc).__name__,'source_mode':source_mode,
             'expert_attempts':getattr(source,'attempts',0),'expert_applied':getattr(source,'applied',0),
             'teacher_references':getattr(source,'references',[]),
             'path_length_m':float(np.linalg.norm(states[:,:2]-before[:,:2],axis=1).sum()),
             'rotation_rad':float(np.abs(states[:,2]-before[:,2]).sum()),
             'polish_evaluations':sum(a['evaluations'] for a in mpc.polish_audits),
             'polish_accepted_steps':sum(a['accepted'] for a in mpc.polish_audits)}
    return result,engine.margin_violations,audit,details


def summarize(rows,*,smoke=False):
    groups={}
    for solver in ('original','polished'):
        for source in ('network','expert_continuous'):
            name=solver+'_'+source
            group=[r['branches'][name] for r in rows]
            groups[name]={'sources':len(group),'successes':sum(b['success'] for b in group),
                          'collisions':sum(b['collision'] for b in group),
                          'strict_successes':sum(b['reached_with_margin'] for b in group)}
    controls=[r for r in rows if r['role']=='success_control']
    retained=sum(r['branches']['polished_network']['success'] for r in controls)
    gate=bool(not smoke and len(rows)==6 and len(controls)==2 and retained==2
              and groups['polished_expert_continuous']['strict_successes']==6
              and groups['polished_network']['collisions']<=groups['original_network']['collisions'])
    return {'sources':len(rows),'variants':groups,'network_success_controls_retained':retained,
            'gate_passed':gate,'decision':'bounded_preflight_only' if smoke else
            ('candidate_for_separate_regression_plan' if gate else 'stop_no_expansion'),
            'automatic_expansion':False,'automatic_training':False,'automatic_deployment':False}
