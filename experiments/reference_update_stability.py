"""训练域参考更新观察与单次延迟反事实；不改变生产控制器。"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from experiments.recovery_execution import ContextCollector, MarginAuditEngine, execution_summary
from runtime import NetworkSource
from scripts.diagnose_closed_loop_failures import first_blocked_index


def window_geometry(state, window, checker):
    points=np.asarray(window,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('参考窗口无效')
    blocked=0 if not checker.pose_free(*points[0]) else first_blocked_index(checker,points[0],points[1:])
    return {'internal_safe':blocked is None,
            'connection_safe':bool(checker.swept_segment_free(state.to_array()[:3],points[0]))}


def compare_windows(previous, fresh):
    previous,fresh=np.asarray(previous),np.asarray(fresh)
    if previous.shape!=fresh.shape or previous.ndim!=2 or previous.shape[1]!=3 or not len(previous):
        raise ValueError('窗口必须在相同时间网格对齐')
    if not np.isfinite(previous).all() or not np.isfinite(fresh).all():raise ValueError('窗口非有限值')
    xy=np.linalg.norm(fresh[:,:2]-previous[:,:2],axis=1)
    delta=fresh[:,2]-previous[:,2]
    yaw=np.abs(np.arctan2(np.sin(delta),np.cos(delta)))
    return {'xy_rms_m':float(np.sqrt(np.mean(xy*xy))),'xy_max_m':float(xy.max()),
            'yaw_rms_rad':float(np.sqrt(np.mean(yaw*yaw))),'yaw_max_rad':float(yaw.max())}


class ReferenceUpdateCollector(ContextCollector):
    """沿用原快照；仅在私有副本计算新初始化/旧续用窗口，无额外推理。"""
    def run(self,start,goal):
        self.updates=[]
        self.reference_plans=[]
        predict=self.source.next_trajectory
        def observe(state):
            previous=deepcopy(self.mpc)
            trajectory,elapsed=predict(state)
            fresh=deepcopy(self.mpc)
            aligned=fresh._align_dt(trajectory)
            fresh._progress_idx=None
            fresh_index=fresh._advance_progress(aligned,state)
            fresh_window=fresh._reference_points(aligned,fresh_index)
            row={'step':len(self.margin_violations),'has_previous':previous._progress_traj is not None,
                 'fresh_progress_index':fresh_index,
                 'fresh_geometry':{name:window_geometry(state,fresh_window,checker) for name,checker in
                                   (('planning',self.planning_checker),('contact',self.collision_checker))}}
            if row['has_previous']:
                if previous._aligned_cache is None:raise ValueError('已有参考却缺少实际对齐缓存')
                old_index=previous._advance_progress(previous._aligned_cache,state)
                old_window=previous._reference_points(previous._aligned_cache,old_index)
                row.update(previous_progress_index=old_index,**compare_windows(old_window,fresh_window))
                row['previous_geometry']={name:window_geometry(state,old_window,checker) for name,checker in
                                          (('planning',self.planning_checker),('contact',self.collision_checker))}
                row['safety_regression']={name:all(row['previous_geometry'][name].values()) and
                                         not all(row['fresh_geometry'][name].values()) for name in ('planning','contact')}
            self.updates.append(row)
            self.reference_plans.append(np.array(trajectory.points,copy=True))
            return trajectory,elapsed
        self.source.next_trajectory=observe
        try:return super().run(start,goal)
        finally:self.source.next_trajectory=predict


def choose_context(contexts,margin_violations,success):
    nonzero=[c for c in contexts if c.step>0]
    if success:
        if not nonzero:return None,'success_missing_nonzero_safe_context'
        middle=len(margin_violations)/2
        return min(nonzero,key=lambda c:abs(c.step-middle)),'success_control'
    first=next((i for i,x in enumerate(margin_violations) if x),None)
    if first is None:return None,'failure_without_margin_loss'
    prior=[c for c in nonzero if c.step<=first]
    return (prior[-1],'pre_loss_recovery') if prior else (None,'missing_nonzero_pre_loss_context')


class SingleDelayedUpdateSource:
    """首次仍消费同次感知/推理，仅返回原对象；之后全部恢复网络更新。"""
    def __init__(self,primary,previous):
        if previous is None:raise ValueError('单次延迟必须有旧参考')
        self.primary,self.previous=primary,previous
        self.calls=self.delays=0

    def begin(self,start,goal):
        self.calls=self.delays=0
        self.primary.begin(start,goal)

    def next_trajectory(self,state):
        fresh,elapsed=self.primary.next_trajectory(state)
        self.calls+=1
        if self.calls==1:
            self.delays+=1
            return self.previous,elapsed
        return fresh,elapsed


def continue_update(context,model,goal,kwargs,planning_checker,*,delay_once,max_steps,replan_every):
    if context.step<=0 or context.step%replan_every or max_steps<=0:
        raise ValueError('必须从非零、同相位上下文继续')
    mpc,pipeline=deepcopy(context.mpc),deepcopy(context.pipeline)
    source=NetworkSource(pipeline,model)
    if delay_once:source=SingleDelayedUpdateSource(source,mpc._progress_traj)
    engine=MarginAuditEngine(mpc=mpc,source=source,planning_checker=planning_checker,
                            max_steps=max_steps,replan_every=replan_every,**kwargs)
    reset=mpc.reset
    mpc.reset=lambda:None
    try:result=engine.run(deepcopy(context.state),goal)
    finally:mpc.reset=reset
    if delay_once and source.delays!=1:raise ValueError('没有恰好延迟一次更新')
    return result,engine.margin_violations


def result_summary(result,margins,*,smoke=False):
    return {**execution_summary(result,margins,truncated=smoke),
            'planning_failure_details':result.meta.get('planning_failure')}


def summarize(rows,*,smoke=False):
    pairs=[r for r in rows if 'fresh' in r and 'delay_once' in r]
    scenes={}
    for scene in ('S4_dump_area','S8_weigh_station','S1_parking_lot','S2_diagonal_lot'):
        group=[r for r in pairs if r['scene_name']==scene]
        scenes[scene]={'pairs':len(group),'success_controls':sum(r['context_role']=='success_control' for r in group),
                       'pre_loss_recoveries':sum(r['context_role']=='pre_loss_recovery' for r in group),
                       'fresh_successes':sum(r['fresh']['success'] for r in group),
                       'delay_successes':sum(r['delay_once']['success'] for r in group),
                       'delay_margin_safe_wins':sum(not r['fresh']['reached_with_margin'] and r['delay_once']['reached_with_margin'] for r in group),
                       'lost_success_controls':sum(r['fresh']['success'] and not r['delay_once']['success'] for r in group)}
    updates=[u for r in rows for u in r['updates'] if u['has_previous']]
    quantiles={key:None if not updates else {'median':float(np.median([u[key] for u in updates])),
                                             'p95':float(np.percentile([u[key] for u in updates],95))}
               for key in ('xy_rms_m','yaw_rms_rad')}
    covered=(len(pairs)>=8 and scenes['S4_dump_area']['pre_loss_recoveries']>=4
             and scenes['S4_dump_area']['success_controls']>=2
             and all(scenes[s]['success_controls']>=2 for s in ('S1_parking_lot','S2_diagonal_lot')))
    return {'selected_sources':len(rows),'paired_sources':len(pairs),'missing_pairs':len(rows)-len(pairs),
            'scenes':scenes,'update_pairs':len(updates),'window_change_quantiles':quantiles,
            'safety_regressions':{name:sum(u['safety_regression'][name] for u in updates) for name in ('planning','contact')},
            'mechanism_coverage_satisfied':bool(covered and not smoke),
            'decision':'bounded_preflight_only' if smoke else ('mechanism_audit_ready_no_training' if covered else 'insufficient_coverage_no_training'),
            's8_success_control_available':scenes['S8_weigh_station']['success_controls']>0,
            'automatic_training':False,'automatic_deployment':False}
