"""只读采集真实MPC消费窗口与固定全局教师的实验适配。"""
from copy import deepcopy
import time

import numpy as np
import torch

from controller import MPCController
from experiments.behavior_preservation import consumed_point_indices
from experiments.recovery_execution import ContextCollector, MarginAuditEngine
from interfaces import Trajectory
from runtime import NetworkSource
from runtime.sources import ReplanningExpertSource
from training.data import to_local


class PreflightBudget:
    def __init__(self,seconds=180.):
        if not 0<seconds<=180:raise ValueError('预检不得超过180秒')
        self.deadline=time.perf_counter()+seconds

    def check(self):
        if time.perf_counter()>=self.deadline:raise TimeoutError('短期预检预算耗尽，不续跑或扩大')


class RecordingMPC(MPCController):
    """控制计算原样调用super，只在计算后给当前监督帧记录消费索引。"""
    def __init__(self,*args,budget,**kwargs):
        super().__init__(*args,**kwargs)
        self.budget=budget
        self.active_frame=None
        self.execution_trace=[]

    def compute(self,trajectory,state):
        self.budget.check()
        command=super().compute(trajectory,state)
        self.execution_trace.append((self._progress_idx,len(self._aligned_cache)))
        frame=self.active_frame
        if frame is not None:
            raw=trajectory.horizon
            used=consumed_point_indices(raw,len(self._aligned_cache),self._progress_idx,self.horizon)
            if frame['kind']=='expert' and frame['offset'] is None:
                frame['offset']=int(used[0])
                remaining=trajectory.points[frame['offset']:]
                if len(remaining)>320:raise ValueError('专家剩余参考超过H320，不截断标签')
                local=to_local(remaining,*frame['pose'])
                frame['points']=np.repeat(local[-1:,:],320,axis=0)
                frame['points'][:len(local)]=local
                frame['length']=len(local)
            offset=frame['offset'] or 0
            local_used=used-offset
            if (local_used<0).any() or (local_used>=320).any():raise ValueError('消费窗口标签对齐失败')
            frame['mask'][local_used]=1.
        return command


class PredictionRecorder:
    def __init__(self,model,budget):
        self.model=model;self.budget=budget;self.pending=None

    def predict(self,bev,goal,state):
        self.budget.check()
        trajectory=self.model.predict(bev,goal,state)
        self.pending={'kind':'network','bev':np.array(bev.data,copy=True),
                      'goal':np.array([goal.x,goal.y,goal.yaw],dtype=np.float32),
                      'state':np.array([state.v,state.omega],dtype=np.float32),
                      'pose':(state.x,state.y,state.yaw),'length':trajectory.horizon,
                      'offset':0,'mask':np.zeros(320,dtype=np.float32)}
        return trajectory


class RecordedNetworkSource(NetworkSource):
    def __init__(self,pipeline,model,mpc,budget):
        super().__init__(pipeline,PredictionRecorder(model,budget))
        self.recording_mpc=mpc;self.frames=[]

    def next_trajectory(self,state):
        trajectory,elapsed=super().next_trajectory(state)
        frame=self.model.pending
        self.frames.append(frame);self.recording_mpc.active_frame=frame
        return trajectory,elapsed


class RecordedFixedExpert:
    """固定归档全局参考只供离线标签验证，不能作为候选网络部署。"""
    def __init__(self,pipeline,trajectory,mpc,budget):
        self.sensor_pipeline=pipeline;self.trajectory=trajectory
        self.recording_mpc=mpc;self.budget=budget;self.frames=[]

    def begin(self,start,goal):
        self.goal=goal;self.sensor_pipeline.set_target_goals([goal])

    def next_trajectory(self,state):
        self.budget.check()
        bev=self.sensor_pipeline.capture_bev(state.x,state.y,state.yaw)
        goal=to_local(np.array([[self.goal.x,self.goal.y,self.goal.yaw]],dtype=np.float32),state.x,state.y,state.yaw)[0]
        frame={'kind':'expert','bev':np.array(bev.data,copy=True),'goal':goal,
               'state':np.array([state.v,state.omega],dtype=np.float32),'pose':(state.x,state.y,state.yaw),
               'mask':np.zeros(320,dtype=np.float32),'offset':None}
        self.frames.append(frame);self.recording_mpc.active_frame=frame
        return self.trajectory,0.


class RecordedReplanningExpert(RecordedFixedExpert):
    """复用现有持续教师，仅在离线分支采集其实际应用参考。"""
    def __init__(self,pipeline,planner,mpc,budget):
        super().__init__(pipeline,None,mpc,budget)
        self.backend=ReplanningExpertSource(deepcopy(planner))

    def begin(self,start,goal):
        super().begin(start,goal)
        self.backend.begin(start,goal)

    def next_trajectory(self,state):
        self.budget.check()
        self.trajectory,elapsed=self.backend.next_trajectory(state)
        trajectory,_=super().next_trajectory(state)
        return trajectory,elapsed


def run_recorded_branch(*,model,pipeline,planner,start,goal,vehicle,engine_kwargs,index,budget,
                        context=None,fixed_trajectory=None,replanning_expert=False):
    budget.check()
    if replanning_expert and (model is not None or fixed_trajectory is not None):
        raise ValueError('持续教师与网络/固定教师不能混用')
    if context is None:
        mpc=RecordingMPC(dt=.1,horizon=10,seed=index,budget=budget,**vehicle.mpc_kwargs())
        active_pipeline=deepcopy(pipeline)
        active_start=start
    else:
        mpc=deepcopy(context.mpc);active_pipeline=deepcopy(context.pipeline);active_start=context.state
        mpc.budget=budget
    source=(RecordedReplanningExpert(active_pipeline,planner,mpc,budget) if replanning_expert else
            RecordedNetworkSource(active_pipeline,model,mpc,budget) if fixed_trajectory is None else
            RecordedFixedExpert(active_pipeline,fixed_trajectory,mpc,budget))
    mpc.execution_trace=[]
    engine_class=ContextCollector if model is not None else MarginAuditEngine
    engine=engine_class(mpc=mpc,source=source,planning_checker=planner._collision_checker,
                       replan_every=10,max_steps=600-(0 if context is None else context.step),**engine_kwargs)
    reset=mpc.reset
    if context is not None:mpc.reset=lambda:None
    try:result=engine.run(deepcopy(active_start),goal)
    finally:mpc.reset=reset
    return result,engine,source.frames


def prepare_supervision_frames(frames,teacher,role,source_index,*,frame_limit=4):
    if not frames:return []
    if frame_limit is not None and (isinstance(frame_limit,bool) or not isinstance(frame_limit,int) or frame_limit<1):
        raise ValueError('帧上限必须为正整数或None')
    # 默认仍为原4帧；全部帧仅由独立探索入口显式请求。
    chosen=(np.arange(len(frames)) if frame_limit is None else
            np.unique(np.linspace(0,len(frames)-1,min(frame_limit,len(frames)),dtype=int)))
    samples=[]
    for frame_index in chosen:
        frame=frames[frame_index]
        if not frame['mask'].any():raise ValueError('监督帧未被实际MPC消费')
        bev=torch.from_numpy(frame['bev']).float();goal=torch.from_numpy(frame['goal']).float()
        state=torch.from_numpy(frame['state']).float()
        if frame['kind']=='network':
            with torch.no_grad():output=teacher.forward_with_stop(bev[None],goal[None],state[None])
            points=output.points[0].detach().clone();stop=output.stop_logits[0].sigmoid().detach()
            hits=torch.nonzero(stop>=teacher.stop_threshold).flatten()
            if (int(hits[0])+1 if len(hits) else 320)!=frame['length']:raise ValueError('监督未复现实际首停')
        else:
            points=torch.from_numpy(frame['points']).float()
            stop=(torch.arange(320)>=frame['length']-1).float()
        samples.append({'bev':bev,'goal':goal,'state':state,'points':points,'stop':stop,
                        'mask':torch.from_numpy(frame['mask']).float(),'role':role,
                        'source_index':source_index,'frame_index':int(frame_index)})
    return samples
