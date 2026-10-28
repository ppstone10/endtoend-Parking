"""固定事件的参考航向响应探针；不优化、执行完整分支或生成监督。"""
from copy import deepcopy

import numpy as np

from interfaces import Trajectory


def yaw_window_basis(raw_count,aligned_count,start,horizon):
    """把下一实际窗口上的余弦脉冲映射到原节点，首尾固定。"""
    values=(raw_count,aligned_count,start,horizon)
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in values):
        raise ValueError('窗口参数必须为整数')
    if raw_count<2 or aligned_count<2 or not 0<=start<aligned_count or horizon<1:
        raise ValueError('非法原节点数/对齐窗口')
    end=min(start+horizon-1,aligned_count-1)
    center=(start+end)/2.;radius=(end-start)/2.+2.
    phase=np.arange(raw_count,dtype=np.float64)*(aligned_count-1)/(raw_count-1)
    distance=np.abs(phase-center)/radius
    basis=np.where(distance<1.,.5*(1.+np.cos(np.pi*np.minimum(distance,1.))),0.)
    basis[[0,-1]]=0.
    return basis


def probe_command(mpc,trajectory,state,amplitude):
    """共享深复制保存轨迹对象关系，只有±0.01rad时重建对齐缓存。"""
    if amplitude not in (0.,-.01,.01):raise ValueError('只允许固定0/±0.01rad响应点')
    if mpc._progress_traj is not trajectory or mpc._progress_idx is None or mpc._aligned_cache is None:
        raise ValueError('必须为已有进度/缓存的事件前MPC及同一参考对象')
    points=np.asarray(trajectory.points)
    if len(points)<2 or not np.isfinite(points).all():raise ValueError('非法参考')
    controller,reference,local_state=deepcopy((mpc,trajectory,state))
    if controller._progress_traj is not reference:raise ValueError('共同复制未保留参考对象身份')
    preview=deepcopy(controller)
    start=preview._advance_progress(preview._aligned_cache,local_state)
    basis=yaw_window_basis(len(points),len(preview._aligned_cache),start,controller.horizon)
    if amplitude:
        updated=reference.points.copy();chosen=basis!=0.
        yaw=updated[chosen,2].astype(np.float64)+amplitude*basis[chosen]
        updated[chosen,2]=np.arctan2(np.sin(yaw),np.cos(yaw))
        reference=Trajectory(updated,reference.dt)
        controller._progress_traj=reference
        controller._aligned_cache=None
    if hasattr(controller,'active_frame'):controller.active_frame=None
    command=controller.compute(reference,local_state)
    if controller._progress_idx!=start:raise ValueError('仅航向扰动却改变xy进度')
    return dict(command=command,trajectory=reference,basis=basis,progress_index=int(start),
        aligned_count=len(controller._aligned_cache))
