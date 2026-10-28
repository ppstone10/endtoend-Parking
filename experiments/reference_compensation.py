"""离线教师航向补偿；不接入网络执行，不修改控制器或余量。"""
import numpy as np

from interfaces import Trajectory


def compensate_reference_yaw(original, current, states, progress_trace):
    """按实际消费进度聚合后状态误差，仅修正已访问的原始参考区间。"""
    base=np.asarray(original.points,dtype=np.float64)
    points=np.array(current.points,dtype=np.float64,copy=True)
    states=np.asarray(states,dtype=np.float64)
    trace=np.asarray(progress_trace)
    if (base.shape!=points.shape or len(base)<2 or current.dt!=original.dt
            or not np.array_equal(base[:,:2],points[:,:2])
            or states.ndim!=2 or states.shape[1]!=5 or not len(states)
            or trace.shape!=(len(states),2)
            or not all(np.isfinite(v).all() for v in (base,points,states,trace))
            or (trace[:,1]<1).any() or (trace[:,0]<0).any()
            or (trace[:,0]>=trace[:,1]).any()
            or not np.array_equal(trace,np.floor(trace))):
        raise ValueError('非法参考/实际进度记录')
    phase=trace[:,0]*(len(base)-1)/np.maximum(1,trace[:,1]-1)
    # 对齐MPC的sin/cos插值，而非跨pi线性插值。
    desired=np.arctan2(np.interp(phase,np.arange(len(base)),np.sin(base[:,2])),
                       np.interp(phase,np.arange(len(base)),np.cos(base[:,2])))
    error=np.arctan2(np.sin(states[:,2]-desired),np.cos(states[:,2]-desired))
    bins=np.rint(phase).astype(int)
    counts=np.bincount(bins,minlength=len(base))
    sums=np.bincount(bins,weights=error,minlength=len(base))
    visited=np.flatnonzero(counts)
    interval=np.arange(visited[0],visited[-1]+1)
    values=np.interp(interval,visited,sums[visited]/counts[visited])
    values=np.convolve(np.pad(values,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
    offset=np.arctan2(np.sin(points[:,2]-base[:,2]),np.cos(points[:,2]-base[:,2]))
    offset[interval]=np.clip(offset[interval]-.5*values,-.1,.1)
    points[:,2]=np.arctan2(np.sin(base[:,2]+offset),np.cos(base[:,2]+offset))
    points[[0,-1]]=base[[0,-1]]
    return Trajectory(points,current.dt)


def reference_internal_safe(trajectory, checker):
    """全局参考自身的扫掠检查；不虚构离策略上下文到全局起点的连接。"""
    points=trajectory.points
    return bool(len(points) and np.isfinite(points).all() and checker.pose_free(*points[0])
                and all(checker.swept_segment_free(a,b) for a,b in zip(points[:-1],points[1:])))
