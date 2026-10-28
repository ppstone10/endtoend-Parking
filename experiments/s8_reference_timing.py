"""固定dt下的离线双倍时长参考，不修改原节点或运行时。"""
import numpy as np

from interfaces import Trajectory


def double_duration_reference(original):
    points=np.asarray(original.points)
    count=2*(len(points)-1)+1
    if (original.dt!=.2 or points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or count>320 or
            not np.isfinite(points).all()):raise ValueError('仅支持dt0.2且双倍时长H≤320的有限参考')
    delta=points[1:,2].astype(np.float64)-points[:-1,2].astype(np.float64)
    angles=np.arctan2(np.sin(delta),np.cos(delta))
    if (np.abs(np.abs(angles)-np.pi)<1e-6).any():raise ValueError('反向航向中点歧义，不任意选择旋转方向')
    result=np.empty((count,3),dtype=points.dtype);result[::2]=points
    result[1::2,:2]=points[:-1,:2]*.5+points[1:,:2]*.5
    mid=points[:-1,2]+angles/2
    result[1::2,2]=np.arctan2(np.sin(mid),np.cos(mid))
    trajectory=Trajectory(result,original.dt)
    if not np.array_equal(trajectory.points[::2],points):raise ValueError('原参考节点被重写')
    return trajectory
