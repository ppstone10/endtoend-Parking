"""矩形SAT有符号间隙，仅为离线搜索提供排名；安全判定仍用原检查器。"""
import numpy as np

from sim.obstacles import RectangleObstacle


def signed_rectangle_gap(poses,checker):
    """单位分离轴间隙的最小障碍值；符号对应几何、数值不是欧氏距离。"""
    poses=np.asarray(poses,dtype=np.float64)
    if poses.ndim!=2 or poses.shape[1]!=3 or not np.isfinite(poses).all():raise ValueError('非法位姿')
    obstacles=[o for o in checker.env.obstacles if o.forbidden]
    if any(not isinstance(o,RectangleObstacle) for o in obstacles):raise ValueError('辅助间隙只支持矩形实体障碍')
    x,y,yaw=poses.T;c=np.cos(yaw);s=np.sin(yaw);ac=np.abs(c);ass=np.abs(s)
    extent_x=ac*checker.half_length+ass*checker.half_width
    extent_y=ass*checker.half_length+ac*checker.half_width
    half=checker.env.world_size/2.
    gap=np.minimum(half-np.abs(x)-extent_x,half-np.abs(y)-extent_y)
    for obstacle in obstacles:
        hx=(obstacle.x_max-obstacle.x_min)/2.;hy=(obstacle.y_max-obstacle.y_min)/2.
        dx=(obstacle.x_min+obstacle.x_max)/2.-x;dy=(obstacle.y_min+obstacle.y_max)/2.-y
        axes=np.stack((np.abs(dx)-extent_x-hx,np.abs(dy)-extent_y-hy,
            np.abs(dx*c+dy*s)-checker.half_length-hx*ac-hy*ass,
            np.abs(-dx*s+dy*c)-checker.half_width-hx*ass-hy*ac))
        # 原矩形SAT将接触及1e-9范围内的障碍间隙计为碰撞。
        gap=np.minimum(gap,np.max(axes,axis=0)-1e-9)
    return gap


def swept_penetration_summary(start,states,checker,margin_flags):
    states=np.asarray(states,dtype=np.float64);start=np.asarray(start,dtype=np.float64)
    flags=np.asarray(margin_flags,dtype=bool)
    if (states.ndim!=2 or states.shape[1] not in (3,5) or flags.shape!=(len(states),)
            or start.ndim!=1 or len(start) not in (3,5) or not np.isfinite(states).all()
            or not np.isfinite(start).all()):raise ValueError('非法实际转移/余量flags')
    poses=states[:,:3];previous=np.vstack((start[:3],poses[:-1]));sampled=[]
    for at in np.flatnonzero(flags):
        first=previous[at];last=poses[at];delta=last-first
        delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
        distance=np.hypot(delta[0],delta[1])+checker.corner_radius*abs(delta[2])
        steps=max(1,int(np.ceil(distance/checker.resolution)))
        sampled.append(first+np.linspace(0.,1.,steps+1)[:,None]*delta)
    if not sampled:return dict(max_penetration=0.,sum_sq_penetration=0.,sampled_points=0)
    gap=signed_rectangle_gap(np.concatenate(sampled),checker);penetration=np.maximum(0.,-gap)
    return dict(max_penetration=float(penetration.max()),sum_sq_penetration=float(np.square(penetration).sum()),
        sampled_points=len(gap),minimum_signed_proxy=float(gap.min()))
