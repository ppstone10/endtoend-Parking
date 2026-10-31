"""NATIVE-FIT-001: task-GT footprint penetration, with the original hard oracle.

The differentiable value is a training penalty, never a safety certificate.
Only the frozen cohort's convex polygons, rectangles and circles are supported.
"""
import math
import numpy as np
import torch
from planner import RectangleFootprintCollisionChecker
from runtime import NetworkSource
from interfaces import VehicleState
from sim import RectangleObstacle,CircleObstacle,PolygonObstacle
from scripts.batched_footprint_check import BatchedSafety


class NativeGeometry:
    def __init__(self,checker):
        if type(checker) is not RectangleFootprintCollisionChecker:
            raise ValueError('original rectangle footprint checker required')
        self.checker=checker
        self.oracle=BatchedSafety(checker,None,0.,0.,0.)
        for obstacle in checker.env.obstacles:
            if not obstacle.forbidden:continue
            if type(obstacle) not in (RectangleObstacle,CircleObstacle,PolygonObstacle):
                raise ValueError('unsupported task geometry; never assume free')
            if type(obstacle) is PolygonObstacle:
                vertices=np.asarray(obstacle.vertices,dtype=float)
                edges=np.roll(vertices,-1,axis=0)-vertices
                turn=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
                area=abs(np.sum(vertices[:,0]*np.roll(vertices,-1,axis=0)[:,1]-vertices[:,1]*np.roll(vertices,-1,axis=0)[:,0]))
                if (area<1e-9 or np.any(np.linalg.norm(edges,axis=1)<1e-9)
                        or not (np.all(turn>=-1e-9) or np.all(turn<=1e-9))):
                    raise ValueError('convex nondegenerate polygons required')

    def pose_penetration(self,poses):
        """Signed separation axes/circle-box SDF; positive means sampled risk."""
        x,y,yaw=poses.unbind(-1);c,s=torch.cos(yaw),torch.sin(yaw)
        hl,hw=self.checker.half_length,self.checker.half_width
        ex,ey=c.abs()*hl+s.abs()*hw,s.abs()*hl+c.abs()*hw
        world=self.checker.env.world_size/2
        risks=[torch.relu(torch.stack((x.abs()+ex-world,y.abs()+ey-world),-1).amax(-1))]
        local=poses.new_tensor([[hl,hw],[hl,-hw],[-hl,-hw],[-hl,hw]])
        corners=torch.stack((x[:,None]+c[:,None]*local[:,0]-s[:,None]*local[:,1],
                             y[:,None]+s[:,None]*local[:,0]+c[:,None]*local[:,1]),-1)
        vehicle_axes=torch.stack((torch.stack((c,s),-1),torch.stack((-s,c),-1)),1)
        for obstacle in self.checker.env.obstacles:
            if not obstacle.forbidden:continue
            if type(obstacle) is CircleObstacle:
                dx,dy=obstacle.x-x,obstacle.y-y
                q=torch.stack(((dx*c+dy*s).abs()-hl,(-dx*s+dy*c).abs()-hw),-1)
                distance=torch.linalg.vector_norm(torch.relu(q),dim=-1)+torch.minimum(q.amax(-1),torch.zeros_like(x))
                risks.append(torch.relu(obstacle.radius-distance+1e-9));continue
            if type(obstacle) is RectangleObstacle:
                a,b,d,e=obstacle.bbox;vertices=poses.new_tensor([[a,d],[b,d],[b,e],[a,e]])
            else:vertices=poses.new_tensor(obstacle.vertices)
            edges=torch.roll(vertices,-1,0)-vertices
            normals=torch.stack((-edges[:,1],edges[:,0]),-1)
            normals=normals/torch.linalg.vector_norm(normals,dim=-1,keepdim=True)
            axes=torch.cat((vehicle_axes,normals[None].expand(len(poses),-1,-1)),1)
            own=torch.einsum('pvc,pac->pav',corners,axes)
            other=torch.einsum('vc,pac->pav',vertices,axes)
            gaps=torch.maximum(own.amin(-1)-other.amax(-1),other.amin(-1)-own.amax(-1))
            risks.append(torch.relu(1e-9-gaps.amax(-1)))
        return torch.stack(risks,-1).amax(-1)

    def sampled_segments(self,points,pose):
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not torch.isfinite(points).all():
            raise ValueError('finite nonempty native points required')
        pose=np.asarray(pose,dtype=np.float64)
        if pose.shape!=(3,) or not np.isfinite(pose).all():raise ValueError('finite pose required')
        raw=points.detach().cpu().numpy()
        native=NetworkSource._to_global(raw,VehicleState(*pose,0.,0.)).astype(np.float64)
        starts=np.vstack((pose,native[:-1]));delta=native-starts
        delta[:,2]=(delta[:,2]+np.pi)%(2*np.pi)-np.pi
        ratio=(np.hypot(delta[:,0],delta[:,1])+self.checker.corner_radius*np.abs(delta[:,2]))/self.checker.resolution
        if not np.isfinite(ratio).all() or (ratio>10000).any():raise ValueError('sweep resource limit; no silent clipping')
        count=np.maximum(1,np.ceil(ratio).astype(int))
        if int((count+1).sum())>200000:
            raise ValueError('sweep resource limit; no silent clipping')
        owner=np.repeat(np.arange(len(points)),count+1)
        fraction=np.concatenate([np.arange(n+1,dtype=np.float64)/n for n in count])
        p=points.to(torch.float64);c,s=math.cos(pose[2]),math.sin(pose[2])
        world=torch.stack((pose[0]+c*p[:,0]-s*p[:,1],pose[1]+s*p[:,0]+c*p[:,1],p[:,2]+pose[2]),-1)
        # Match native float32 global rounding in forward, preserving the analytic
        # local->world derivative instead of differentiating a NumPy conversion.
        world=world+(world.new_tensor(native)-world).detach()
        before=torch.cat((world.new_tensor(pose)[None],world[:-1]),0)
        change=world-before
        angle=torch.remainder(change[:,2]+np.pi,2*np.pi)-np.pi
        change=torch.cat((change[:,:2],angle[:,None]),-1)
        index=torch.as_tensor(owner,device=points.device)
        sampled=before[index]+change[index]*world.new_tensor(fraction)[:,None]
        sampled=torch.cat((sampled[:,:2],(torch.remainder(sampled[:,2]+np.pi,2*np.pi)-np.pi)[:,None]),-1)
        return sampled,owner,count

    def safe_prefix(self,points,pose):
        sampled,owner,count=self.sampled_segments(points,pose)
        free=self.oracle.poses_free(sampled.detach().cpu().numpy())
        offsets=np.r_[0,np.cumsum(count+1)[:-1]]
        segments=np.logical_and.reduceat(free,offsets)
        return np.logical_and.accumulate(segments)

    def loss(self,points,pose):
        sampled,_,_=self.sampled_segments(points,pose)
        return self.pose_penetration(sampled).amax()

    def terminal_index(self,points,pose,goal,tolerances):
        delta=points.detach().cpu().numpy().astype(np.float64)-np.asarray(goal,dtype=np.float64)
        tol=np.asarray(tolerances,dtype=np.float64)
        good=(np.linalg.norm(delta[:,:2],axis=1)<tol[0])&(np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])))<tol[1])
        valid=np.flatnonzero(good&self.safe_prefix(points.detach(),pose))
        return int(valid[0]) if len(valid) else None
