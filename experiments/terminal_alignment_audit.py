"""只读区分部署停点、真值末点索引与完整自由滚动联合可达点。"""
from collections import Counter
import math

import numpy as np
import torch


def analyze_terminal_alignment(points,logits,targets,masks,goals,metadata,*,threshold,tol_pos=.3,tol_yaw=math.pi/18,indices=None):
    points=np.asarray(points);logits=np.asarray(logits);targets=np.asarray(targets)
    masks=np.asarray(masks);goals=np.asarray(goals)
    if (points.ndim!=3 or points.shape[-1]!=3 or targets.shape!=points.shape
            or logits.shape!=points.shape[:2] or masks.shape!=logits.shape
            or goals.shape!=(len(points),3) or len(metadata)!=len(points) or not len(points)):
        raise ValueError('预测/目标/停止/mask/goal/元数据形状不符')
    if (not all(np.isfinite(x).all() for x in (points,logits,targets,masks,goals))
            or not np.logical_or(masks==0,masks==1).all() or np.any(masks[:,1:]>masks[:,:-1])
            or np.any(masks.sum(1)<=0) or not 0<threshold<1
            or not np.isfinite([tol_pos,tol_yaw]).all() or min(tol_pos,tol_yaw)<=0):
        raise ValueError('输入有限值/前缀/阈值/容差不符')
    indices=list(range(len(points))) if indices is None else list(indices)
    if len(indices)!=len(points) or len(set(indices))!=len(indices):raise ValueError('来源索引重复或数量不符')
    # 与实际predict使用相同dtype的torch.sigmoid，避免阈值附近精度改变首交叉。
    if not np.issubdtype(logits.dtype,np.floating):logits=logits.astype(np.float64)
    probabilities=torch.sigmoid(torch.as_tensor(logits)).numpy()
    def pose_errors(point,goal):
        pos=float(np.linalg.norm(point[:2]-goal[:2]));delta=point[2]-goal[2]
        yaw=float(abs(np.arctan2(np.sin(delta),np.cos(delta))))
        return {'position_m':pos,'yaw_rad':yaw,'joint_valid':bool(pos<=tol_pos and yaw<=tol_yaw)}
    rows=[]
    for j,(pred,target,mask,goal,meta) in enumerate(zip(points,targets,masks,goals,metadata)):
        found=np.flatnonzero(probabilities[j]>=threshold);stop=int(found[0]) if len(found) else len(pred)-1
        length=int(mask.sum());pos=np.linalg.norm(pred[:,:2]-goal[:2],axis=1);delta=pred[:,2]-goal[2]
        yaw=np.abs(np.arctan2(np.sin(delta),np.cos(delta)));valid=np.flatnonzero((pos<=tol_pos)&(yaw<=tol_yaw))
        deployed=pose_errors(pred[stop],goal)
        category='deployment_joint_valid' if deployed['joint_valid'] else ('stop_misaligned_with_joint_point' if len(valid) else 'no_joint_point')
        timing=None
        if category=='stop_misaligned_with_joint_point':timing='early' if stop<valid[0] else ('late' if stop>valid[-1] else 'inside_gap')
        rows.append({'dataset_index':int(indices[j]),'task_id':meta['task_id'],'scene_name':meta['scene_name'],
                     'stop_length':stop+1,'target_length':length,'stop_found':bool(len(found)),
                     'deployment_endpoint':deployed,'teacher_index_prediction':pose_errors(pred[length-1],goal),
                     'label_endpoint':pose_errors(target[length-1],goal),'category':category,'stop_timing':timing,
                     'first_joint_index':int(valid[0]) if len(valid) else None,'last_joint_index':int(valid[-1]) if len(valid) else None,
                     'joint_point_count':len(valid)})
    def group_summary(group):
        counts=Counter(r['category'] for r in group)
        return {'samples':len(group),'categories':{key:counts[key] for key in ('deployment_joint_valid','stop_misaligned_with_joint_point','no_joint_point')},
                'stop_timing':dict(Counter(r['stop_timing'] for r in group if r['stop_timing'])),
                'teacher_index_joint_valid':sum(r['teacher_index_prediction']['joint_valid'] for r in group),
                'label_endpoint_joint_valid':sum(r['label_endpoint']['joint_valid'] for r in group),
                'stop_not_found':sum(not r['stop_found'] for r in group)}
    summary={**group_summary(rows),'by_scene':{name:group_summary([r for r in rows if r['scene_name']==name]) for name in sorted({r['scene_name'] for r in rows})},
             'diagnostic_only':True,'closed_loop_or_geometry_evaluated':False,'automatic_training':False,'automatic_deployment':False}
    return rows,summary
