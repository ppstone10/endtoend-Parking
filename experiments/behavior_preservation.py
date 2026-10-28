"""有界预检的监督与收益门禁；不依赖运行时或文件系统。"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch.nn import functional as F


def consumed_point_indices(raw_count, aligned_count, progress, horizon):
    """保留MPC时间插值的两端原始点，不假定只消费网络前20点。"""
    if min(raw_count,aligned_count,horizon)<1 or not 0<=progress<aligned_count:
        raise ValueError('非法MPC参考范围')
    if aligned_count==1:return np.array([0],dtype=int)
    positions=np.minimum(np.arange(progress,progress+horizon),aligned_count-1)*(raw_count-1)/(aligned_count-1)
    return np.unique(np.r_[np.floor(positions),np.ceil(positions)].astype(int))


def coverage_gate(rows):
    if len({r['dataset_index'] for r in rows})!=len(rows):raise ValueError('拟合来源重复')
    anchors=sum(r['supervision']=='anchor' for r in rows)
    scenes={scene:sum(r['scene_name']==scene and r['supervision'] in {'improve','recovery'}
                     and r['teacher_strict'] for r in rows) for scene in ('S4_dump_area','S8_weigh_station')}
    return {'anchors':anchors,'strict_improvement_sources':scenes,
            'passed':bool(anchors and all(scenes.values()))}


def probe_gate(baseline,candidate):
    def mapping(rows):
        values={r['dataset_index']:r for r in rows}
        if not values or len(values)!=len(rows):raise ValueError('空/重复探针')
        return values
    old,new=mapping(baseline),mapping(candidate)
    if old.keys()!=new.keys():raise ValueError('探针分母改变')
    for i in old:
        if any(old[i][k]!=new[i][k] for k in ('task_id','scene_name')):raise ValueError('探针身份改变')
    ds=sum(r['success'] for r in new.values())-sum(r['success'] for r in old.values())
    dc=sum(r['collision'] for r in new.values())-sum(r['collision'] for r in old.values())
    lost=[i for i in old if old[i]['success'] and not new[i]['success']]
    gained=[i for i in old if not old[i]['success'] and new[i]['success']]
    passed=ds>=1 and dc<=-1 and not lost
    return {'success_delta':ds,'collision_delta':dc,'lost_success_indices':lost,
            'new_success_indices':gained,'passed':passed,'main_protocol_passed':False}


def behavior_loss(points,logits,target_points,stop_probabilities,point_mask,roles):
    if (points.ndim!=3 or points.shape[-1]!=3 or target_points.shape!=points.shape
            or logits.shape!=points.shape[:2] or stop_probabilities.shape!=logits.shape
            or point_mask.shape!=logits.shape or len(roles)!=len(points)
            or not set(roles)<={'anchor','improve','recovery'}
            or not all(torch.isfinite(x).all() for x in (points,logits,target_points,stop_probabilities,point_mask))
            or ((stop_probabilities<0)|(stop_probabilities>1)).any()
            or not ((point_mask==0)|(point_mask==1)).all() or (point_mask.sum(1)==0).any()):
        raise ValueError('非法保持监督')
    xy=(points[...,:2]-target_points[...,:2]).square().sum(-1)
    delta=points[...,2]-target_points[...,2]
    yaw=torch.atan2(torch.sin(delta),torch.cos(delta)).square()
    pose=((xy+10*yaw)*point_mask).sum(1)/point_mask.sum(1)
    q=stop_probabilities.detach()
    entropy=-(torch.special.xlogy(q,q)+torch.special.xlogy(1-q,1-q))
    stop=(F.binary_cross_entropy_with_logits(logits,q,reduction='none')-entropy).mean(1)
    per_sample=pose+.2*stop
    # 每种监督角色等权，不能让长成功轨迹数量压没少量困难恢复。
    return torch.stack([per_sample[torch.tensor([r==role for r in roles],device=points.device)].mean()
                        for role in sorted(set(roles))]).mean()


def fit_bounded_copy(parent,samples,*,steps=12,learning_rate=1e-5,check_budget=lambda:None):
    if not samples or steps!=12 or not math.isfinite(learning_rate) or learning_rate!=1e-5:
        raise ValueError('只允许预声明12步短拟合')
    model=deepcopy(parent).cpu();model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    history=[];roles=sorted({s['role'] for s in samples})
    for step in range(steps):
        check_budget()
        selected=[]
        for role in roles:
            pool=[s for s in samples if s['role']==role]
            selected.extend(pool[(step*2+j)%len(pool)] for j in range(2))
        stack=lambda key:torch.stack([s[key] for s in selected])
        optimizer.zero_grad(set_to_none=True)
        output=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
        loss=behavior_loss(output.points,output.stop_logits,stack('points'),stack('stop'),stack('mask'),[s['role'] for s in selected])
        if not torch.isfinite(loss):raise ValueError('短拟合loss非有限')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    return model,history
