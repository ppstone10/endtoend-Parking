"""完整已执行恢复参考的监督边界；不拥有运行时或文件。"""
import numpy as np
import torch
from torch.nn import functional as F

from experiments.behavior_preservation import behavior_loss,consumed_point_indices
from model.network import variable_loss_fn,terminal_pose_loss


def complete_consumption_mask(raw_count,offset,length,traces,*,horizon=10):
    if not 0<=offset<raw_count or length!=raw_count-offset or not 1<=length<=320 or not traces:
        raise ValueError('剩余参考长度/实际窗口证据不符')
    used=np.unique(np.concatenate([consumed_point_indices(raw_count,aligned,progress,horizon)
                                   for progress,aligned in traces]))-offset
    if not np.array_equal(used,np.arange(length)):
        raise ValueError('实际消费并集不是完整连续剩余参考')
    return (torch.arange(320)<length).float()


def full_reference_joint_loss(points,logits,sample):
    length=int(sample['mask'].sum())
    if (sample['role']!='recovery' or not 1<=length<=320 or
            not torch.equal(sample['mask'],(torch.arange(320)<length).float()) or
            not torch.equal(sample['stop'],(torch.arange(320)>=length-1).float())):
        raise ValueError('完整恢复参考的mask/累计停止/角色不一致')
    target=sample['points'][None]; mask=sample['mask'][None]; stop=sample['stop'][None]
    base=behavior_loss(points,logits,target,stop,mask,['recovery'])
    new_stop=variable_loss_fn(target,logits,target,mask,stop_weight=.2,balance_stop=True,
                             stop_target_mode='terminal',yaw_error_mode='wrapped')
    terminal=terminal_pose_loss(points,target,mask,yaw_weight=10,yaw_error_mode='wrapped')
    return base+(new_stop-.2*F.binary_cross_entropy_with_logits(logits,stop)+terminal)


def pair_schedule(first,second):
    if first.get('query_step')!=400 or second.get('query_step')!=410:
        raise ValueError('只能使用预定400/410状态对')
    return [first if i%2==0 else second for i in range(120)]


def reference_at_query(record,start_step,query_step):
    offset=query_step-start_step
    if offset<0 or offset%10 or offset>=len(record.refs):
        raise ValueError('查询不在逐控制步参考记录范围内')
    return np.asarray(record.refs[offset]).copy()


def reference_audit(reference,state,case,check_budget=lambda:None):
    points=np.asarray(reference,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('参考非法')
    chain=np.vstack((state.to_array()[:3],points)); risks={}
    for name,checker in [('planning',case['planner']._collision_checker),('contact',case['engine_kwargs']['collision_checker'])]:
        indices=[]
        for i,(a,b) in enumerate(zip(chain[:-1],chain[1:],strict=True)):
            check_budget()
            if not checker.swept_segment_free(a,b): indices.append(i)
        risks[name]=indices
    goal=case['goal']; d=points[-1]-[goal.x,goal.y,goal.yaw]
    return dict(length=len(points),unsafe_segment_end_raw_indices=risks,
        endpoint_pos_err=float(np.linalg.norm(d[:2])),
        endpoint_yaw_err_deg=float(abs(np.arctan2(np.sin(d[2]),np.cos(d[2])))*180/np.pi))
