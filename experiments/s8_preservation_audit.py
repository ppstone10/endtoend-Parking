"""S8旧监督与当前策略的同输入指标、局部梯度及安全查询筛选。"""
import math

import torch
from torch.nn import functional as F

from experiments.behavior_exploration import first_stop


def comparison_metrics(output, sample, threshold):
    points=output.points[0]; logits=output.stop_logits[0]
    target=sample['points']; q=sample['stop']; mask=sample['mask']
    if (points.shape!=target.shape or logits.shape!=q.shape or mask.shape!=q.shape or
            not all(torch.isfinite(t).all() for t in (points,logits,target,q,mask)) or
            not ((q>=0)&(q<=1)).all() or not ((mask==0)|(mask==1)).all() or not mask.any()):
        raise ValueError('指标输入/消费mask非法')
    xy=(points[:,:2]-target[:,:2]).square().sum(-1)
    delta=points[:,2]-target[:,2]; yaw=torch.atan2(delta.sin(),delta.cos()).square()
    xy_mse=float((xy*mask).sum()/mask.sum()); yaw_mse=float((yaw*mask).sum()/mask.sum())
    entropy=-(torch.special.xlogy(q,q)+torch.special.xlogy(1-q,1-q))
    kl=float((F.binary_cross_entropy_with_logits(logits,q,reduction='none')-entropy).mean())
    length=first_stop(logits.sigmoid(),threshold)
    return dict(xy_rmse=math.sqrt(xy_mse),yaw_rmse_deg=math.sqrt(yaw_mse)*180/math.pi,
        pose_loss=xy_mse+10*yaw_mse,stop_kl=kl,total_loss=xy_mse+10*yaw_mse+.2*kl,
        predicted_length=length,target_length=first_stop(q,threshold),
        mask_beyond_prediction=bool(int(torch.nonzero(mask).max())>=length))


def gradient_relation(first_loss, second_loss, parameters):
    parameters=list(parameters)
    first=torch.autograd.grad(first_loss,parameters,retain_graph=True,allow_unused=True)
    second=torch.autograd.grad(second_loss,parameters,allow_unused=True)
    first_sq=second_sq=dot=0.
    for a,b in zip(first,second,strict=True):
        if a is not None: first_sq+=float(a.detach().double().square().sum())
        if b is not None: second_sq+=float(b.detach().double().square().sum())
        if a is not None and b is not None: dot+=float((a.detach().double()*b.detach().double()).sum())
    norm_a=math.sqrt(first_sq);norm_b=math.sqrt(second_sq)
    if not all(math.isfinite(v) for v in (norm_a,norm_b,dot)): raise ValueError('非有限梯度')
    return dict(first_norm=norm_a,second_norm=norm_b,dot=dot,
        cosine=None if norm_a==0 or norm_b==0 else max(-1.,min(1.,dot/(norm_a*norm_b))))


def temporal_clean_queries(samples, windows, *, source_success):
    by_frame={r['frame_index']:r for r in windows}
    if len(by_frame)!=len(windows) or set(by_frame)!={s['frame_index'] for s in samples}:
        raise ValueError('访问帧与安全窗口未对齐')
    from experiments.effective_recovery_data import input_id
    eligible=[];seen=set()
    for s in sorted(samples,key=lambda s:s['frame_index']):
        identity=input_id(s)
        if s['frame_index']>0 and by_frame[s['frame_index']]['clean'] and identity not in seen:
            eligible.append(s);seen.add(identity)
    passed=bool(source_success and len(eligible)>=3)
    chosen=[eligible[int((i+.5)*len(eligible)/3)] for i in range(3)] if passed else []
    return chosen,dict(passed=passed,source_success=bool(source_success),
        unique_clean_nonzero_inputs=len(eligible),selected_frames=[s['frame_index'] for s in chosen])
