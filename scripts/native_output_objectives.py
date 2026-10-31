"""NATIVE-FIT-002: free coordinates plus the deployed first-threshold stop."""
import numpy as np
import torch
from torch.nn import functional as F
from model.network import loss_fn,endpoint_alignment_loss,terminal_pose_loss
from experiments.emitted_output_joint import native_lengths,paired_objective
from scripts.native_output_geometry import NativeGeometry

ARMS=('A','B','AB')
OBJECTIVE=dict(window_logit_margin=.5,window_weight=.2,native_pose_weight=.1,geometry_weight=1.,
    geometry='task_GT_convex_penetration_original_sweep',empty_window='defer_stop_keep_coordinates',
    native_threshold=.5,geometry_length='native_detached',candidate='AB')


def stop_window_loss(logits,indices):
    if logits.ndim!=2 or min(logits.shape)<1 or len(indices)!=len(logits) or not torch.isfinite(logits).all():
        raise ValueError('finite B,H logits and one index per sample required')
    values=[];margin=OBJECTIVE['window_logit_margin']
    for row,index in zip(logits,indices):
        if index is None:values.append(row.sum()*0);continue
        if not isinstance(index,int) or not 0<=index<len(row):raise ValueError('invalid terminal index')
        negative=F.relu(row[:index]+margin).mean() if index else row.sum()*0
        values.append(negative+F.relu(margin-row[index]))
    return torch.stack(values).mean()


def native_pose_loss(points,logits,goals,tolerances):
    length=native_lengths(logits).detach()
    endpoint=points[torch.arange(len(points),device=points.device),length-1]
    delta=endpoint-goals
    yaw=torch.atan2(torch.sin(delta[:,2]),torch.cos(delta[:,2]))
    return (delta[:,:2].square().sum(-1)/tolerances[:,0].square()+yaw.square()/tolerances[:,1].square()).mean()


def objective(model,batch,cases,arm,seed,step):
    if arm not in ARMS:raise ValueError('predeclared A/B/AB only')
    stack=lambda key:torch.stack([s[key] for s in batch])
    free=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
    record=dict(step=step,samples=[dict(source_index=s['source_index'],frame_index=s['frame_index']) for s in batch])
    if arm in ('A','AB'):
        indices=[NativeGeometry(cases[s['source_index']]['planner']._collision_checker).terminal_index(
            free.points[j],s['pose'],s['goal'].numpy(),s['tolerances'].numpy()) for j,s in enumerate(batch)]
        imitation=(loss_fn(free.points,stack('points'),stack('mask'))
            +.5*endpoint_alignment_loss(free.points,stack('points'),stack('mask'),tail_points=8)
            +terminal_pose_loss(free.points,stack('points'),stack('mask'),yaw_weight=10.))
        stopping=stop_window_loss(free.stop_logits,indices)
        terminal=native_pose_loss(free.points,free.stop_logits,stack('goal'),stack('tolerances'))
        total=imitation+OBJECTIVE['window_weight']*stopping+OBJECTIVE['native_pose_weight']*terminal
        record.update(imitation_loss=float(imitation.detach()),window_loss=float(stopping.detach()),
            native_pose_loss=float(terminal.detach()),window_indices=indices,
            stop_deferred=sum(i is None for i in indices))
    else:
        guided=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'),teacher_points=stack('points'),
            teacher_forcing_ratio=.2,sampling_generator=torch.Generator().manual_seed(seed+step))
        total=paired_objective(guided.points,guided.stop_logits,stack('points'),stack('mask'),joint_weight=.1,
            free_points=free.points,free_logits=free.stop_logits,goals=stack('goal'),tolerances=stack('tolerances'))
        record['legacy_joint_loss']=float(total.detach())
    if arm in ('B','AB'):
        lengths=native_lengths(free.stop_logits).detach().tolist()
        safety=torch.stack([NativeGeometry(cases[s['source_index']]['planner']._collision_checker).loss(
            free.points[j,:lengths[j]],s['pose']) for j,s in enumerate(batch)]).mean()
        total=total+OBJECTIVE['geometry_weight']*safety
        record.update(geometry_loss=float(safety.detach()),geometry_native_lengths=lengths)
    record['total_loss']=float(total.detach())
    return total,record


def update(model,optimizer,batch,cases,arm,seed,step):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss,record=objective(model,batch,cases,arm,seed,step)
    if not torch.isfinite(loss):raise ValueError('nonfinite native objective')
    loss.backward()
    if step==1:
        gradients=[p.grad for p in (model.decoder.point_head.weight,model.decoder.stop_head.weight)]
        record['head_gradient_norms']=[float(g.norm()) if g is not None else 0. for g in gradients]
        if not all(np.isfinite(record['head_gradient_norms'])) or record['head_gradient_norms'][0]<=0:
            raise ValueError('coordinate gradient missing/nonfinite')
        if arm in ('A','AB') and record['head_gradient_norms'][1]<=0:raise ValueError('stop gradient missing')
    record['unclipped_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite parameters')
    return record
