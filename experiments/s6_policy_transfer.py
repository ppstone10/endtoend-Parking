"""成功策略标签的配对消费与学生续训；不拥有文件或闭环运行。"""
from collections import Counter
from copy import deepcopy

import numpy as np
import torch

from experiments.behavior_preservation import behavior_loss
from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_sampling import TRAINING_SEEDS
from experiments.s6_recovery_learning import batch_identity


ARMS = ('self_targets', 'policy_targets')
STEPS = 28


def validate_policy_samples(samples, teacher_sha):
    if len(samples) != 53 or sorted(s['frame_index'] for s in samples) != list(range(1,54)):
        raise ValueError('必须是53个非零连续成功策略帧')
    if len({input_id(s) for s in samples}) != 53:
        raise ValueError('成功策略输入重复')
    for s in samples:
        if (s['source_index'] != 2149 or s['role'] != 'recovery' or
                s['origin'] != 'validated_success_policy' or
                s['query_policy'] != 'chain120-bias-noncandidate-2149' or
                s['query_step'] != 10*s['frame_index'] or
                s['teacher_checkpoint_sha256'] != teacher_sha):
            raise ValueError('策略来源/身份/相位漂移')
        if (s['points'].shape != (320,3) or s['stop'].shape != (320,) or s['mask'].shape != (320,) or
                not all(torch.isfinite(s[k]).all() for k in ('bev','goal','state','points','stop','mask')) or
                not ((s['stop']>=0)&(s['stop']<=1)).all() or
                not ((s['mask']==0)|(s['mask']==1)).all() or not s['mask'].any()):
            raise ValueError('策略监督尺寸/概率/消费mask非法')


@torch.no_grad()
def build_transfer_schedules(base, samples, parent, seed, check_budget=lambda:None):
    if (seed not in TRAINING_SEEDS or len(base)!=24 or
            any([s['role'] for s in b]!=['anchor','anchor','recovery','recovery'] for b in base)):
        raise ValueError('原24步布局/固定顺序不符')
    validate_policy_samples(samples,samples[0]['teacher_checkpoint_sha256'])
    ordered=sorted(samples,key=lambda s:s['frame_index'])
    if seed==TRAINING_SEEDS[1]:
        ordered=[ordered[i] for i in np.random.default_rng(seed).permutation(len(ordered))]
    parent.eval(); controls={}
    for sample in ordered:
        check_budget()
        out=parent.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
        controls[input_id(sample)]=dict(sample,points=out.points[0].detach().clone(),
                                       stop=out.stop_logits[0].sigmoid().detach().clone())
    plans={a:[] for a in ARMS}
    for j in range(STEPS):
        policy=[ordered[(2*j+k)%len(ordered)] for k in range(2)]
        plans['policy_targets'].append(list(base[j%24])+policy)
        plans['self_targets'].append(list(base[j%24])+[controls[input_id(s)] for s in policy])
    for left,right in zip(plans['self_targets'],plans['policy_targets'],strict=True):
        for a,b in zip(left,right,strict=True):
            if input_id(a)!=input_id(b) or not torch.equal(a['mask'],b['mask']):
                raise ValueError('配对输入或消费mask不同')
    exposed=[s for b in plans['policy_targets'] for s in b[4:]]
    return plans,dict(policy_slots=len(exposed),unique_policy_inputs=len({input_id(s) for s in exposed}),
        frame_order=[s['frame_index'] for s in exposed],legacy_slots=STEPS*4,
        self_label_changes=sum(label_id(controls[input_id(s)])!=label_id(s) for s in samples),
        mask_support='actual_success_policy_consumption_not_parent_executable_claim')


def fit_transfer(parent, optimizer_state, schedule, planned, seed, check_budget, on_update=lambda count:None):
    if (seed not in TRAINING_SEEDS or len(schedule)!=STEPS or len(planned)!=STEPS or
            not optimizer_state['state'] or {int(s['step']) for s in optimizer_state['state'].values()}!={240}):
        raise ValueError('非固定28更新/Adam240起点')
    torch.manual_seed(seed); model=deepcopy(parent).cpu().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    if any(g['lr']!=1e-5 for g in optimizer.param_groups): raise ValueError('学习率漂移')
    history=[]; presented=[]
    for i,batch in enumerate(schedule):
        check_budget()
        if Counter(s['role'] for s in batch)!={'anchor':2,'recovery':4} or batch_identity(batch)!=planned[i]:
            raise ValueError('实际梯度批次角色/输入/标签漂移')
        stack=lambda key:torch.stack([s[key] for s in batch])
        optimizer.zero_grad(set_to_none=True)
        out=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
        loss=behavior_loss(out.points,out.stop_logits,stack('points'),stack('stop'),stack('mask'),[s['role'] for s in batch])
        if not torch.isfinite(loss): raise ValueError('非有限loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        history.append(float(loss.detach())); presented.append(batch_identity(batch))
        on_update(len(history))
    if {int(s['step']) for s in optimizer.state.values()}!={268}: raise ValueError('非Adam268终态')
    model.eval(); return model,optimizer,history,presented
