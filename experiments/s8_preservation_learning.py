"""保留原六样本目标，额外保持已验证C240成功策略；不拥有文件。"""
from collections import Counter
from copy import deepcopy

import numpy as np
import torch

from experiments.behavior_preservation import behavior_loss
from experiments.effective_recovery_data import input_id
from experiments.s6_policy_transfer import STEPS
from experiments.s6_recovery_learning import batch_identity
from experiments.retention_sampling import TRAINING_SEEDS


PRESERVATION_WEIGHT=.25


def preservation_schedule(by_source,seed):
    if seed not in TRAINING_SEEDS or not by_source or any(len(v)!=3 for v in by_source.values()):
        raise ValueError('保持池须为固定顺序/每源3真实查询')
    sources=sorted(by_source)
    if seed==TRAINING_SEEDS[1]:sources=[sources[i] for i in np.random.default_rng(seed).permutation(len(sources))]
    frames={i:sorted(by_source[i],key=lambda s:s['frame_index']) for i in sources}
    if any(s['source_index']!=i or s['role']!='anchor' or s['frame_index']<=0 for i in sources for s in frames[i]):
        raise ValueError('非真实非零保持查询')
    all_samples=[s for v in frames.values() for s in v]
    if len({input_id(s) for s in all_samples})!=len(all_samples):raise ValueError('保持查询输入重复')
    visits=Counter();flat=[]
    for position in range(STEPS*2):
        index=sources[position%len(sources)];flat.append(frames[index][visits[index]%3]);visits[index]+=1
    if {input_id(s) for s in flat}!={input_id(s) for s in all_samples}:raise ValueError('预算无法覆盖全部保持输入')
    return [flat[i:i+2] for i in range(0,len(flat),2)]


def batch_loss(model,batch):
    stack=lambda key:torch.stack([s[key] for s in batch])
    out=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
    return behavior_loss(out.points,out.stop_logits,stack('points'),stack('stop'),stack('mask'),[s['role'] for s in batch])


def preservation_objective(model,base,queries,*,weight=PRESERVATION_WEIGHT):
    if weight not in (0.,PRESERVATION_WEIGHT):raise ValueError('不允许扫描保持系数')
    primary=batch_loss(model,base)
    if weight==0:return primary,primary,primary.detach()*0
    was_training=model.training
    try:
        model.eval()
        preservation=batch_loss(model,queries)
    finally:model.train(was_training)
    return primary+weight*preservation,primary,preservation


def fit_preservation(parent,state,base,queries,planned_base,planned_queries,seed,budget_check,on_update=lambda count:None):
    if (seed not in TRAINING_SEEDS or any(len(v)!=STEPS for v in (base,queries,planned_base,planned_queries)) or
            not state['state'] or {int(v['step']) for v in state['state'].values()}!={240}):
        raise ValueError('非原Adam240/28更新计划')
    torch.manual_seed(seed);model=deepcopy(parent).cpu().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5);optimizer.load_state_dict(deepcopy(state))
    if any(g['lr']!=1e-5 for g in optimizer.param_groups):raise ValueError('学习率漂移')
    history=[];presented=[]
    for i,(primary,keep) in enumerate(zip(base,queries,strict=True)):
        budget_check()
        if (batch_identity(primary)!=planned_base[i] or batch_identity(keep)!=planned_queries[i] or
                Counter(s['role'] for s in primary)!={'anchor':2,'recovery':4} or
                [s['role'] for s in keep]!=['anchor','anchor']):raise ValueError('实际主/保持梯度批次身份漂移')
        optimizer.zero_grad(set_to_none=True)
        loss,base_loss,keep_loss=preservation_objective(model,primary,keep)
        if not torch.isfinite(loss):raise ValueError('非有限保持loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        history.append(dict(total=float(loss.detach()),primary=float(base_loss.detach()),preservation=float(keep_loss.detach()),
            weight=PRESERVATION_WEIGHT,gradient_norm=float(norm)))
        presented.append(dict(primary=batch_identity(primary),preservation=batch_identity(keep)));on_update(len(history))
    if {int(v['step']) for v in optimizer.state.values()}!={268}:raise ValueError('非Adam268终态')
    model.eval();return model,optimizer,history,presented
