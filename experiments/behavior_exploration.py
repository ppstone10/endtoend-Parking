"""闭环探索的学习对照与配对裁决；不负责运行时或文件持久化。"""
from copy import deepcopy

import numpy as np
import torch

from experiments.behavior_preservation import behavior_loss


TRAINING_SEEDS=(20261020,20261021)
PRIMARY_ARMS=('sparse_joint','dense_joint','dense_parent_stop','dense_recovery_only')


def first_stop(probabilities,threshold):
    hits=torch.nonzero(probabilities>=threshold).flatten()
    return int(hits[0])+1 if len(hits) else len(probabilities)


def arm_samples(sparse,dense,arm):
    if arm not in PRIMARY_ARMS:raise ValueError('未知主对照')
    samples=sparse if arm=='sparse_joint' else dense
    if arm=='dense_recovery_only':samples=[s for s in samples if s['role']=='recovery']
    if arm=='dense_parent_stop':samples=[dict(s,stop=s['parent_stop']) for s in samples]
    if not samples:raise ValueError('对照监督为空')
    return samples


def source_balanced_pools(samples,seed):
    groups={}
    for sample in samples:
        groups.setdefault(sample['role'],{}).setdefault(sample.get('source_index',0),[]).append(sample)
    generator=np.random.default_rng(seed);result={}
    for role in sorted(groups):
        pools=list(groups[role].values())
        if seed!=TRAINING_SEEDS[0]:
            pools=[[pool[i] for i in generator.permutation(len(pool))] for pool in pools]
            pools=[pools[i] for i in generator.permutation(len(pools))]
        result[role]=pools
    return result


def source_balanced_batch(pools,step):
    # 每源四帧再切换；稀疏每源恰四帧时复现旧顺序，dense不因轨迹更长获得更大权重。
    selected=[]
    for sources in pools.values():
        for j in range(2):
            position=step*2+j;block=position//4
            source=sources[block%len(sources)];cycle=block//len(sources)
            selected.append(source[(cycle*4+position%4)%len(source)])
    return selected


@torch.no_grad()
def fixed_frame_metrics(model,samples,check_budget=lambda:None):
    """所有对照在同一完整缓存评价，不比较轮换训练批次的首末loss。"""
    model.eval();groups={}
    for sample in samples:
        check_budget()
        out=model.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
        points=out.points[0];mask=sample['mask'];denom=mask.sum().clamp_min(1)
        delta=points[:,2]-sample['points'][:,2]
        xy=float(((points[:,:2]-sample['points'][:,:2]).square().sum(-1)*mask).sum()/denom)
        yaw=float((torch.atan2(torch.sin(delta),torch.cos(delta)).square()*mask).sum()/denom)
        length=first_stop(out.stop_logits[0].sigmoid(),model.stop_threshold)
        role=sample['role'];group=groups.setdefault(role,[])
        group.append((xy,yaw,abs(length-sample['parent_length']),length!=sample['parent_length']))
    return {role:dict(zip(('xy_mse','yaw_mse','parent_length_abs_shift','parent_length_changed_fraction'),
                          np.asarray(values,dtype=float).mean(axis=0).tolist()),frames=len(values))
            for role,values in groups.items()}


def fit_exploration_arm(parent,samples,*,seed,steps,snapshots,check_budget,checkpoint,
                        optimizer_state=None,start_step=0):
    """固定学习率/角色批量；checkpoint负责闭环评价及持久化。"""
    if seed not in TRAINING_SEEDS or steps not in (2,240) or not set(snapshots)<={2,12,60,240}:
        raise ValueError('探索只允许预声明更新预算')
    if not samples or not snapshots or max(snapshots)!=steps:raise ValueError('缺少最终检查点')
    torch.manual_seed(seed);model=deepcopy(parent).cpu();model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    if optimizer_state is not None:optimizer.load_state_dict(deepcopy(optimizer_state))
    pools=source_balanced_pools(samples,seed)
    history=[]
    for step in range(steps):
        check_budget();selected=source_balanced_batch(pools,start_step+step)
        stack=lambda k:torch.stack([s[k] for s in selected])
        model.train();optimizer.zero_grad(set_to_none=True)
        out=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
        loss=behavior_loss(out.points,out.stop_logits,stack('points'),stack('stop'),stack('mask'),[s['role'] for s in selected])
        if not torch.isfinite(loss):raise ValueError('探索loss非有限')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        history.append(float(loss.detach()))
        if step+1 in snapshots:
            checkpoint(start_step+step+1,model,optimizer,history.copy())
    return model


def paired_counts(baseline,candidate):
    old={r['dataset_index']:r for r in baseline};new={r['dataset_index']:r for r in candidate}
    if not old or old.keys()!=new.keys() or len(old)!=len(baseline) or len(new)!=len(candidate):
        raise ValueError('闭环配对分母不一致/重复')
    if any((old[i]['task_id'],old[i]['scene_name'])!=(new[i]['task_id'],new[i]['scene_name']) for i in old):
        raise ValueError('闭环配对身份不一致')
    scene_delta={scene:sum(int(new[i]['success'])-int(old[i]['success']) for i in old if old[i]['scene_name']==scene)
                 for scene in sorted({r['scene_name'] for r in baseline})}
    ds=sum(int(new[i]['success'])-int(old[i]['success']) for i in old)
    dc=sum(int(new[i]['collision'])-int(old[i]['collision']) for i in old)
    lost=[i for i in old if old[i]['success'] and not new[i]['success']]
    return {'count':len(old),'success_delta':ds,'collision_delta':dc,'lost_success_indices':lost,
            'scene_success_delta':scene_delta,'successes':sum(r['success'] for r in candidate),
            'collisions':sum(r['collision'] for r in candidate),
            'discovery_passed':ds>=1 and dc<=-1 and not lost,
            'confirmation_passed':len(old)==60 and ds>=3 and dc<=-3 and len(lost)<=2 and min(scene_delta.values())>=-1}


def rank_directions(finals,baseline):
    """只看发现集最终点；无过线也继续复核最前两个，但不追认其准入。"""
    ranked=[]
    for name,by_seed in finals.items():
        if set(by_seed)!=set(TRAINING_SEEDS):continue
        pairs=[paired_counts(baseline,by_seed[s]['probe']) for s in TRAINING_SEEDS]
        eligible=all(p['discovery_passed'] for p in pairs)
        score=(int(eligible),sum(p['success_delta']-p['collision_delta'] for p in pairs),
               -sum(len(p['lost_success_indices']) for p in pairs),-sum(p['collisions'] for p in pairs))
        ranked.append({'direction':name,'discovery_eligible':eligible,'pairs':pairs,'score':score})
    return sorted(ranked,key=lambda r:(*[-v for v in r['score']],r['direction']))[:2]


def contrast_report(finals,baseline_fit,baseline_probe):
    comparisons=(('coverage','sparse_joint','dense_joint'),('stop_targets','dense_joint','dense_parent_stop'),
                 ('success_preservation','dense_joint','dense_recovery_only'),
                 ('learner_state_aggregation','continued_dense','aggregated_dense'))
    result=[]
    for hypothesis,left,right in comparisons:
        per_seed=[]
        for seed in TRAINING_SEEDS:
            if seed not in finals.get(left,{}) or seed not in finals.get(right,{}):continue
            a,b=finals[left][seed],finals[right][seed]
            per_seed.append({'training_seed':seed,'fit':paired_counts(a['fit'],b['fit']),
                             'probe':paired_counts(a['probe'],b['probe']),
                             'left_fixed_metrics':a['fixed_metrics'],'right_fixed_metrics':b['fixed_metrics']})
        result.append({'hypothesis':hypothesis,'left':left,'right':right,'pairs':per_seed,
                       'interpretation':'配对干预读数，仅支持或反对本次配置假设；不自动宣称统一根因'})
    return result
