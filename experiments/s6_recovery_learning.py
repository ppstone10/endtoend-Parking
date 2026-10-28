"""S6真实失败查询的同输入目标干预与闭环裁决；不改原训练器。"""
import torch

from experiments.behavior_exploration import first_stop, paired_counts
from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_consumption import consumption_qualification
from experiments.retention_sampling import TRAINING_SEEDS


ARMS = ('self_targets', 'strict_targets')
QUERY_STEPS = (10, 200, 400)
SLOTS = ((0, 2), (8, 2), (16, 2))


def batch_identity(batch):
    return [dict(role=s['role'], source_index=s['source_index'], frame_index=s['frame_index'],
                 input_sha256=input_id(s), label_sha256=label_id(s)) for s in batch]


@torch.no_grad()
def build_recovery_schedules(base, queries, self_teacher, check_budget=lambda: None):
    if len(base) != 24 or any([s['role'] for s in batch] != ['anchor', 'anchor', 'recovery', 'recovery'] for batch in base):
        raise ValueError('原24批次2保持＋2恢复布局不符')
    if len(queries) != 3 or {q['query_step'] for q in queries} != set(QUERY_STEPS):
        raise ValueError('不是固定三个实际失败查询')
    if len({input_id(q) for q in queries}) != 3:
        raise ValueError('三个查询输入重复')
    plans = {arm: [list(batch) for batch in base] for arm in ARMS}; targets = []
    if hasattr(self_teacher, 'eval'):
        self_teacher.eval()
    for (step, slot), query in zip(SLOTS, sorted(queries, key=lambda s: s['query_step']), strict=True):
        check_budget()
        if (query['role'] != 'recovery' or query['source_index'] != 2149 or
                query['origin'] != 'learner_query' or query['query_policy'] != 'C240-20261020' or
                query['frame_index'] != query['query_step']//10):
            raise ValueError('S6查询真实来源/角色/相位不符')
        points, stop, mask = query['points'], query['stop'], query['mask']
        if (points.shape != (320, 3) or stop.shape != (320,) or mask.shape != (320,) or
                not all(torch.isfinite(query[k]).all() for k in ('bev', 'goal', 'state', 'points', 'stop', 'mask')) or
                not torch.isin(mask, torch.tensor([0., 1.])).all() or not mask.any()):
            raise ValueError('查询监督尺寸/有限性/消费mask非法')
        length = first_stop(stop, self_teacher.stop_threshold)
        if not torch.equal(stop, (torch.arange(320) >= length-1).float()):
            raise ValueError('严格教师不是原实际参考的累积硬停止目标')
        out = self_teacher.forward_with_stop(query['bev'][None], query['goal'][None], query['state'][None])
        control = dict(query, points=out.points[0].detach().clone(), stop=out.stop_logits[0].sigmoid().detach().clone())
        used = torch.nonzero(mask).flatten()
        control_length = first_stop(control['stop'], self_teacher.stop_threshold)
        if int(used.max()) >= min(length, control_length):
            raise ValueError('共同消费mask超出教师/自目标首停支持，不改mask补救')
        if control['points'].shape != (320, 3) or not torch.isfinite(control['points']).all():
            raise ValueError('自目标轨迹非法')
        if label_id(control) == label_id(query):
            raise ValueError('本查询没有目标干预变量')
        plans['self_targets'][step][slot] = control
        plans['strict_targets'][step][slot] = query
        targets.append(dict(batch_index=step, slot=slot, query_step=query['query_step'],
            input_sha256=input_id(query), teacher_length=length, control_length=control_length,
            self_label_sha256=label_id(control), strict_label_sha256=label_id(query),
            replaced=batch_identity([base[step][slot]])[0]))
    for j, original in enumerate(base):
        for k, sample in enumerate(original):
            left, right = plans['self_targets'][j][k], plans['strict_targets'][j][k]
            if input_id(left) != input_id(right) or not torch.equal(left['mask'], right['mask']):
                raise ValueError('共同输入/消费mask漂移')
            if (j, k) not in SLOTS and (batch_identity([sample]) != batch_identity([left]) or
                                      batch_identity([sample]) != batch_identity([right])):
                raise ValueError('非目标保持/恢复样本被改写')
    return plans, dict(passed=True, changed_slots=[list(p) for p in SLOTS], query_steps=list(QUERY_STEPS),
                       untouched_anchor_slots=48, untouched_recovery_slots=45, targets=targets)


def recovery_qualification(r0, parents, flat, consumption, controls, candidates, parent_fit, candidate_fit):
    if any(set(values) != set(TRAINING_SEEDS) for values in (parents, flat, consumption, controls, candidates,
                                                            parent_fit, candidate_fit)):
        return dict(passed=False, reason='incomplete_seed_combinations', main_protocol_passed=False)
    original = consumption_qualification(r0, parents, flat, consumption, candidates)
    pairs = [dict(seed=s, **paired_counts(controls[s], candidates[s])) for s in TRAINING_SEEDS]
    incremental = (all(p['success_delta'] >= 0 and p['collision_delta'] <= 0 and not p['lost_success_indices']
                       for p in pairs) and sum(p['success_delta'] for p in pairs) >= 1)
    fit = []
    for seed in TRAINING_SEEDS:
        pair = paired_counts(parent_fit[seed], candidate_fit[seed])
        rows = {r['dataset_index']: r for r in candidate_fit[seed]}
        strict = 2149 in rows and rows[2149]['reached_with_margin']
        fit.append(dict(seed=seed, S6_2149_strict_recovered=bool(strict),
                        parent_fit_successes_preserved=not pair['lost_success_indices'], pair=pair))
    fit_passed = all(r['S6_2149_strict_recovered'] and r['parent_fit_successes_preserved'] for r in fit)
    return dict(passed=original['passed'] and incremental and fit_passed, original_gate=original,
                same_input_pairs=pairs, incremental_passed=incremental, fit_gate=fit, fit_passed=fit_passed,
                main_protocol_passed=False, automatic_deployment=False)
