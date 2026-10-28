"""在真实C240成功访问的共同输入上，只比较保持软目标的教师对象。"""
import numpy as np
import torch

from experiments.behavior_exploration import first_stop, paired_counts
from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_consumption import KEY_SOURCES, consumption_qualification
from experiments.retention_sampling import TRAINING_SEEDS


@torch.no_grad()
def select_teacher_pairs(samples, quality_rows, r0_model, check_budget=lambda: None):
    rows = {r['frame_index']: r for r in quality_rows}
    if len(rows) != len(quality_rows) or set(rows) != {s['frame_index'] for s in samples}:
        raise ValueError('访问样本与安全窗口审计不匹配')
    eligible = []; known = set(); unsupported = 0
    r0_model.eval()
    for sample in sorted(samples, key=lambda s: s['frame_index']):
        check_budget(); row = rows[sample['frame_index']]
        if row['input_sha256'] != input_id(sample) or row['label_sha256'] != label_id(sample):
            raise ValueError('C240访问监督身份漂移')
        if not row['clean'] or input_id(sample) in known:
            continue
        out = r0_model.forward_with_stop(sample['bev'][None], sample['goal'][None], sample['state'][None])
        probabilities = out.stop_logits[0].sigmoid().detach().clone()
        mask = torch.nonzero(sample['mask']).flatten()
        if not len(mask) or int(mask[-1]) >= first_stop(probabilities, r0_model.stop_threshold):
            unsupported += 1; continue
        r0 = dict(sample, points=out.points[0].detach().clone(), stop=probabilities, target_teacher='R0')
        c240 = dict(sample, target_teacher='C240')
        eligible.append(dict(r0=r0, c240=c240)); known.add(input_id(sample))
    audit = dict(passed=len(eligible) >= 3, candidates=len(samples), eligible_unique=len(eligible),
                 unsupported_R0=unsupported, selected_frames=[])
    if not audit['passed']:
        return [], audit
    indices = ((np.arange(3) + .5) * len(eligible) / 3).astype(int)
    selected = [eligible[int(i)] for i in indices]
    audit['selected_frames'] = [p['c240']['frame_index'] for p in selected]
    audit['targets'] = [dict(frame_index=p['c240']['frame_index'], input_sha256=input_id(p['c240']),
                             R0_label_sha256=label_id(p['r0']), C240_label_sha256=label_id(p['c240']),
                             R0_length=first_stop(p['r0']['stop'], r0_model.stop_threshold),
                             C240_length=first_stop(p['c240']['stop'], r0_model.stop_threshold)) for p in selected]
    return selected, audit


def build_teacher_schedules(base, pairs_by_source, *, key_sources=KEY_SOURCES):
    if len(base) != 24 or set(pairs_by_source) != set(key_sources):
        raise ValueError('固定24更新与四个完整保持源')
    plans = {'R0_targets': [list(b) for b in base], 'C240_targets': [list(b) for b in base]}
    all_inputs = []; slots = []
    for index in key_sources:
        positions = [(i, j) for i, b in enumerate(base) for j in (0, 1) if b[j]['source_index'] == index]
        pairs = sorted(pairs_by_source[index], key=lambda p: p['c240']['frame_index'])
        if len(positions) != 3 or len(pairs) != 3:
            raise ValueError('每个保持源必须恰好三个槽和三个查询')
        positions.sort(key=lambda p: base[p[0]][p[1]]['frame_index'])
        for (step, slot), pair in zip(positions, pairs, strict=True):
            a, b = pair['r0'], pair['c240']
            if (a['role'] != 'anchor' or b['role'] != 'anchor' or
                    a['source_index'] != index or b['source_index'] != index or
                    a['frame_index'] != b['frame_index'] or input_id(a) != input_id(b) or
                    not torch.equal(a['mask'], b['mask'])):
                raise ValueError('两教师输入/来源/mask不相同')
            plans['R0_targets'][step][slot] = a; plans['C240_targets'][step][slot] = b
            all_inputs.append(input_id(a)); slots.append(dict(step=step+241, slot=slot, source_index=index,
                frame_index=b['frame_index'], input_sha256=input_id(b), R0_label_sha256=label_id(a),
                C240_label_sha256=label_id(b)))
    if len(set(all_inputs)) != 12:
        raise ValueError('共同访问输入不足12个唯一值')
    for original, r, c in zip(base, plans['R0_targets'], plans['C240_targets'], strict=True):
        if [s['role'] for s in original] != ['anchor', 'anchor', 'recovery', 'recovery']:
            raise ValueError('批次角色不符')
        for j in range(4):
            if input_id(r[j]) != input_id(c[j]) or not torch.equal(r[j]['mask'], c[j]['mask']):
                raise ValueError('两个臂实际输入或mask漂移')
            if j >= 2 or original[j]['source_index'] not in key_sources:
                if input_id(original[j]) != input_id(r[j]) or label_id(original[j]) != label_id(r[j]) or label_id(r[j]) != label_id(c[j]):
                    raise ValueError('非目标监督发生变化')
    return plans, dict(passed=True, query_slots=12, common_unique_inputs=12, slots=slots)


def teacher_qualification(r0, parents, flat_control, consumption_control, r0_targets, c240_targets, fit,
                          *, key_sources=KEY_SOURCES):
    original = consumption_qualification(r0, parents, flat_control, consumption_control, c240_targets)
    control_gate = consumption_qualification(r0, parents, flat_control, consumption_control, r0_targets)
    pairs = [dict(seed=s, **paired_counts(r0_targets[s], c240_targets[s])) for s in TRAINING_SEEDS
             if s in r0_targets and s in c240_targets]
    gain = (len(pairs) == 2 and all(p['success_delta'] >= 0 and p['collision_delta'] <= 0 for p in pairs)
            and sum(p['success_delta'] for p in pairs) >= 1)
    retained = all(s in fit and len(fit[s]) == 4 and {r['dataset_index'] for r in fit[s]} == set(key_sources)
                   and all(r['success'] for r in fit[s]) for s in TRAINING_SEEDS)
    return dict(passed=original['passed'] and gain and retained, C240_gate=original, R0_control_gate=control_gate,
                teacher_pairs=pairs, teacher_incremental_passed=gain, source_success_retained=retained,
                main_protocol_passed=False, automatic_deployment=False)
