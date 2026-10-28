"""在固定保持槽内保障关键新增输入；不修改冻结LT17排程或训练器。"""
from collections import Counter, defaultdict

import numpy as np

from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_sampling import build_schedule, qualification, TRAINING_SEEDS
from experiments.behavior_exploration import paired_counts


KEY_SOURCES = (1196, 2149, 1400, 2256)
PER_SOURCE = 3


def _identity(sample):
    return (sample['role'], sample['source_index'], sample['frame_index'],
            input_id(sample), label_id(sample))


def _novel_groups(old, novel, key_sources):
    if len(key_sources) != 4 or len(set(key_sources)) != 4:
        raise ValueError('必须声明四个不同关键源')
    known = {input_id(s) for s in old}
    observed = set(); groups = defaultdict(list)
    for sample in novel:
        identity = input_id(sample)
        if sample['role'] != 'anchor' or identity in known or identity in observed:
            raise ValueError('新增监督不是唯一新增保持输入')
        if sample['source_index'] not in key_sources:
            raise ValueError('出现未声明新增来源')
        observed.add(identity); groups[sample['source_index']].append(sample)
    if set(groups) != set(key_sources) or any(len(v) < PER_SOURCE for v in groups.values()):
        raise ValueError('关键源新增输入不足，不自动补源或降低配额')
    return {i: sorted(groups[i], key=lambda s: s['frame_index']) for i in key_sources}


def consumption_audit(schedule, baseline, old, novel, *, key_sources=KEY_SOURCES):
    """核验实际有限批次；与缓存数、源内首尾覆盖分开。"""
    groups = _novel_groups(old, novel, key_sources)
    if len(schedule) != 24 or len(baseline) != 24:
        raise ValueError('消费短检固定24步')
    known = {_identity(s) for s in old + novel}
    selected = []; replacements = []
    for step, (batch, original) in enumerate(zip(schedule, baseline, strict=True)):
        if len(batch) != 4 or [s['role'] for s in batch] != ['anchor', 'anchor', 'recovery', 'recovery']:
            raise ValueError('批次必须为保持2＋恢复2')
        if [_identity(s) for s in batch[2:]] != [_identity(s) for s in original[2:]]:
            raise ValueError('恢复角色序列漂移')
        if any(_identity(s) not in known for s in batch):
            raise ValueError('批次含缓存外输入或标签')
        selected.extend(batch[:2])
        for slot in (0, 1):
            if _identity(batch[slot]) != _identity(original[slot]):
                describe = lambda s: dict(role=s['role'], source_index=s['source_index'],
                                          frame_index=s['frame_index'], input_sha256=input_id(s))
                replacements.append(dict(step=step + 241, slot=slot,
                                         before=describe(original[slot]), after=describe(batch[slot])))
    new_ids = {input_id(s) for s in novel}
    used_new = [s for s in selected if input_id(s) in new_ids]
    required = []
    for index, frames in groups.items():
        # Three temporal strata, not outcome-selected or endpoint-only frames.
        positions = ((np.arange(PER_SOURCE) + .5) * len(frames) / PER_SOURCE).astype(int)
        expected = [frames[int(i)] for i in positions]
        used_ids = {input_id(s) for s in selected if s['source_index'] == index}
        if not all(input_id(s) in used_ids for s in expected):
            raise ValueError(f'关键源#{index}时间分层新增帧未全部分配')
        required.append(dict(source_index=index, unique_inputs=PER_SOURCE,
                             frame_indices=[s['frame_index'] for s in expected],
                             input_sha256=[input_id(s) for s in expected]))
    return dict(passed=True, required_sources=required, anchor_slots=48, recovery_slots=48,
                novel_visits=len(used_new), novel_unique_inputs=len({input_id(s) for s in used_new}),
                replacements=replacements)


def build_consumption_schedule(old, novel, metadata, *, seed, key_sources=KEY_SOURCES):
    """优先保留同源位置，只在同场景重新分配不足的保持槽。"""
    groups = _novel_groups(old, novel, key_sources)
    baseline = build_schedule(old + novel, metadata, seed=seed, steps=24,
                              mode='hierarchy', recovery_reference=old)
    schedule = [list(batch) for batch in baseline]
    positions = [(step, slot) for step in range(24) for slot in (0, 1)]
    allocated = {}; reserved = set()
    # Reserve existing source slots for every key before filling deficits.
    for index in key_sources:
        own = [p for p in positions if baseline[p[0]][p[1]]['source_index'] == index]
        allocated[index] = own[:PER_SOURCE]; reserved.update(allocated[index])
    for index in key_sources:
        scene = metadata[index]['scene_name']
        available = [p for p in positions if p not in reserved and
                     metadata[baseline[p[0]][p[1]]['source_index']]['scene_name'] == scene]
        needed = PER_SOURCE - len(allocated[index])
        if len(available) < needed:
            raise ValueError(f'场景{scene}预算不足，不跨场景挤占')
        extra = available[:needed]; allocated[index].extend(extra); reserved.update(extra)
    generator = np.random.default_rng(seed)
    for index in key_sources:
        frames = groups[index]
        chosen = ((np.arange(PER_SOURCE) + .5) * len(frames) / PER_SOURCE).astype(int)
        if seed != TRAINING_SEEDS[0]:
            chosen = chosen[generator.permutation(PER_SOURCE)]
        for (step, slot), j in zip(allocated[index], chosen, strict=True):
            schedule[step][slot] = frames[int(j)]
    for step, slot in positions:
        previous = metadata[baseline[step][slot]['source_index']]['scene_name']
        current = metadata[schedule[step][slot]['source_index']]['scene_name']
        if previous != current:
            raise ValueError('保持场景槽位漂移')
    counts = Counter(metadata[s['source_index']]['scene_name'] for b in schedule for s in b[:2])
    if len(counts) != 6 or set(counts.values()) != {8}:
        raise ValueError('必须保持六场景各8个保持槽')
    audit = consumption_audit(schedule, baseline, old, novel, key_sources=key_sources)
    audit['anchor_scene_counts'] = dict(counts)
    return schedule, audit


def consumption_qualification(r0, parents, flat_control, same_pool_control, candidates):
    """原资格门槛不放宽；再检查消费修正对同池同剂量控制的净增量。"""
    original = qualification(r0, parents, {'flat_old': flat_control, 'consumption': candidates})
    pairs = []
    for seed in TRAINING_SEEDS:
        if seed in candidates:
            pair = paired_counts(same_pool_control[seed], candidates[seed])
            pairs.append(dict(seed=seed, **pair))
    incremental = (len(pairs) == 2 and all(p['success_delta'] >= 0 and p['collision_delta'] <= 0 for p in pairs)
                   and sum(p['success_delta'] for p in pairs) >= 1)
    return dict(passed=bool(original['qualified']) and incremental,
                original_gate=original, same_pool_pairs=pairs, incremental_passed=incremental,
                main_protocol_passed=False, automatic_deployment=False)
