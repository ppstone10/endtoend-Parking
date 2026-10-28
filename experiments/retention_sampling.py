"""保持角色有限分层计划与匹配续训；不拥有文件或闭环执行。"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib

import numpy as np
import torch

from experiments.behavior_exploration import TRAINING_SEEDS, paired_counts, source_balanced_batch, source_balanced_pools
from experiments.behavior_preservation import behavior_loss
from experiments.effective_recovery_data import cell, input_id, scene_number
from scripts.build_selection_dataset import collect_metadata_ids


def select_groups(metadata, forbidden, *, gap_counts=(4, 4)):
    if len(gap_counts) != 2 or any(not isinstance(n, int) or n < 1 for n in gap_counts):
        raise ValueError('补源预算必须为两正整数')
    forbidden = set(forbidden)
    def take(number, task, maneuver, count, purpose):
        pool = [i for i, m in enumerate(metadata) if scene_number(m) == number and m['task_type'] == task
                and (maneuver is None or m['difficulty']['maneuver'] == maneuver)]
        ordered = sorted(pool, key=lambda i: hashlib.sha256(
            (f'retention-v1-{purpose}-S{number}-{task}-{maneuver}:' + metadata[i]['task_id']).encode()).hexdigest())
        selected = []
        for index in ordered:
            identities = collect_metadata_ids(metadata[index])
            if identities & forbidden:
                continue
            selected.append(index); forbidden.update(identities)
            if len(selected) == count:
                return selected
        raise ValueError(f'独立{purpose}来源不足：S{number}/{task}/{maneuver}')
    probes = []
    for number, task, maneuver in [(3, 'T4', 'reverse'), (4, 'T2', 'forward'),
                                    (6, 'T3', 'reverse'), (6, 'T5', 'forward'), (8, 'T5', None)]:
        probes.extend(take(number, task, maneuver, 2 if number == 8 else 1, 'probe'))
    gaps = take(4, 'T2', 'forward', gap_counts[0], 'gap') + take(6, 'T5', 'forward', gap_counts[1], 'gap')
    return {'probes': probes, 'gap_candidates': gaps, 'dense_replay_sources': [1196, 2149]}


def spaced_indices(length, visits):
    if min(length, visits) < 1:
        raise ValueError('时间帧及访问数必须为正')
    if visits == 1:
        return np.asarray([(length - 1) // 2], dtype=int)
    if visits <= length:
        return np.linspace(0, length - 1, visits, dtype=int)
    return np.resize(np.arange(length), visits)


def regression_candidates(metadata, forbidden):
    forbidden = set(forbidden); selected = []
    for number, task, maneuver in [(3, 'T4', 'reverse'), (4, 'T2', 'forward'),
                                    (6, 'T3', 'reverse'), (6, 'T5', 'forward')]:
        pool = [i for i, m in enumerate(metadata) if scene_number(m) == number and m['task_type'] == task
                and m['difficulty']['maneuver'] == maneuver]
        pool.sort(key=lambda i: hashlib.sha256(('retention-v1-regression-audit:' + metadata[i]['task_id']).encode()).hexdigest())
        taken = []
        for index in pool:
            identities = collect_metadata_ids(metadata[index])
            if identities & forbidden:
                continue
            taken.append(index); forbidden.update(identities)
            if len(taken) == 4:
                break
        if len(taken) != 4:
            raise ValueError('退化基线候选不足，不替换任务类型')
        selected.extend(taken)
    return selected


def choose_witnesses(r0_rows, parents, candidates):
    original = {r['dataset_index']: r for r in r0_rows}; witnesses = {}; s8 = {}
    for seed in TRAINING_SEEDS:
        current = {r['dataset_index']: r for r in parents[seed]}
        hits = [i for i in candidates if original[i]['success'] and not current[i]['success']
                and current[i]['scene_name'].split('_')[0] in ('S3', 'S4', 'S6')]
        witnesses[seed] = hits[0] if hits else None
        s8[seed] = [i for i, row in current.items() if row['scene_name'].startswith('S8_') and row['success']]
    selected = list(dict.fromkeys(i for i in witnesses.values() if i is not None))
    return {'passed': all(i is not None for i in witnesses.values()) and all(s8.values()),
            'witness_by_seed': witnesses, 'selected_indices': selected, 's8_success_by_seed': s8,
            'purpose': 'baseline_selected_regression_diagnostic_not_unbiased_generalization'}


def build_schedule(samples, metadata, *, seed, steps, mode, recovery_reference):
    if seed not in TRAINING_SEEDS or steps < 1 or mode not in ('flat', 'hierarchy'):
        raise ValueError('未知采样配置')
    if {s['role'] for s in samples} != {'anchor', 'recovery'}:
        raise ValueError('必须包含保持/恢复两角色')
    pools = source_balanced_pools(samples, seed)
    original = source_balanced_pools(recovery_reference, seed)
    schedule = []
    for step in range(240, 240 + steps):
        anchor = [s for s in source_balanced_batch(pools, step) if s['role'] == 'anchor']
        recovery = [s for s in source_balanced_batch(original, step) if s['role'] == 'recovery']
        if len(anchor) != 2 or len(recovery) != 2:
            raise ValueError('角色批次不为2+2')
        schedule.append(anchor + recovery)
    if mode == 'flat':
        return schedule
    groups = {}; source_frames = {}
    for sample in samples:
        if sample['role'] != 'anchor':
            continue
        index = sample['source_index']; key = cell(metadata[index])
        groups.setdefault(key[0], {}).setdefault(key[1:], set()).add(index)
        source_frames.setdefault(index, []).append(sample)
    generator = np.random.default_rng(seed)
    def order(values):
        values = sorted(values)
        return values if seed == TRAINING_SEEDS[0] else [values[i] for i in generator.permutation(len(values))]
    scenes = order(groups)
    cells = {scene: order(groups[scene]) for scene in scenes}
    sources = {(scene, key): order(groups[scene][key]) for scene in scenes for key in cells[scene]}
    scene_visits = Counter(); cell_visits = Counter(); allocations = defaultdict(list)
    for position in range(steps * 2):
        scene = scenes[position % len(scenes)]
        key = cells[scene][scene_visits[scene] % len(cells[scene])]
        choices = sources[scene, key]; index = choices[cell_visits[scene, key] % len(choices)]
        allocations[index].append(position)
        scene_visits[scene] += 1; cell_visits[scene, key] += 1
    for index, positions in sorted(allocations.items()):
        frames = sorted(source_frames[index], key=lambda s: s['frame_index'])
        chosen = spaced_indices(len(frames), len(positions))
        if seed != TRAINING_SEEDS[0]:
            chosen = chosen[generator.permutation(len(chosen))]
        for position, frame_index in zip(positions, chosen, strict=True):
            schedule[position // 2][position % 2] = frames[int(frame_index)]
    return schedule


def sampling_audit(schedule, metadata):
    flat = [s for batch in schedule for s in batch]
    sources = defaultdict(list)
    for sample in flat:
        sources[sample['role'], sample['source_index']].append(sample)
    return {'role_counts': dict(Counter(s['role'] for s in flat)),
            'anchor_scene_counts': dict(Counter(metadata[s['source_index']]['scene_name'] for s in flat if s['role'] == 'anchor')),
            'anchor_cell_counts': dict(Counter('|'.join(cell(metadata[s['source_index']])) for s in flat if s['role'] == 'anchor')),
            'sources': [{'role': role, 'source_index': index, 'visits': len(values),
                         'unique_inputs': len({input_id(s) for s in values}),
                         'frame_indices': [s['frame_index'] for s in values]} for (role, index), values in sorted(sources.items())],
            'batches': [[{'role': s['role'], 'source_index': s['source_index'], 'frame_index': s['frame_index'],
                         'input_sha256': input_id(s)} for s in batch] for batch in schedule]}


def fit_continuation(parent, optimizer_state, schedule, *, seed, check_budget):
    if len(schedule) != 24 or seed not in TRAINING_SEEDS:
        raise ValueError('短检只允许24更新与固定顺序seed')
    if not optimizer_state['state'] or any(int(s['step']) != 240 for s in optimizer_state['state'].values()):
        raise ValueError('必须继承真实Adam240步起点')
    torch.manual_seed(seed); model = deepcopy(parent).cpu(); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    if any(group['lr'] != 1e-5 for group in optimizer.param_groups):
        raise ValueError('学习率漂移')
    history = []
    for batch in schedule:
        check_budget()
        if Counter(s['role'] for s in batch) != {'anchor': 2, 'recovery': 2}:
            raise ValueError('训练批次角色配比漂移')
        stack = lambda key: torch.stack([s[key] for s in batch])
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_with_stop(stack('bev'), stack('goal'), stack('state'))
        loss = behavior_loss(out.points, out.stop_logits, stack('points'), stack('stop'), stack('mask'), [s['role'] for s in batch])
        if not torch.isfinite(loss):
            raise ValueError('短检loss非有限')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    return model, optimizer, history


def qualification(r0_rows, parents, arms):
    original = {r['dataset_index']: r for r in r0_rows}
    details = {}; qualified = []
    for name, by_seed in arms.items():
        per_seed = []
        for seed in TRAINING_SEEDS:
            if seed not in by_seed or seed not in arms.get('flat_old', {}):
                continue
            rows = by_seed[seed]; parent = parents[seed]
            pair = paired_counts(parent, rows); control = paired_counts(arms['flat_old'][seed], rows)
            old = {r['dataset_index']: r for r in parent}; new = {r['dataset_index']: r for r in rows}
            regressions = [i for i in old if original[i]['success'] and not old[i]['success']
                           and old[i]['scene_name'].split('_')[0] in ('S3', 'S4', 'S6')]
            restored = [i for i in regressions if new[i]['success']]
            s8 = [i for i in old if old[i]['scene_name'].startswith('S8_') and old[i]['success']]
            checks = {'success_gain': pair['success_delta'] >= 1, 'collision_not_increased': pair['collision_delta'] <= 0,
                      'parent_success_retained': not pair['lost_success_indices'], 'old_regression_restored': bool(restored),
                      's8_gain_present_and_retained': bool(s8) and all(new[i]['success'] for i in s8),
                      'matched_control_not_worse': control['success_delta'] >= 0 and control['collision_delta'] <= 0
                                                   and not control['lost_success_indices']}
            per_seed.append({'seed': seed, 'checks': checks, 'parent_pair': pair, 'control_pair': control,
                             'regression_indices': regressions, 'restored_indices': restored, 's8_retained_indices': s8})
        passed = (name != 'flat_old' and len(per_seed) == 2 and all(all(p['checks'].values()) for p in per_seed)
                  and sum(p['control_pair']['success_delta'] for p in per_seed) >= 1)
        if passed:
            qualified.append(name)
        details[name] = {'passed': passed, 'pairs': per_seed}
    return {'qualified': qualified, 'directions': details, 'main_protocol_passed': False, 'automatic_deployment': False}
