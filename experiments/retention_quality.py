"""区分保持参考/执行窗口质量，按同源同剂量替换而不改标签值。"""
import numpy as np

from experiments.behavior_exploration import paired_counts
from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_consumption import consumption_qualification
from experiments.retention_sampling import TRAINING_SEEDS
from interfaces import VehicleState
from runtime.sources import NetworkSource


def _identity(sample):
    return (sample['role'], sample['source_index'], sample['frame_index'], input_id(sample), label_id(sample))


def audit_source_windows(samples, states, case, check_budget=lambda: None):
    """参考为原始消费点链，执行为真实R0转移；两者分别记录。"""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 5 or not len(states) or not np.isfinite(states).all():
        raise ValueError('R0状态记录非法')
    previous = np.vstack((case['start'].to_array(), states[:-1]))
    planning = case['planner']._collision_checker
    contact = case['engine_kwargs']['collision_checker']
    source = NetworkSource(None, None); source.begin(case['start'], case['goal'])
    rows = []
    for sample in sorted(samples, key=lambda s: s['frame_index']):
        check_budget(); frame = sample['frame_index']; start = frame * 10
        if not isinstance(frame, int) or frame < 0 or start >= len(states) or sample['role'] != 'anchor':
            raise ValueError('来源帧号/角色无法对齐')
        state = VehicleState.from_array(previous[start])
        if (not np.array_equal(np.asarray(sample['state']), np.asarray(previous[start, 3:], dtype=np.float32)) or
                not np.array_equal(np.asarray(sample['goal']), source._to_local_goal(state).astype(np.float32))):
            raise ValueError('来源帧与实际R0状态/目标输入不一致')
        points = np.asarray(sample['points']); mask = np.asarray(sample['mask'])
        if (points.ndim != 2 or points.shape[1] != 3 or mask.shape != (len(points),) or
                not np.isfinite(points).all() or not np.isin(mask, (0, 1)).all() or not mask.any()):
            raise ValueError('消费mask/标签非法')
        used = np.flatnonzero(mask)
        global_points = NetworkSource._to_global(points, state)
        chain = np.vstack((previous[start, :3], global_points[used]))
        end = min(start + 10, len(states))
        execution = [(previous[k, :3], states[k, :3]) for k in range(start, end)]
        reference = list(zip(chain[:-1], chain[1:]))
        count = lambda checker, segments: sum(not bool(checker.swept_segment_free(a, b)) for a, b in segments)
        current = bool(planning.pose_free(state.x, state.y, state.yaw))
        execution_risk = count(planning, execution); reference_risk = count(planning, reference)
        rows.append(dict(source_index=sample['source_index'], frame_index=frame, start_step=start,
                         input_sha256=input_id(sample), label_sha256=label_id(sample), current_safe=current,
                         execution_margin_violations=execution_risk, reference_margin_violations=reference_risk,
                         execution_contact_violations=count(contact, execution),
                         reference_contact_violations=count(contact, reference),
                         execution_steps=end-start, reference_segments=len(reference),
                         clean=current and execution_risk == 0 and reference_risk == 0))
    return rows


def build_quality_schedules(base, novel, quality_rows, *, source_index=1196, target_frames=(11, 18)):
    """仅按冻结源几何/时间距离选两对替换，完全不读候选闭环结果。"""
    if len(base) != 24 or len(target_frames) != 2 or len(set(target_frames)) != 2:
        raise ValueError('固定24步与两个替换目标')
    samples = [s for s in novel if s['source_index'] == source_index]
    by_frame = {s['frame_index']: s for s in samples}
    rows = {r['frame_index']: r for r in quality_rows}
    if len(by_frame) != len(samples) or len(rows) != len(quality_rows) or set(rows) != set(by_frame):
        raise ValueError('来源窗口审计不完整或帧号重复')
    for frame, sample in by_frame.items():
        row = rows[frame]
        if (row['source_index'] != source_index or row['input_sha256'] != input_id(sample) or
                row['label_sha256'] != label_id(sample)):
            raise ValueError('质量判定与监督身份不符')
    positions = {}
    used_ids = {input_id(s) for b in base for s in b}
    for step, batch in enumerate(base):
        if len(batch) != 4 or [s['role'] for s in batch] != ['anchor', 'anchor', 'recovery', 'recovery']:
            raise ValueError('原批次角色不符')
        for slot, sample in enumerate(batch[:2]):
            if sample['source_index'] == source_index and sample['frame_index'] in target_frames:
                frame = sample['frame_index']
                if frame in positions or rows[frame]['clean'] or not rows[frame]['current_safe']:
                    raise ValueError('目标不是两个不同的起点安全/窗口风险样本')
                if _identity(sample) != _identity(by_frame[frame]):
                    raise ValueError('目标样本与审计缓存不符')
                positions[frame] = (step, slot)
    if set(positions) != set(target_frames):
        raise ValueError('原计划缺固定风险槽位')
    plans = {}; chosen_frames = {}; audits = {}
    novel_ids = {input_id(s) for s in novel}
    for arm, clean in [('clean_windows', True), ('risky_windows', False)]:
        available = [f for f, r in rows.items() if r['current_safe'] and r['clean'] == clean
                     and input_id(by_frame[f]) not in used_ids]
        chosen = []
        for target in target_frames:
            if not available:
                raise ValueError(f'{arm}缺少同源同数替换，不换源/降低数量')
            frame = min(available, key=lambda f: (abs(f-target), f))
            chosen.append(frame); available.remove(frame)
        plan = [list(batch) for batch in base]
        for target, replacement in zip(target_frames, chosen, strict=True):
            step, slot = positions[target]; plan[step][slot] = by_frame[replacement]
        changed = []
        for step, (original, updated) in enumerate(zip(base, plan, strict=True)):
            for slot in range(4):
                if _identity(original[slot]) != _identity(updated[slot]):
                    if (step, slot) not in positions.values() or original[slot]['source_index'] != updated[slot]['source_index']:
                        raise ValueError('改变了非目标槽位或来源')
                    changed.append(dict(step=step+241, slot=slot, before_frame=original[slot]['frame_index'],
                                        after_frame=updated[slot]['frame_index'], input_sha256=input_id(updated[slot])))
        new_ids = {input_id(s) for b in plan for s in b[:2] if input_id(s) in novel_ids}
        if len(changed) != 2 or len(new_ids) != 12:
            raise ValueError('替换剂量/新增输入数不符')
        plans[arm] = plan; chosen_frames[arm] = chosen
        audits[arm] = dict(passed=True, changed_slots=changed, novel_unique_inputs=len(new_ids))
    return plans, dict(passed=True, source_index=source_index, target_frames=list(target_frames),
                       replacement_frames=chosen_frames, arms=audits)


def quality_qualification(r0, parents, flat_control, consumption_control, clean, risky):
    original = consumption_qualification(r0, parents, flat_control, consumption_control, clean)
    risky_gate = consumption_qualification(r0, parents, flat_control, consumption_control, risky)
    pairs = [dict(seed=seed, **paired_counts(risky[seed], clean[seed]))
             for seed in TRAINING_SEEDS if seed in clean and seed in risky]
    discriminating = (len(pairs) == 2 and all(p['success_delta'] >= 0 and p['collision_delta'] <= 0 for p in pairs)
                      and sum(p['success_delta'] for p in pairs) >= 1)
    return dict(passed=original['passed'] and discriminating, clean_gate=original, risky_gate=risky_gate,
                clean_vs_risky=pairs, quality_discrimination_passed=discriminating,
                main_protocol_passed=False, automatic_deployment=False)
