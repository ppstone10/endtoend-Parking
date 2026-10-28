"""在原24批次中，仅对已严格执行的S6终段复用停止边界/终点监督。"""
import torch
from torch.nn import functional as F

from experiments.behavior_exploration import paired_counts
from experiments.behavior_preservation import behavior_loss
from experiments.retention_sampling import TRAINING_SEEDS
from model.network import terminal_pose_loss, variable_loss_fn


ARMS = ('boundary_only', 'boundary_terminal')


def terminal_query(sample):
    identity = dict(role='recovery', source_index=2149, origin='learner_query',
                    query_policy='C240-20261020', query_step=400, frame_index=40)
    if not all(sample.get(key) == value for key, value in identity.items()):
        return False
    if (not torch.equal(sample['mask'], (torch.arange(320) < 19).float()) or
            not torch.equal(sample['stop'], (torch.arange(320) >= 18).float())):
        raise ValueError('固定终段查询缺完整连续19点/原硬停止目标')
    return True


def objective_slots(schedule):
    slots = [[i,j] for i,batch in enumerate(schedule) for j,s in enumerate(batch) if terminal_query(s)]
    if len(schedule) != 24 or slots != [[16,2]]:
        raise ValueError('只能干预原24批次的唯一step400恢复槽')
    return slots


def objective_loss(points, logits, samples, arm):
    if arm not in ARMS:
        raise ValueError('未知终段监督实验臂')
    stack = lambda key: torch.stack([s[key] for s in samples])
    target, stop, mask = stack('points'), stack('stop'), stack('mask')
    roles = [s['role'] for s in samples]
    original = behavior_loss(points, logits, target, stop, mask, roles)
    selected = [i for i,s in enumerate(samples) if terminal_query(s)]
    if not selected:
        return original
    if len(selected) != 1:
        raise ValueError('同批次终段查询重复')
    i = selected[0]
    # 保留原角色均值/角色间等权及yaw=10的轨迹项；只替换此样本停止项。
    old_stop = .2 * F.binary_cross_entropy_with_logits(logits[i], stop[i])
    new_stop = variable_loss_fn(target[i:i+1], logits[i:i+1], target[i:i+1], mask[i:i+1],
        stop_weight=.2, balance_stop=True, stop_target_mode='terminal', yaw_error_mode='wrapped')
    correction = new_stop - old_stop
    if arm == 'boundary_terminal':
        correction = correction + terminal_pose_loss(points[i:i+1], target[i:i+1], mask[i:i+1],
                                                     yaw_weight=10, yaw_error_mode='wrapped')
    return original + correction / (len(set(roles)) * roles.count('recovery'))


def objective_qualification(original, baseline, candidates):
    if any(set(values) != set(TRAINING_SEEDS) for values in (baseline,candidates)):
        return dict(passed=False, reason='incomplete_two_order_comparison', main_protocol_passed=False)
    pairs = [dict(seed=s, **paired_counts(baseline[s], candidates[s])) for s in TRAINING_SEEDS]
    incremental = (all(p['success_delta'] >= 0 and p['collision_delta'] <= 0 and
                       not p['lost_success_indices'] for p in pairs) and
                   sum(p['success_delta'] for p in pairs) >= 1)
    return dict(passed=bool(original['passed'] and incremental), original_gate=original,
                objective_vs_terminal_mask=pairs, objective_incremental_passed=incremental,
                main_protocol_passed=False, automatic_deployment=False)
