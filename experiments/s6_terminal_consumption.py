"""只扩展一个已有S6查询的实际完整消费mask，不改变输入/目标或训练剂量。"""
import torch

from experiments.behavior_exploration import paired_counts, first_stop
from experiments.effective_recovery_data import fingerprint
from experiments.s6_recovery_learning import batch_identity
from experiments.retention_sampling import TRAINING_SEEDS


def terminal_schedule(base, union_mask):
    if len(base) != 24 or any([s['role'] for s in batch] != ['anchor','anchor','recovery','recovery'] for batch in base):
        raise ValueError('不是原24批次恢复布局')
    query = base[16][2]
    if any(query.get(k) != v for k,v in dict(source_index=2149,query_step=400,frame_index=40,
                                            origin='learner_query',query_policy='C240-20261020').items()):
        raise ValueError('不是原step400固定恢复槽')
    expected_old = (torch.arange(320) < 11).float()
    expected_full = (torch.arange(320) < 19).float()
    union = torch.as_tensor(union_mask).detach().clone().float()
    if not torch.equal(query['mask'], expected_old) or not torch.equal(union, expected_full):
        raise ValueError('消费范围不是已验证的0–10到0–18')
    if first_stop(query['stop'], .5) != 19 or not torch.equal(query['stop'], (torch.arange(320) >= 18).float()):
        raise ValueError('补点超过原完整教师19点支持')
    updated = [list(batch) for batch in base]
    updated[16][2] = dict(query, mask=union)
    changes=[]
    for j,(old,new) in enumerate(zip(base,updated,strict=True)):
        for k,(a,b) in enumerate(zip(old,new,strict=True)):
            if fingerprint(a,('bev','goal','state','points','stop')) != fingerprint(b,('bev','goal','state','points','stop')):
                raise ValueError('除mask以外的目标或输入被改变')
            if batch_identity([a]) != batch_identity([b]): changes.append([j,k])
    if changes != [[16,2]]:
        raise ValueError('不是唯一预定槽的mask干预')
    return updated, dict(passed=True,changed_slot=[16,2],old_indices=list(range(11)),
                         new_indices=list(range(19)),added_indices=list(range(11,19)),
                         unchanged_slots=95,original_identity=batch_identity([query])[0],
                         updated_identity=batch_identity([updated[16][2]])[0])


def terminal_qualification(original, window, candidates):
    if set(window) != set(TRAINING_SEEDS) or set(candidates) != set(TRAINING_SEEDS):
        return dict(passed=False,reason='incomplete_two_order_comparison',main_protocol_passed=False)
    pairs=[dict(seed=s,**paired_counts(window[s],candidates[s])) for s in TRAINING_SEEDS]
    incremental=(all(p['success_delta'] >= 0 and p['collision_delta'] <= 0 and not p['lost_success_indices'] for p in pairs)
                 and sum(p['success_delta'] for p in pairs) >= 1)
    return dict(passed=original['passed'] and incremental,original_gate=original,
                terminal_vs_window=pairs,terminal_incremental_passed=incremental,
                main_protocol_passed=False,automatic_deployment=False)
