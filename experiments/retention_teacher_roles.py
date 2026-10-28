"""只保持已验证成功的C240来源，其失败来源保留可信R0恢复目标。"""
import numpy as np

from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_teacher import build_teacher_schedules


def source_role(c240_row, r0_row):
    if c240_row['success']:
        if not c240_row['query_gate']['passed']:
            raise ValueError('C240成功但查询不够，不能伪装成R0恢复分支')
        return 'C240_preserve'
    if not r0_row['success']:
        raise ValueError('两个来源策略均失败，不能制造正教师')
    return 'R0_restore'


def fallback_pairs(samples, quality):
    rows = {r['frame_index']: r for r in quality}
    if len(rows) != len(quality) or set(rows) != {s['frame_index'] for s in samples}:
        raise ValueError('R0恢复窗口审计不完整')
    eligible = []; seen = set()
    for sample in sorted(samples, key=lambda s: s['frame_index']):
        row = rows[sample['frame_index']]
        if input_id(sample) != row['input_sha256'] or label_id(sample) != row['label_sha256']:
            raise ValueError('R0实际访问与监督身份漂移')
        if row['clean'] and input_id(sample) not in seen:
            eligible.append(sample); seen.add(input_id(sample))
    if len(eligible) < 3:
        raise ValueError('R0恢复窗口不足，不再自动改来源')
    positions = ((np.arange(3) + .5) * len(eligible) / 3).astype(int)
    return [dict(r0=dict(eligible[int(i)], target_teacher='R0_restore'),
                 c240=dict(eligible[int(i)], target_teacher='R0_restore')) for i in positions]


def build_role_schedules(base, pairs, roles, *, key_sources=(1196, 2149, 1400, 2256)):
    if set(roles) != set(key_sources) or not set(roles.values()) <= {'C240_preserve', 'R0_restore'}:
        raise ValueError('教师来源分工不完整')
    if not any(v == 'C240_preserve' for v in roles.values()):
        raise ValueError('没有可保持C240来源，本次没有教师对照变量')
    for index, role in roles.items():
        if role == 'R0_restore' and any(label_id(p['r0']) != label_id(p['c240']) for p in pairs[index]):
            raise ValueError('R0恢复来源被错误改成其他目标')
    plans, audit = build_teacher_schedules(base, pairs, key_sources=key_sources)
    plans['verified_targets'] = plans.pop('C240_targets')
    for slot in audit['slots']:
        slot['verified_label_sha256'] = slot.pop('C240_label_sha256')
        slot['source_role'] = roles[slot['source_index']]
    audit['source_roles'] = roles
    audit['changed_target_slots'] = sum(s['R0_label_sha256'] != s['verified_label_sha256'] for s in audit['slots'])
    return plans, audit
