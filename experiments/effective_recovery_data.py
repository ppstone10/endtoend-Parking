"""有效输入增量、预固定来源与离线严格恢复查询；不改生产执行。"""
import hashlib

import numpy as np
import torch

from experiments.behavior_exploration import first_stop
from experiments.behavior_exploration_data import cache_samples, run_case, save_record, strict_teacher
from experiments.executable_behavior import prepare_supervision_frames
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.preflight_behavior_repair import FIT, PROBE, result_row


def fingerprint(sample, fields):
    digest = hashlib.sha256()
    for name in fields:
        value = sample[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype='<f4').copy(order='C')
        if not np.isfinite(array).all():
            raise ValueError('非有限监督不能参与身份计算')
        array[array == 0] = 0  # 将负零规范化，避免无信息的字节差异。
        digest.update(name.encode() + str(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


def input_id(sample):
    return fingerprint(sample, ('bev', 'goal', 'state'))


def label_id(sample):
    return fingerprint(sample, ('points', 'stop', 'mask'))


def novel_samples(old, proposed):
    known = {}
    observed = {}
    for sample in old:
        known.setdefault(input_id(sample), set()).add(label_id(sample))
        observed.setdefault(fingerprint(sample, ('bev', 'state')), []).append(sample)
    initial = {key: set(value) for key, value in known.items()}
    accepted = []; decisions = []; batch_ids = set(); pairs = set()
    for sample in proposed:
        identity, label = input_id(sample), label_id(sample)
        batch_ids.add(identity); pairs.add((identity, label))
        if identity in known:
            decision = 'duplicate' if label in known[identity] else 'same_input_conflicting_label'
        elif any(np.allclose(np.asarray(sample['goal']), np.asarray(prior['goal']), rtol=0, atol=2e-5)
                 for prior in observed.get(fingerprint(sample, ('bev', 'state')), [])):
            # 旧教师goal与生产NetworkSource存在ULP级算式差异，不能当新访问状态。
            decision = 'input_rounding_alias'
        else:
            decision = 'accepted_new_input'; accepted.append(sample)
        known.setdefault(identity, set()).add(label)
        observed.setdefault(fingerprint(sample, ('bev', 'state')), []).append(sample)
        decisions.append({'input_sha256': identity, 'label_sha256': label, 'decision': decision,
                          'source_index': sample['source_index'], 'frame_index': sample['frame_index']})
    return accepted, {'appended': len(proposed), 'batch_unique_inputs': len(batch_ids),
                      'novel_inputs_vs_old': len(batch_ids - initial.keys()),
                      'novel_pairs_vs_old': sum(label not in initial.get(key, ()) for key, label in pairs),
                      'accepted_new_inputs': len(accepted),
                      'duplicates': sum(r['decision'] == 'duplicate' for r in decisions),
                      'rounding_aliases': sum(r['decision'] == 'input_rounding_alias' for r in decisions),
                      'conflicts': sum(r['decision'] == 'same_input_conflicting_label' for r in decisions),
                      'decisions': decisions}


def scene_number(metadata):
    return int(metadata['scene_name'].split('_')[0][1:])


def cell(metadata):
    difficulty = metadata['difficulty']
    return (metadata['scene_name'], metadata['task_type'], difficulty['maneuver'], difficulty['distance_tier'])


def select_sources(metadata, old_confirmation, reserved_ids):
    """只依元数据分组，不读成功/碰撞结果；失败来源不替换。"""
    forbidden = set(reserved_ids) | collect_metadata_ids([metadata[i] for i in FIT + PROBE + old_confirmation])
    def take(pool, quota, salt):
        selected = []
        for index in sorted(pool, key=lambda i: hashlib.sha256((salt + metadata[i]['task_id']).encode()).hexdigest()):
            identities = collect_metadata_ids(metadata[index])
            if identities & forbidden:
                continue
            selected.append(index); forbidden.update(identities)
            if len(selected) == quota:
                break
        if len(selected) != quota:
            raise ValueError(f'预固定来源不足：{salt} {len(selected)}/{quota}')
        return selected
    confirmation = []; recovery = []
    for number in (1, 2, 3, 4, 5, 6, 8):
        confirmation.extend(take([i for i, m in enumerate(metadata) if scene_number(m) == number],
                                 15 if number in (4, 8) else 6, f'effective-v1-confirm-S{number}:'))
    for number in (4, 8):
        recovery.extend(take([i for i, m in enumerate(metadata) if scene_number(m) == number],
                             6, f'effective-v1-recovery-S{number}:'))
    groups = {}
    for i, m in enumerate(metadata):
        if not collect_metadata_ids(m) & forbidden:
            groups.setdefault(cell(m), []).append(i)
    replay = []
    for key, pool in sorted(groups.items()):
        replay.extend(take(pool, min(2, len(pool)), 'effective-v1-replay:' + '|'.join(key)))
    return {'fresh_confirmation': confirmation, 'recovery_sources': recovery, 'replay_sources': replay,
            'replay_cell_count': len(groups)}


def uniform_contexts(contexts, limit=3):
    safe = sorted((c for c in contexts if c.step > 0 and c.step % 10 == 0), key=lambda c: c.step)
    return [safe[i] for i in np.unique(np.linspace(0, len(safe) - 1, min(limit, len(safe)), dtype=int))] if safe else []


@torch.no_grad()
def query_sample(visited, teacher_frames, index, step, policy, parent, budget):
    """标签来自教师首帧，输入严格来自网络；教师续行帧不入训练。"""
    teacher = teacher_frames[0]
    for key in ('pose', 'bev', 'state'):
        if not np.array_equal(visited[key], teacher[key]):
            raise ValueError(f'查询起点{key}未复现网络实际输入')
    if not np.allclose(visited['goal'], teacher['goal'], rtol=0, atol=2e-5):
        raise ValueError('查询起点goal几何不一致')
    sample = prepare_supervision_frames([teacher], None, 'recovery', index)[0]
    for key in ('bev', 'goal', 'state'):
        sample[key] = torch.from_numpy(np.array(visited[key], copy=True)).float()
    budget.check()
    prediction = parent.forward_with_stop(sample['bev'][None], sample['goal'][None], sample['state'][None])
    sample.update(parent_stop=prediction.stop_logits[0].sigmoid().detach().clone(),
                  frame_index=step // 10, query_step=step, query_policy=policy,
                  origin='learner_query', global_pose=list(visited['pose']))
    sample['parent_length'] = first_stop(sample['parent_stop'], parent.stop_threshold)
    return sample


def collect_replay(case, parent, budget, directory):
    result, engine, frames = run_case(case, parent, budget)
    save_record(directory, 'R0', result)
    samples, _ = cache_samples(frames, parent, 'anchor', case['index'], parent, budget) if result.success else ([], [])
    return {'samples': samples, 'row': result_row(case, result, engine)}


def collect_queries(case, data, learner, parent, policy, budget, directory, *, steps=None):
    result, engine, frames = run_case(case, learner, budget)
    save_record(directory, 'learner', result)
    contexts = uniform_contexts(engine.contexts) if steps is None else [c for c in engine.contexts if c.step in steps]
    samples = []; queries = []
    for context in contexts:
        budget.check()
        if not case['planner']._collision_checker.pose_free(context.state.x, context.state.y, context.state.yaw):
            raise ValueError('查询上下文并非规划安全')
        try:
            teacher, attempts = strict_teacher(case, context, data, budget, directory, f'query-{context.step}')
        except ValueError as exc:
            queries.append({'step': context.step, 'decision': 'teacher_execution_error', 'error': str(exc)})
            continue
        row = {'step': context.step, 'teacher_attempts': attempts, 'teacher_continuation_frames': max(0, len(teacher) - 1)}
        if teacher:
            sample = query_sample(frames[context.step // 10], teacher, case['index'], context.step, policy, parent, budget)
            samples.append(sample); row.update(decision='strict_query_proposed', input_sha256=input_id(sample))
        else:
            row['decision'] = 'strict_teacher_rejected'
        queries.append(row)
    return {'samples': samples, 'row': result_row(case, result, engine), 'queries': queries, 'policy': policy}


def data_readiness(replay, queries, metadata):
    indices = sorted({s['source_index'] for s in replay})
    cells = [cell(metadata[i]) for i in indices]
    by_scene = {f'S{n}': sorted({s['source_index'] for s in queries if scene_number(metadata[s['source_index']]) == n}) for n in (4, 8)}
    checks = {'replay_24_tasks': len(indices) >= 24,
              'replay_six_scenes': set(range(1, 7)) <= {scene_number(metadata[i]) for i in indices},
              'replay_four_task_types': len({c[1] for c in cells}) >= 4,
              'replay_both_maneuvers': {'forward', 'reverse'} <= {c[2] for c in cells},
              'replay_three_distances': {'near', 'medium', 'long'} <= {c[3] for c in cells},
              'novel_S4_two_tasks': len(by_scene['S4']) >= 2,
              'novel_S8_two_tasks': len(by_scene['S8']) >= 2,
              'query_origin_valid': bool(queries) and all(s.get('origin') == 'learner_query' and
                  s.get('query_step', 0) > 0 and s['query_step'] % 10 == 0 for s in queries)}
    return {'passed': all(checks.values()), 'checks': checks, 'replay_tasks': indices,
            'replay_cells': sorted(set(cells)), 'novel_query_sources': by_scene, 'novel_query_count': len(queries)}


ARMS = ('old_old', 'broad_old', 'old_novel', 'broad_novel')


def matched_pools(old, replay, novel):
    anchors = [s for s in old if s['role'] == 'anchor']
    recovery = [s for s in old if s['role'] == 'recovery']
    if not anchors or not recovery:
        raise ValueError('LT15原始监督缺保持或恢复角色')
    return {'old_old': anchors + recovery, 'broad_old': anchors + replay + recovery,
            'old_novel': anchors + recovery + novel, 'broad_novel': anchors + replay + recovery + novel}
