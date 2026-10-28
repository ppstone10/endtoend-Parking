"""固定S3/S6失败回合的离线教师可执行性；不改变旧教师或部署路径。"""
import numpy as np

from experiments.effective_recovery_data import input_id, query_sample, uniform_contexts
from experiments.retention_quality import audit_source_windows


SOURCES = ((20261020, 1196), (20261021, 1196), (20261020, 2149))


def failure_contexts(contexts, margins):
    """均匀覆盖首次危险前的真实安全访问，至少留下原总时限内200步。"""
    if any(c.step < 0 or c.step >= 600 or c.step % 10 for c in contexts):
        raise ValueError('上下文不是原600步内的重规划相位')
    if len({c.step for c in contexts}) != len(contexts):
        raise ValueError('上下文时间重复')
    first = next((i for i, hit in enumerate(margins) if hit), 600)
    return uniform_contexts([c for c in contexts if c.step <= min(first, 400)])


def validated_query(visited, teacher_frames, case, step, policy, parent, budget,
                    learner_states, teacher_states):
    """只验证教师首帧；拼接真实前缀用于复用只读消费窗口审计。"""
    if step <= 0 or step % 10 or len(learner_states) < step or not len(teacher_states):
        raise ValueError('教师首窗口缺真实前缀/转移')
    sample = query_sample(visited, teacher_frames, case['index'], step, policy, parent, budget)
    states = np.concatenate((learner_states[:step], teacher_states))
    # 审计器的anchor仅为其输入契约；缓存本身仍是recovery，不改任何旧数据。
    audit = audit_source_windows([dict(sample, role='anchor')], states, case, budget.check)[0]
    return sample, audit


def readiness(rows, new_input_ids):
    identities = [(r['seed'], r['dataset_index']) for r in rows]
    complete = len(identities) == len(SOURCES) and set(identities) == set(SOURCES)
    latest = []
    for row in rows:
        queries = row['queries']
        query = max(queries, key=lambda q: q['step']) if queries else None
        latest.append(dict(seed=row['seed'], dataset_index=row['dataset_index'],
            latest_step=None if query is None else query['step'],
            passed=bool(query and query['decision'] == 'strict_query_accepted' and
                        query['input_sha256'] in new_input_ids)))
    return dict(passed=complete and all(r['passed'] for r in latest),
                all_fixed_sources=complete, latest_queries=latest,
                learning_benefit_tested=False, main_protocol_passed=False)
