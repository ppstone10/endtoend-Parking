"""显式支持起点的实际访问查询；不修改旧非零恢复契约。"""
import numpy as np

from experiments.effective_recovery_data import query_sample
from experiments.retention_quality import audit_source_windows


def select_entry_contexts(contexts, margins):
    steps=[c.step for c in contexts]
    if (not steps or 0 not in steps or len(set(steps))!=len(steps) or
            any(not isinstance(s,int) or s<0 or s>=600 or s%10 for s in steps) or len(margins)>600):
        raise ValueError('缺起始安全上下文或重规划相位/范围非法')
    first=next((i for i,hit in enumerate(margins) if hit),600)
    ordered=sorted(contexts,key=lambda c:c.step)
    nonzero=[c for c in ordered if 0<c.step<=min(first,400)]
    selected=[next(c for c in ordered if c.step==0)]+([nonzero[-1]] if nonzero else [])
    return selected,dict(selected_steps=[c.step for c in selected],first_margin_step=None if first==600 else first,
        nonzero_missing=not bool(nonzero),selection='zero_and_latest_nonzero_before_first_margin_max400')


def validated_entry_query(visited,teacher_frames,case,step,policy,parent,budget,learner_states,teacher_states):
    learner_states=np.asarray(learner_states);teacher_states=np.asarray(teacher_states)
    if (not isinstance(step,int) or not 0<=step<=400 or step%10 or not teacher_frames or
            learner_states.ndim!=2 or learner_states.shape[1]!=5 or len(learner_states)<step or
            teacher_states.ndim!=2 or teacher_states.shape[1]!=5 or not len(teacher_states) or
            not np.isfinite(learner_states).all() or not np.isfinite(teacher_states).all()):
        raise ValueError('起点/非零查询缺真实前缀、教师首帧或实际转移')
    sample=query_sample(visited,teacher_frames,case['index'],step,policy,parent,budget)
    states=np.concatenate((learner_states[:step],teacher_states))
    window=audit_source_windows([dict(sample,role='anchor')],states,case,budget.check)[0]
    return sample,window
