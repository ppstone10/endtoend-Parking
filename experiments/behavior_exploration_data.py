"""探索专用闭环监督采集；专家只在拟合源离线使用。"""
import numpy as np
import torch

from experiments.behavior_exploration import first_stop
from experiments.behavior_preservation import coverage_gate
from experiments.executable_behavior import prepare_supervision_frames,run_recorded_branch
from experiments.recovery_execution import select_context,verify_network_suffix,record_arrays
from experiments.reference_compensation import compensate_reference_yaw,reference_internal_safe
from interfaces import Trajectory
from scripts.preflight_behavior_repair import FIT,PROBE,result_row


def run_case(case,model,budget,*,context=None,trajectory=None,replanning_expert=False,control_seed=0):
    if context is not None and control_seed:raise ValueError('分叉上下文不可更换控制seed')
    return run_recorded_branch(model=model,pipeline=case['pipeline'],planner=case['planner'],
                               start=case['start'],goal=case['goal'],vehicle=case['vehicle'],
                               engine_kwargs=case['engine_kwargs'],index=case['index']+10000*control_seed,
                               budget=budget,context=context,fixed_trajectory=trajectory,
                               replanning_expert=replanning_expert)


def save_record(output,name,result,context=None,**arrays):
    states,commands=record_arrays(result)
    if context is not None:arrays['context_state']=context.state.to_array()
    np.savez_compressed(output/f'{name}.npz',states=states,commands=commands,**arrays)


def strict_teacher(case,context,data,budget,output,prefix):
    """确定的场景标签策略；不扫参数，不把未通过分支当正标签。"""
    index=case['index'];attempts=[]
    if case['scene_name']=='S4_dump_area':
        result,engine,frames=run_case(case,None,budget,context=context,replanning_expert=True)
        save_record(output,f'{prefix}-rolling',result,context)
        row=result_row(case,result,engine);attempts.append(dict(kind='rolling',**row))
        return frames if row['reached_with_margin'] else [],attempts
    if case['scene_name']!='S8_weigh_station':return [],[{'decision':'unsupported_teacher_scene'}]
    length=int(data['masks'][index].sum())
    original=Trajectory(data['trajs'][index,:length].copy(),float(np.asarray(data['dt']).reshape(-1)[0]))
    teacher=original
    for update in range(3):
        result,engine,frames=run_case(case,None,budget,context=context,trajectory=teacher)
        save_record(output,f'{prefix}-fixed{update}',result,context,reference=teacher.points,
                    progress_trace=np.asarray(engine.mpc.execution_trace))
        row=result_row(case,result,engine);attempts.append(dict(kind='fixed',update=update,**row))
        if row['reached_with_margin']:return frames,attempts
        if update==2:break
        states,_=record_arrays(result)
        if not len(states):break
        proposed=compensate_reference_yaw(original,teacher,states,engine.mpc.execution_trace)
        if not reference_internal_safe(proposed,case['planner']._collision_checker):
            attempts.append({'update':update+1,'decision':'reject_unsafe_reference'});break
        teacher=proposed
    return [],attempts


@torch.no_grad()
def cache_samples(frames,teacher,role,index,parent,budget):
    dense=prepare_supervision_frames(frames,teacher,role,index,frame_limit=None)
    for sample in dense:
        budget.check()
        output=parent.forward_with_stop(sample['bev'][None],sample['goal'][None],sample['state'][None])
        sample['parent_stop']=output.stop_logits[0].sigmoid().detach().clone()
        sample['parent_length']=first_stop(sample['parent_stop'],parent.stop_threshold)
    selected=np.unique(np.linspace(0,len(dense)-1,min(4,len(dense)),dtype=int)) if dense else []
    return [dense[i] for i in selected],dense


def collect_initial(cases,data,parent,budget,output,expected_arrays):
    sparse=[];dense=[];fit_rows=[];baseline_fit=[];baseline_probe=[]
    for index in FIT+PROBE:
        budget.check();case=cases[index]
        print(f'COLLECT R0 #{index}',flush=True)
        result,engine,frames=run_case(case,parent,budget)
        save_record(output,f'R0-{index}',result)
        states,commands=record_arrays(result)
        with np.load(expected_arrays[index]) as expected:
            if not np.array_equal(states,expected['states']) or not np.array_equal(commands,expected['commands']):
                raise ValueError(f'R0 #{index}未逐元素复现前次实际回合')
        row=result_row(case,result,engine)
        (baseline_fit if index in FIT else baseline_probe).append(row)
        if index not in FIT:continue
        supervision='anchor' if result.success else None
        source={'dataset_index':index,'scene_name':case['scene_name'],'supervision':supervision,
                'teacher_strict':False,'R0':row}
        if result.success:
            short,full=cache_samples(frames,parent,'anchor',index,parent,budget)
        else:
            context,reason=select_context(engine.contexts,engine.margin_violations,result.success)
            source.update(context_step=None if context is None else context.step,context_reason=reason)
            accepted=[]
            if context is not None:
                replay,_,_=run_case(case,parent,budget,context=context)
                source['suffix_verification']=verify_network_suffix(result,replay,context.step)
                accepted,attempts=strict_teacher(case,context,data,budget,output,f'teacher-{index}')
                source['teacher_attempts']=attempts
            short,full=cache_samples(accepted,None,'recovery',index,parent,budget)
            if full:source.update(supervision='recovery',teacher_strict=True)
        sparse.extend(short);dense.extend(full);fit_rows.append(source)
    gate=coverage_gate(fit_rows)
    if not gate['passed']:raise ValueError(f'原始教师严格覆盖失败，不伪造标签：{gate}')
    return {'sparse':sparse,'dense':dense,'fit_sources':fit_rows,'coverage':gate,
            'baseline_fit':baseline_fit,'baseline_probe':baseline_probe}


def collect_learner_states(cases,data,learner,parent,budget,output):
    extra=[];rows=[]
    for index in FIT:
        case=cases[index];result,engine,_=run_case(case,learner,budget)
        save_record(output,f'learner-{index}',result)
        row={'dataset_index':index,'learner':result_row(case,result,engine),'added_frames':0}
        if result.success and not any(engine.margin_violations):
            row['decision']='already_strict_success'
        else:
            context,reason=select_context(engine.contexts,engine.margin_violations,result.success)
            row.update(context_step=None if context is None else context.step,context_reason=reason)
            if context is not None and case['scene_name'] in ('S4_dump_area','S8_weigh_station'):
                replay,_,_=run_case(case,learner,budget,context=context)
                row['suffix_verification']=verify_network_suffix(result,replay,context.step)
                frames,attempts=strict_teacher(case,context,data,budget,output,f'aggregate-{index}')
                _,samples=cache_samples(frames,None,'recovery',index,parent,budget)
                row.update(teacher_attempts=attempts,added_frames=len(samples),
                           decision='strict_teacher_added' if samples else 'teacher_rejected')
                extra.extend(samples)
            else:row['decision']='no_safe_context_or_unsupported_scene'
        rows.append(row)
    return {'extra':extra,'sources':rows,'added_frames':len(extra),
            'aggregation_had_new_labels':bool(extra),'fit_sources_only':True}


def evaluate_cases(indices,cases,model,budget,output,prefix,control_seed=0):
    rows=[]
    for index in indices:
        budget.check();case=cases[index]
        result,engine,_=run_case(case,model,budget,control_seed=control_seed)
        row=result_row(case,result,engine);row['control_seed']=control_seed;rows.append(row)
        save_record(output,f'{prefix}-{index}-control{control_seed}',result)
    return rows
