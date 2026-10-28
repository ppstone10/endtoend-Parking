"""固定训练域成功保持/可执行教师预检；不自动生成或运行长任务。"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from experiments.behavior_preservation import coverage_gate,probe_gate,fit_bounded_copy
from experiments.executable_behavior import PreflightBudget,run_recorded_branch,prepare_supervision_frames
from experiments.reference_compensation import compensate_reference_yaw,reference_internal_safe
from experiments.recovery_execution import select_context,execution_summary,execute_branch,verify_network_suffix,record_arrays
from interfaces import Trajectory,VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import TerminalChecker
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest,metadata_only,runtime_identity
from sim import VehicleConfig,DifferentialDriveModel
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

FIT=[860,1975,1334,1559,2330]
PROBE=[861,1976,1680,1441,2348]
INPUTS={
    'train':ROOT/'data/task_dataset/tracked_pivot_v8_3000/train.npz',
    'manifest':ROOT/'data/task_dataset/tracked_pivot_v8_3000/manifest.json',
    'R0':ROOT/'runs/validation/arch-ab/dep-epoch0007.pt',
    'R2':ROOT/'runs/training/recovery-balance-ab-20261018-v2/R2/checkpoints/epoch0002.pt',
    'val':ROOT/'data/task_dataset/tracked_pivot_v8_3000/val.npz',
    'selection':ROOT/'data/task_dataset/independent_selection_v1/selection.npz'}
EXPECTED={'train':'8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f',
          'R0':'d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8',
          'R2':'fe258ad0186a0ce40717befbc96452bd2f959ea2e97293b323aaa2a33f26b620',
          'val':'ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963',
          'selection':'797e9e5d0d02576c45df65c74f142cc05e83770eb6bbfcc28be560dbc722d51d'}


def frozen_config(*,refine_teacher=False,rolling_s4=False):
    inputs={k:{'path':str(p),'sha256':file_digest(p)} for k,p in INPUTS.items()}
    if any(inputs[k]['sha256']!=v for k,v in EXPECTED.items()):raise ValueError('固定输入漂移')
    metadata=metadata_only(INPUTS['train'])
    if len(metadata)!=2400 or len(set(FIT+PROBE))!=10:raise ValueError('预检来源分组不符')
    reserved=collect_metadata_ids(metadata_only(INPUTS['val']))|collect_metadata_ids(metadata_only(INPUTS['selection']))
    if collect_metadata_ids(metadata)&reserved:raise ValueError('训练来源与保留集重叠')
    code={**code_identity(),str(Path(__file__).resolve()):file_digest(__file__)}
    return {'kind':'bounded_behavior_repair_preflight','inputs':inputs,'code':code,'runtime':runtime_identity(),
            'teacher_refinement':{'enabled':refine_teacher,'max_updates':2,'gain':.5,'smooth_points':5,'max_yaw_offset':.1},
            'rolling_s4_teacher':rolling_s4,
            'fit_indices':FIT,'probe_indices':PROBE,'budget_seconds':180,'optimizer_steps':12,'learning_rate':1e-5,
            'max_steps':600,'replan_every':10,'control_seed':0,'training_reserved_overlap':0,
            'source_tasks':{str(i):{'task_id':metadata[i]['task_id'],'scene_name':metadata[i]['scene_name']} for i in FIT+PROBE}}


def build_case(data,manifest,index,vehicle):
    metadata=data['task_meta'][index]
    restored=matrix.reconstruct_dataset_task(metadata,root_seed=manifest['seed'],vehicle=vehicle)
    planner,pipeline=matrix._build_pipeline(matrix.EXPERIMENT_SPECS['E5'],restored.task,vehicle)
    state=VehicleState.from_array(data['states'][index])
    initial=deepcopy(pipeline);initial.set_target_goals([restored.goal])
    if not np.array_equal(initial.capture_bev(state.x,state.y,state.yaw).data,data['bevs'][index]):
        raise ValueError('实际E5初态输入与训练归档不一致，不切换通道')
    contact=RectangleFootprintCollisionChecker(restored.task.scene.env,vehicle_length=vehicle.length,vehicle_width=vehicle.width,
                                               collision_margin=0.,resolution=vehicle.collision_check_resolution)
    kwargs={'vehicle_model':DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
            'terminal':TerminalChecker(restored.tol_pos,restored.tol_yaw),'env':restored.task.scene.env,
            'collision_checker':contact,**vehicle.collision_kwargs()}
    return {'index':index,'task_id':metadata['task_id'],'scene_name':metadata['scene_name'],'planner':planner,
            'pipeline':pipeline,'start':state,'goal':restored.goal,'vehicle':vehicle,'engine_kwargs':kwargs}


def rollout(case,model,budget,*,context=None,trajectory=None,replanning_expert=False):
    return run_recorded_branch(model=model,pipeline=case['pipeline'],planner=case['planner'],start=case['start'],
                               goal=case['goal'],vehicle=case['vehicle'],engine_kwargs=case['engine_kwargs'],
                               index=case['index'],budget=budget,context=context,fixed_trajectory=trajectory,
                               replanning_expert=replanning_expert)


def result_row(case,result,engine):
    return {'dataset_index':case['index'],'task_id':case['task_id'],'scene_name':case['scene_name'],
            **execution_summary(result,engine.margin_violations)}


def run(output,*,refine_teacher=False,rolling_s4=False):
    if rolling_s4 and not refine_teacher:raise ValueError('组合教师须同时显式启用refine_teacher')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('预检输出必须不存在，不覆盖/续跑')
    torch.set_num_threads(1);torch.manual_seed(20261020)
    config=frozen_config(refine_teacher=refine_teacher,rolling_s4=rolling_s4)
    output.mkdir(parents=True,exist_ok=False);atomic_write_json(output/'config.json',config)
    started=time.perf_counter();budget=PreflightBudget(180)
    report={'status':'running','fit_sources':[],'baseline_probe':[],'candidate_probe':[],
            'training_performed':False,'effective':False,'long_task_plan_created':False,
            'main_protocol_passed':False,'automatic_deployment':False,'automatic_training':False,
            'config_sha256':file_digest(output/'config.json')}
    def save():
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_write_json(output/'report.json',report)
    try:
        data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train'])
        vehicle=VehicleConfig(**manifest['vehicle_model'])
        R0=load_model_checkpoint(INPUTS['R0']).model;R2=load_model_checkpoint(INPUTS['R2']).model
        if R0.stop_threshold!=R2.stop_threshold or R0.stop_threshold!=.26000000000000006:raise ValueError('停止阈值不符')
        pending=[];cases={};originals={}
        for index in FIT+PROBE:
            budget.check();case=build_case(data,manifest,index,vehicle);cases[index]=case
            print(f'R0 #{index} {case["scene_name"]} BEGIN',flush=True)
            result,engine,frames=rollout(case,R0,budget);originals[index]=(result,engine,frames)
            row=result_row(case,result,engine)
            states,commands=record_arrays(result)
            np.savez_compressed(output/f'R0-{index}.npz',states=states,commands=commands)
            if index in PROBE:report['baseline_probe'].append(row)
            print(f'R0 #{index} {row["failure"] or "success"} steps={row["steps"]}',flush=True)
            save()
        for index in FIT:
            case=cases[index];result,engine,frames=originals[index]
            row={'dataset_index':index,'task_id':case['task_id'],'scene_name':case['scene_name'],
                 'R0':result_row(case,result,engine),'supervision':None,'teacher_strict':False}
            if result.success:
                row.update(supervision='anchor',teacher_strict=not any(engine.margin_violations))
                pending.append((frames,R0,'anchor',index))
            else:
                print(f'R2 #{index} BEGIN',flush=True)
                r2,e2,f2=rollout(case,R2,budget);row['R2']=result_row(case,r2,e2)
                if r2.success:
                    row.update(supervision='improve',teacher_strict=not any(e2.margin_violations))
                    pending.append((f2,R2,'improve',index))
                else:
                    context,reason=select_context(engine.contexts,engine.margin_violations,result.success)
                    row.update(context_step=None if context is None else context.step,context_reason=reason)
                    if context is not None:
                        replay,replay_margins=execute_branch(context,'network',R0,case['planner'],case['goal'],case['engine_kwargs'],max_steps=600-context.step,replan_every=10)
                        row['suffix_verification']=verify_network_suffix(result,replay,context.step)
                        length=int(np.asarray(data['masks'][index]).sum())
                        teacher=Trajectory(np.array(data['trajs'][index,:length],copy=True),float(np.asarray(data['dt']).reshape(-1)[0]))
                        print(f'FIXED EXPERT #{index} step={context.step} BEGIN',flush=True)
                        expert,expert_engine,expert_frames=rollout(case,None,budget,context=context,trajectory=teacher)
                        row['fixed_expert']=result_row(case,expert,expert_engine)
                        states,commands=record_arrays(expert)
                        np.savez_compressed(output/f'expert-{index}.npz',states=states,commands=commands,context_state=context.state.to_array())
                        accepted=row['fixed_expert']['reached_with_margin']
                        use_rolling=rolling_s4 and case['scene_name']=='S4_dump_area'
                        if use_rolling and not accepted:
                            expert,expert_engine,expert_frames=rollout(case,None,budget,context=context,replanning_expert=True)
                            states,commands=record_arrays(expert)
                            row['rolling_expert']=result_row(case,expert,expert_engine)
                            np.savez_compressed(output/f'rolling-expert-{index}.npz',states=states,commands=commands,
                                                context_state=context.state.to_array())
                            accepted=row['rolling_expert']['reached_with_margin']
                            print(f'ROLLING #{index} strict={accepted} steps={expert.steps}',flush=True)
                        if refine_teacher and not use_rolling and not accepted:
                            original_teacher=teacher
                            row['reference_compensation']=[]
                            for update in range(1,3):
                                budget.check()
                                proposed=compensate_reference_yaw(original_teacher,teacher,states,expert_engine.mpc.execution_trace)
                                safe=reference_internal_safe(proposed,case['planner']._collision_checker)
                                attempt={'update':update,'reference_internal_safe':safe}
                                row['reference_compensation'].append(attempt)
                                np.savez_compressed(output/f'reference-{index}-update{update}.npz',points=proposed.points,dt=proposed.dt)
                                if not safe:
                                    attempt['decision']='reject_unsafe_reference';break
                                teacher=proposed
                                expert,expert_engine,expert_frames=rollout(case,None,budget,context=context,trajectory=teacher)
                                states,commands=record_arrays(expert)
                                attempt['execution']=result_row(case,expert,expert_engine)
                                np.savez_compressed(output/f'expert-{index}-update{update}.npz',states=states,commands=commands,
                                                    progress_trace=expert_engine.mpc.execution_trace,context_state=context.state.to_array())
                                accepted=attempt['execution']['reached_with_margin']
                                print(f'REFINE #{index} update={update} strict={accepted} margin={sum(expert_engine.margin_violations)}',flush=True)
                                save()
                                if accepted:break
                        if accepted:
                            row.update(supervision='recovery',teacher_strict=True)
                            pending.append((expert_frames,None,'recovery',index))
            report['fit_sources'].append(row)
            print(f'FIT #{index} role={row["supervision"]} strict={row["teacher_strict"]}',flush=True);save()
        coverage=coverage_gate(report['fit_sources']);report['teacher_coverage']=coverage
        if frozen_config(refine_teacher=refine_teacher,rolling_s4=rolling_s4)!=config:raise ValueError('采集期间输入或执行源身份变化')
        if not coverage['passed']:
            report.update(status='completed',decision='teacher_coverage_failed_no_training_no_long_task')
            save();print(json.dumps({'decision':report['decision'],'coverage':coverage},ensure_ascii=False),flush=True);return report
        samples=[]
        for frames,teacher,role,index in pending:
            budget.check();samples.extend(prepare_supervision_frames(frames,teacher,role,index))
        if not {s['source_index'] for s in samples}<=set(FIT):raise ValueError('探针混入拟合')
        report['supervision_samples']=len(samples)
        candidate,history=fit_bounded_copy(R0,samples,check_budget=budget.check)
        report.update(training_performed=True,optimizer_steps=12,loss_history=history)
        torch.save({'model_state':candidate.state_dict(),'purpose':'bounded_preflight_not_deployment','threshold':candidate.stop_threshold},output/'candidate-state.pt')
        for index in PROBE:
            print(f'CANDIDATE #{index} BEGIN',flush=True)
            result,engine,frames=rollout(cases[index],candidate,budget)
            report['candidate_probe'].append(result_row(cases[index],result,engine));save()
        gate=probe_gate(report['baseline_probe'],report['candidate_probe'])
        report.update(status='completed',effective=gate['passed'],probe_gate=gate,
                      decision='network_probe_gain_ready_for_separate_long_plan' if gate['passed'] else 'network_probe_failed_no_long_task')
        if frozen_config(refine_teacher=refine_teacher,rolling_s4=rolling_s4)!=config:raise ValueError('执行中输入或执行源身份变化')
        save();print(json.dumps({'decision':report['decision'],'probe_gate':gate},ensure_ascii=False),flush=True)
        return report
    except BaseException as exc:
        report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',
                      error=str(exc),effective=False,decision='preflight_incomplete_no_long_task')
        save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--refine-teacher',action='store_true',help='离线教师至多两次固定航向补偿；不改原MPC或门禁')
    parser.add_argument('--rolling-s4',action='store_true',help='显式组合教师：S4复用持续重规划，S8固定补偿')
    args=parser.parse_args();run(args.output,refine_teacher=args.refine_teacher,rolling_s4=args.rolling_s4)
