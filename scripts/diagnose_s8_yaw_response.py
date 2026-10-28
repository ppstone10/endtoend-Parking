"""90秒零训练：五个固定首次侵入事件的同随机流航向窗口响应。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from controller import MPCController
from dataset import DatasetGenerator
from experiments.behavior_exploration_data import run_case
from experiments.executable_behavior import PreflightBudget,RecordingMPC
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.recovery_execution import record_arrays
from experiments.reference_compensation import reference_internal_safe
from experiments.s8_yaw_response import probe_command
from interfaces import Trajectory
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix,result_row
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s8_entry_recovery import exact_record,PREVIOUS
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


ENTRY=ROOT/'runs/diagnostics/s8-entry-recovery-preflight-20261022-v1'
HASHES={'config.json':'4683e8e80fdeb2c6a3358ecc00491c6b5ded0ec4a8e95a3b7390ee5451ee8f32',
    'report.json':'cf7ac8c6bbbb891097f410fe2bcc732e9cb2d265b4db60893ac3be4d3e74abc3'}
EVENTS=((20261020,2338,0,0,5),(20261020,2338,0,1,5),(20261020,2351,0,0,17),
    (20261020,2351,0,2,201),(20261021,2351,40,1,197))
MPC_FIELDS=('dt','horizon','max_v','max_omega','w_xy','w_yaw','w_terminal','w_v','w_omega',
    'population','elite_frac','iterations')


def prepare_config(budget):
    verify_files({str(ENTRY/k):v for k,v in HASHES.items()})
    report=json.loads((ENTRY/'report.json').read_text(encoding='utf-8'))
    cfg=json.loads((ENTRY/'config.json').read_text(encoding='utf-8'));verify_config(cfg,budget)
    if report['status']!='completed' or report['complete_queries']!=6:raise ValueError('非完整六查询前置')
    for name,digest in report['artifacts_sha256'].items():
        budget.check();verify_files({str(artifact_path(ENTRY,name)):digest})
    cfg=deepcopy(cfg);cfg.update(kind='s8_first_loss_yaw_response_v1',budget_seconds=90,
        events=[list(e) for e in EVENTS],amplitudes=[0.,-.01,.01],teacher_labels_created=False,
        response_interpretation='fixed_secants_not_converged_gradients_or_teacher_success')
    cfg['frozen_files'].update({str(ENTRY/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(ENTRY,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s8_yaw_response
    for p in (Path(__file__).resolve(),Path(s8_yaw_response.__file__).resolve()):cfg['code'][str(p)]=file_digest(p)
    verify_config(cfg,budget);return cfg


def capture_event(case,context,trajectory,archive,event,budget):
    snapshot=[];seen=[0];compute=RecordingMPC.compute
    def recording(mpc,reference,state):
        if seen[0]==event:snapshot.append(deepcopy((mpc,reference,state)))
        seen[0]+=1
        return compute(mpc,reference,state)
    with patch.object(RecordingMPC,'compute',recording):
        result,engine,_=run_case(case,None,budget,context=context,trajectory=trajectory)
    states,commands=exact_record(result,archive)
    with np.load(archive) as z:
        if not np.array_equal(np.asarray(engine.margin_violations),z['margin_violations']):raise ValueError('教师余量未重现')
        first=np.flatnonzero(z['margin_violations'])
        if not len(first) or first[0]!=event:raise ValueError('首次侵入事件漂移')
    if len(snapshot)!=1:raise ValueError('事件前MPC快照缺失或重复')
    return snapshot[0],states,commands,result_row(case,result,engine)


def diagnose_event(case,branch_start,snapshot,states,commands,event,budget,directory):
    mpc,trajectory,state=snapshot;checker=case['planner']._collision_checker
    zero=probe_command(mpc,trajectory,state,0.)
    expected=case['engine_kwargs']['vehicle_model'].step(state,zero['command'],mpc.dt).to_array()
    baseline_cmd=np.asarray([zero['command'].v,zero['command'].omega])
    if not np.array_equal(baseline_cmd,commands[event]) or not np.array_equal(expected,states[event]):
        raise ValueError('事件原命令/下一状态未逐元素复现')
    params={k:getattr(mpc,k) for k in MPC_FIELDS};params['catchup_window']=mpc._catchup_window
    atomic_write_json(directory/'snapshot.json',dict(mpc_parameters=params,rng=mpc._rng.bit_generator.state,
        progress_before=mpc._progress_idx,event=event))
    np.savez_compressed(directory/'snapshot.npz',reference=trajectory.points,dt=np.asarray(trajectory.dt),
        aligned_cache=mpc._aligned_cache,last_u=mpc._last_u,state=state.to_array(),
        expected_command=commands[event],expected_next_state=states[event])
    aligned=mpc._align_dt(trajectory);reference_pose=aligned[zero['progress_index']]
    actual=states[event,:3]
    swaps=dict(actual_pose_safe=checker.pose_free(*actual),
        actual_xy_reference_yaw_safe=checker.pose_free(actual[0],actual[1],reference_pose[2]),
        reference_xy_actual_yaw_safe=checker.pose_free(reference_pose[0],reference_pose[1],actual[2]),
        xy_error=float(np.linalg.norm(actual[:2]-reference_pose[:2])),
        yaw_error_rad=float(np.arctan2(np.sin(actual[2]-reference_pose[2]),np.cos(actual[2]-reference_pose[2]))))
    responses=[]
    for amplitude in (0.,-.01,.01):
        budget.check();response=zero if amplitude==0. else probe_command(mpc,trajectory,state,amplitude)
        reference=response['trajectory'];command=response['command']
        next_state=case['engine_kwargs']['vehicle_model'].step(state,command,mpc.dt)
        full=reference_internal_safe(reference,checker)
        window=mpc._reference_points(mpc._align_dt(reference),response['progress_index'])
        previous=state.to_array()[:3]
        local=all(checker.swept_segment_free(a,b) for a,b in zip(np.vstack((previous,window[:-1])),window))
        row=dict(amplitude=amplitude,command=[command.v,command.omega],next_state=next_state.to_array().tolist(),
            progress_index=response['progress_index'],reference_internal_safe=full,
            original_start_connected_safe=checker.swept_segment_free(branch_start.to_array()[:3],reference.points[0]),
            event_to_first_reference_safe=checker.swept_segment_free(previous,reference.points[0]),
            event_window_connected_safe=bool(local),
            actual_step_safe=checker.swept_segment_free(previous,next_state.to_array()[:3]),
            next_pose_safe=checker.pose_free(next_state.x,next_state.y,next_state.yaw),
            command_delta=(np.asarray([command.v,command.omega])-baseline_cmd).tolist())
        responses.append(row)
        np.savez_compressed(directory/f'probe-{amplitude:+.2f}.npz',reference=reference.points,basis=response['basis'],
            command=np.asarray(row['command']),next_state=next_state.to_array())
    minus=np.asarray(responses[1]['command']);plus=np.asarray(responses[2]['command'])
    return dict(exact_baseline_command_and_next_state=True,swaps=swaps,responses=responses,
        centered_secant=((plus-minus)/.02).tolist(),left_secant=((baseline_cmd-minus)/.01).tolist(),
        right_secant=((plus-baseline_cmd)/.01).tolist())


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();budget=PreflightBudget(90);torch.set_num_threads(1)
    report=dict(status='running',stage='identity',events=[],training_updates=0,teacher_labels_created=False,
        learning_benefit_tested=False,allowed_to_prepare_training_plan=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-started;atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            cfg=prepare_config(budget);atomic_write_json(output/'config.json',cfg);report['config_sha256']=file_digest(output/'config.json')
            data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train']);vehicle=VehicleConfig(**manifest['vehicle_model'])
            parent=load_model_checkpoint(INPUTS['R0']).model;cached={}
            for seed,index,step,update,event in EVENTS:
                budget.check();report['stage']=f'event-{seed}-{index}-{step}-{update}';save()
                if (seed,index) not in cached:
                    case=build_case(data,manifest,index,vehicle);model,_=load_candidate(cfg['checkpoints']['preservation_student'][str(seed)],parent)
                    original,engine,_=run_case(case,model,budget)
                    exact_record(original,PREVIOUS/f'preservation_student-{seed}/source-{index}-control0.npz')
                    cached[seed,index]=(case,engine.contexts)
                case,contexts=cached[seed,index];context=next(c for c in contexts if c.step==step)
                archive=ENTRY/f'source-{seed}-{index}/query-{step}/teacher-{update}.npz'
                with np.load(archive) as z:trajectory=Trajectory(z['reference'].copy(),float(z['dt']))
                snapshot,states,commands,execution=capture_event(case,context,trajectory,archive,event,budget)
                directory=output/f'event-{seed}-{index}-{step}-{update}';directory.mkdir()
                row=dict(seed=seed,source_index=index,query_step=step,teacher_update=update,event=event,
                    baseline_execution=execution,**diagnose_event(case,context.state,snapshot,states,commands,event,budget,directory))
                report['events'].append(row);save();print('EVENT',seed,index,step,update,event,
                    [(r['amplitude'],r['actual_step_safe']) for r in row['responses']],flush=True)
            verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',complete_events=len(report['events']),
                decision='fixed_local_response_diagnostic_only_no_teacher_or_gradient_claim')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p!=output/'report.json' and p.name not in ('run.lock','console.log')}
            save();print('FINISHED',report['complete_events'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def precheck():
    cfg=prepare_config(PreflightBudget(45));print('YAW RESPONSE INPUTS VERIFIED; ZERO ROLLOUT/CONTROL/TRAINING',flush=True);return cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--precheck',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.precheck:
        if args.output:parser.error('precheck不写output')
        precheck()
    elif args.output is None:parser.error('需要不存在的新output')
    else:run(args.output)
