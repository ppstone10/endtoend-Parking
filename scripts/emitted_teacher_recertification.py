"""EMIT-RECERT-001: training-only labels from frozen LT39 teacher execution."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from controller.mpc import MPCController
from experiments.behavior_preservation import consumed_point_indices
from experiments.emitted_output_data import (
    INDICES,load_cohort,case_at,read_json,file_digest,reference_audit)
from interfaces import Trajectory,VehicleState
from runtime import NetworkSource
from scripts.diagnose_emitted_execution import PARENT
from scripts.feasible_cem_capture import audit_execution,audit_predictions
from scripts.zero_hold_motion_capture import audit_rescue_execution
from training.data import to_local
from training.reporting import atomic_write_json

ROOT=Path(__file__).resolve().parents[1]
TEACHERS=ROOT/'runs/diagnostics/lt39-zero-motion-rescue-20261029/results'
ORIGINAL_TEACHERS=ROOT/'runs/training/lt34-emitted-joint-20261028/teacher'


def strict_pose(point,goal,tolerances):
    delta=np.asarray(point,dtype=np.float64)-np.asarray(goal,dtype=np.float64)
    position=float(np.linalg.norm(delta[:2]))
    yaw=float(abs(np.arctan2(np.sin(delta[2]),np.cos(delta[2]))))
    return dict(position_m=position,yaw_rad=yaw,
        joint_passed=bool(position<float(tolerances[0]) and yaw<float(tolerances[1])))


def remaining_label(reference,aligned_count,progress,state):
    offset=int(consumed_point_indices(len(reference),aligned_count,progress,10)[0])
    local=to_local(reference[offset:],state.x,state.y,state.yaw)
    if not 1<=len(local)<=320:raise ValueError('remaining teacher prefix must fit H320 without truncation')
    points=torch.zeros(320,3);points[:len(local)]=torch.from_numpy(local)
    mask=(torch.arange(320)<len(local)).float()
    return points,mask,offset


def teacher_gate(episodes,frames):
    if ([r['index'] for r in episodes]!=INDICES or len(frames)!=32
            or any(sum(f['source_index']==i for f in frames)!=4 for i in INDICES)
            or len({(f['source_index'],f['frame_index']) for f in frames})!=32):
        raise ValueError('all eight sources and 32 unique predeclared frames required')
    failed=[i for i in INDICES if not next(r['passed'] for r in episodes if r['index']==i)
            or not all(f['passed'] for f in frames if f['source_index']==i)]
    return dict(passed=not failed,failed_sources=failed,
        decision='authorized_for_one_bounded_paired_fit' if not failed else 'stop_fix_supervision_or_control_no_fit',
        original_mpc_gate_replaced=False,experimental_teacher_contract='LT39_zero_rescue_GT',
        original_lt34_fit_unlocked=False,candidate_qualified=False,production_qualified=False)


def build_samples(budget, *, replay_teacher):
    data,manifest,vehicle=load_cohort(PARENT)
    samples=[];frames=[];episodes=[];cases={};transitions=0
    for ordinal,index in enumerate(INDICES):
        budget.check();case=case_at(data,manifest,vehicle,ordinal);cases[index]=case
        output=TEACHERS/f'{index:04d}-teacher_fixed';detail=read_json(output.with_suffix('.json'))
        if (detail['dataset_index']!=index or detail['task_id']!=case['task_id']
                or detail['branch']!='teacher_fixed' or detail['profile']!='zero_rescue'
                or detail['training_performed'] or detail['truncated_preflight']
                or detail['geometry_source']!='simulator_ground_truth' or detail['required_margin_m']!=.2
                or detail['execution_npz_sha256']!=file_digest(output.with_suffix('.npz'))):
            raise ValueError('frozen experimental training teacher identity mismatch')
        audit_predictions(output,detail,ORIGINAL_TEACHERS/f'teacher-{index}.npz')
        if replay_teacher:
            transitions+=audit_execution(output,detail,case)['selected_h10_transitions']
            audit_rescue_execution(output,detail,case)
        with np.load(output.with_suffix('.npz'),allow_pickle=False) as saved:
            reference=saved['global_plan_0000'];states=saved['states_before']
            np.testing.assert_array_equal(saved['initial_state'],case['start'].to_array())
            aligned=MPCController(dt=.1,horizon=10)._align_dt(Trajectory(reference,.2))
            if len(states)<4:raise ValueError('four unique teacher inputs required')
            chosen=np.unique(np.linspace(0,len(states)-1,4,dtype=int))
            source=NetworkSource(deepcopy(case['pipeline']),None);source.begin(case['start'],case['goal'])
            terminal=case['engine_kwargs']['terminal']
            tolerances=torch.tensor([terminal.tol_pos,terminal.tol_yaw],dtype=torch.float64)
            for step_value in chosen:
                budget.check();step=int(step_value);state=VehicleState.from_array(states[step])
                progress=detail['frames'][step]['progress_index']
                points,mask,offset=remaining_label(reference,len(aligned),progress,state)
                bev=source.sensor_pipeline.capture_bev(state.x,state.y,state.yaw)
                if bev.camera_projection!='per_view_v2':raise ValueError('native per-view input required')
                if step==0:np.testing.assert_array_equal(bev.data,data['bevs'][ordinal])
                goal=torch.from_numpy(source._to_local_goal(state).astype(np.float32))
                length=int(mask.sum());pose=state.to_array()[:3]
                endpoint=strict_pose(points[length-1].numpy(),goal.numpy(),tolerances.numpy())
                safety=reference_audit(points[:length].numpy(),pose,case)
                passed=bool(endpoint['joint_passed'] and all(safety.values()))
                frame=dict(source_index=index,task_id=case['task_id'],frame_index=step,
                    actual_state=state.to_array().tolist(),progress_index=progress,reference_offset=offset,
                    length=length,endpoint=endpoint,**safety,passed=passed,
                    teacher_execution_sha256=detail['execution_npz_sha256'])
                frames.append(frame)
                samples.append(dict(bev=torch.from_numpy(np.array(bev.data,copy=True)).float(),goal=goal,
                    state=torch.tensor([state.v,state.omega],dtype=torch.float32),points=points,mask=mask,
                    pose=pose.copy(),tolerances=tolerances.clone(),source_index=index,frame_index=step,
                    reference_offset=offset))
        passed=bool(detail['success'] and not detail['collision'] and not detail['margin_violation_steps']
                    and not detail['safety_stop'] and detail['control_timing']['over_100ms']==0)
        episodes.append(dict(index=index,task_id=case['task_id'],steps=detail['steps'],passed=passed,
            execution_json_sha256=file_digest(output.with_suffix('.json')),
            execution_npz_sha256=detail['execution_npz_sha256']))
    return samples,frames,episodes,data,cases,transitions


def certify(output,context,budget):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',context=context,gate=None,training_performed=False,
        reused_teacher_episodes=8,new_teacher_episodes=0,validation_used_for_fitting=False)
    atomic_write_json(output/'report.json',report)
    try:
        samples,frames,episodes,_,_,transitions=build_samples(budget,replay_teacher=True)
        report.update(status='completed',frames=frames,episodes=episodes,gate=teacher_gate(episodes,frames),
            selected_samples=len(samples),scalar_h10_transitions_reaudited=transitions)
        torch.save(samples,output/'samples.pt')
        atomic_write_json(output/'report.json',report)
        atomic_write_json(output/'artifacts.json',
            {str(p.resolve()):file_digest(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete',error=str(error));atomic_write_json(output/'report.json',report)
        raise


def load_authorized(output,context):
    output=Path(output).resolve();report=read_json(output/'report.json')
    expected={str((output/n).resolve()) for n in ('report.json','samples.pt')}
    artifacts=read_json(output/'artifacts.json')
    actual={str(p.resolve()) for p in output.iterdir() if p.is_file() and p.name!='artifacts.json'}
    if set(artifacts)!=expected or actual!=expected:raise ValueError('teacher authorization artifacts incomplete')
    if any(file_digest(p)!=digest for p,digest in artifacts.items()):raise ValueError('teacher authorization drift')
    if (report['status']!='completed' or report['context']!=context or report['training_performed']
            or report['validation_used_for_fitting'] or report['selected_samples']!=32
            or report['gate']!=teacher_gate(report['episodes'],report['frames']) or not report['gate']['passed']):
        raise ValueError('new teacher gate not passed; fitting forbidden before optimizer/output creation')
    samples=torch.load(output/'samples.pt',weights_only=False,map_location='cpu')
    if len(samples)!=32:raise ValueError('supervision denominator changed')
    for sample,frame in zip(samples,report['frames']):
        if (sample['source_index']!=frame['source_index'] or sample['frame_index']!=frame['frame_index']
                or sample['reference_offset']!=frame['reference_offset']
                or int(sample['mask'].sum())!=frame['length']):
            raise ValueError('sample provenance mismatch')
    return samples,report


def audit_authorization(output,context,budget):
    saved,report=load_authorized(output,context)
    samples,frames,episodes,_,_,_=build_samples(budget,replay_teacher=False)
    if frames!=report['frames'] or episodes!=report['episodes']:raise ValueError('reconstructed teacher provenance changed')
    for a,b in zip(saved,samples):
        for key in ('bev','goal','state','points','mask','tolerances'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        np.testing.assert_array_equal(a['pose'],b['pose'])
    return dict(status='verified',samples=32,tasks=8,rebuilt_native_inputs=32,gate=report['gate'])
