"""EMIT-RECERT-002..004: existing objective, fixed dose and strict native checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from experiments.emitted_output_data import RECIPE,read_json,file_digest
from experiments.emitted_output_study import (
    batch_plan,update,evaluate as original_evaluate,fitting_decision)
from scripts.emitted_teacher_recertification import strict_pose
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


def native_evaluate(model,samples,data,cases,output):
    # Reuse public predict/full-head equality and safety audit; strengthen the
    # legacy <= endpoint flag to the actual TerminalChecker's strict contract.
    report=original_evaluate(model,samples,data,cases,output)
    with np.load(Path(output).with_suffix('.npz'),allow_pickle=False) as raw:
        for row,sample,points,logits in zip(report['rows'],samples,raw['points'],raw['logits']):
            goal=sample['goal'].numpy();tol=sample['tolerances'].numpy()
            row['endpoint']=strict_pose(points[row['native_length']-1],goal,tol)
            row['teacher_index_endpoint']=strict_pose(points[row['teacher_length']-1],goal,tol)
            row['passed']=bool(row['endpoint']['joint_passed'] and row['contact_safe'] and row['margin_safe'])
    initial=[r for r in report['rows'] if r['frame_index']==0]
    report.update(initial_passed=sum(r['passed'] for r in initial),
        all_frames_passed=sum(r['passed'] for r in report['rows']),passed=all(r['passed'] for r in report['rows']),
        teacher_length_used_for_inference=False,strict_position_yaw=True)
    atomic_write_json(Path(output).with_suffix('.json'),report)
    return report


def expected_files():
    names=['config.json','report.json']
    for seed in RECIPE['seeds']:
        names.append(f'plan-{seed}.json')
        for arm in ('control','joint'):
            prefix=f'{seed}-{arm}/'
            names.append(prefix+'history.json')
            for step in RECIPE['checkpoints']:
                names.extend([prefix+f'native-{step:04d}.json',prefix+f'native-{step:04d}.npz',
                              prefix+f'update-{step:04d}.pt'])
    return names


def execute_fit(config,samples,data,cases,destination,budget,final_check):
    """Caller must authorize before this function creates a directory/optimizer."""
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',config_sha256=config['config_sha256'],arms=[],gate=None,
        teacher_artifacts_sha256=config['teacher_artifacts']['sha256'],
        training_performed=False,validation_used_for_fitting=False,closed_loop_evaluated=False,
        original_lt34_fit_unlocked=False,candidate_qualified=False,production_qualified=False)
    atomic_write_json(destination/'config.json',config);atomic_write_json(destination/'report.json',report)
    try:
        loaded=load_model_checkpoint(config['checkpoint']['path'])
        for seed in RECIPE['seeds']:
            plan=batch_plan(samples,seed,RECIPE['updates']);atomic_write_json(destination/f'plan-{seed}.json',plan)
            for arm in ('control','joint'):
                budget.check();torch.manual_seed(seed);model=deepcopy(loaded.model)
                optimizer=torch.optim.Adam(model.parameters(),lr=RECIPE['learning_rate'])
                arm_dir=destination/f'{seed}-{arm}';arm_dir.mkdir()
                history=[];snapshots=[]
                for step in range(RECIPE['updates']+1):
                    budget.check()
                    if step:
                        history.append(update(model,optimizer,[samples[j] for j in plan[step-1]],seed=seed,step=step,
                            joint_weight=RECIPE['joint_weight'] if arm=='joint' else 0.,
                            inspect_gradients=step==1 and arm=='joint'))
                        report['training_performed']=True
                    if step in RECIPE['checkpoints']:
                        diagnostic=native_evaluate(model,samples,data,cases,arm_dir/f'native-{step:04d}')
                        torch.save(dict(schema_version=1,model_name=loaded.model_name,model_config=loaded.model_config,
                            model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=-1,
                            experiment='EMIT-RECERT',seed=seed,arm=arm,updates=step,
                            config_sha256=config['config_sha256'],experimental_only=True),arm_dir/f'update-{step:04d}.pt')
                        snapshots.append(dict(step=step,**diagnostic));atomic_write_json(arm_dir/'history.json',history)
                        print(f'FIT {seed} {arm} step={step}/200 initial={diagnostic["initial_passed"]}/8 '
                            f'all={diagnostic["all_frames_passed"]}/32',flush=True)
                report['arms'].append(dict(seed=seed,arm=arm,updates=200,checkpoints=snapshots,final=snapshots[-1]))
                atomic_write_json(destination/'report.json',report)
        final_check();budget.check()
        report.update(status='completed',gate=fitting_decision(report['arms']),
            elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(destination/'report.json',report)
        atomic_write_json(destination/'artifacts.json',
            {str(p.resolve()):file_digest(p) for p in destination.rglob('*') if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete',error=str(error));atomic_write_json(destination/'report.json',report)
        raise


def inspect_fit(config,samples,data,cases,destination,budget):
    destination=Path(destination).resolve();report=read_json(destination/'report.json')
    artifacts=read_json(destination/'artifacts.json')
    expected={str((destination/n).resolve()) for n in expected_files()}
    actual={str(p.resolve()) for p in destination.rglob('*') if p.is_file() and p!=destination/'artifacts.json'}
    if (set(artifacts)!=expected or actual!=expected or read_json(destination/'config.json')!=config
            or report['status']!='completed' or report['config_sha256']!=config['config_sha256']
            or report['teacher_artifacts_sha256']!=config['teacher_artifacts']['sha256']
            or not report['training_performed'] or report['validation_used_for_fitting']
            or report['closed_loop_evaluated'] or report['original_lt34_fit_unlocked']
            or report['candidate_qualified'] or report['production_qualified']):
        raise ValueError('fit completion/config/artifact denominator mismatch')
    if any(file_digest(p)!=digest for p,digest in artifacts.items()):raise ValueError('fit artifact drift')
    parent=load_model_checkpoint(config['checkpoint']['path']).model.state_dict()
    reconstructed=[];predictions=0
    with tempfile.TemporaryDirectory(prefix='emitted-recert-audit-') as temp:
        for seed in RECIPE['seeds']:
            budget.check();plan=batch_plan(samples,seed,200)
            if read_json(destination/f'plan-{seed}.json')!=plan:raise ValueError('paired sample plan drift')
            for arm in ('control','joint'):
                folder=destination/f'{seed}-{arm}';history=read_json(folder/'history.json')
                if len(history)!=200:raise ValueError('fixed update dose changed')
                for step,record in enumerate(history,start=1):
                    expected_samples=[dict(source_index=samples[j]['source_index'],frame_index=samples[j]['frame_index']) for j in plan[step-1]]
                    if record['step']!=step or record['samples']!=expected_samples:
                        raise ValueError('update/sample order drift')
                    if not np.isfinite([record['total_loss'],record['unclipped_gradient_norm']]).all():
                        raise ValueError('nonfinite update history')
                if arm=='joint':
                    gradient=history[0].get('joint_head_gradient_norms',[])
                    if len(gradient)!=2 or not np.isfinite(gradient).all() or not all(x>0 for x in gradient):
                        raise ValueError('missing both-head gradient evidence')
                snapshots=[]
                for step in RECIPE['checkpoints']:
                    budget.check();checkpoint=folder/f'update-{step:04d}.pt'
                    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if (payload.get('experiment')!='EMIT-RECERT' or payload.get('seed')!=seed
                            or payload.get('arm')!=arm or payload.get('updates')!=step
                            or payload.get('config_sha256')!=config['config_sha256']):
                        raise ValueError('checkpoint identity drift')
                    if step==0:
                        if set(payload['model_state'])!=set(parent):raise ValueError('parent parameter keys changed')
                        for k,v in parent.items():torch.testing.assert_close(payload['model_state'][k],v,rtol=0,atol=0)
                        if payload['optimizer_state']['state']:raise ValueError('Adam not reset at shared origin')
                    model=load_model_checkpoint(checkpoint).model
                    rebuilt=native_evaluate(model,samples,data,cases,Path(temp)/'native')
                    if rebuilt!=read_json(folder/f'native-{step:04d}.json'):raise ValueError('native hard-output metrics mismatch')
                    with np.load(Path(temp)/'native.npz',allow_pickle=False) as a:
                        with np.load(folder/f'native-{step:04d}.npz',allow_pickle=False) as b:
                            if set(a.files)!=set(b.files):raise ValueError('native output fields changed')
                            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                    snapshots.append(dict(step=step,**rebuilt));predictions+=32
                    print(f'AUDIT {seed} {arm} step={step} inputs=32',flush=True)
                reconstructed.append(dict(seed=seed,arm=arm,updates=200,checkpoints=snapshots,final=snapshots[-1]))
    if report['arms']!=reconstructed or report['gate']!=fitting_decision(reconstructed):
        raise ValueError('final prescribed checkpoint decision mismatch')
    return dict(status='verified',tasks=8,samples=32,arms=4,updates=800,checkpoints=16,
        native_evaluations=predictions,artifacts=len(artifacts),gate=report['gate'],
        original_lt34_fit_unlocked=False,closed_loop_evaluated=False)
