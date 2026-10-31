"""NATIVE-FIT-003/005: fixed factorial study and independent checkpoint audit."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from experiments.emitted_output_data import RECIPE,read_json,file_digest
from experiments.emitted_output_study import batch_plan
from scripts.recertified_emitted_fit import native_evaluate
from scripts.native_output_objectives import ARMS,update
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


def expected_files():
    names=['config.json','report.json']
    for seed in RECIPE['seeds']:
        names.append(f'plan-{seed}.json')
        for arm in ARMS:
            names.append(f'{seed}-{arm}/history.json')
            for step in RECIPE['checkpoints']:
                names.extend(f'{seed}-{arm}/{name}-{step:04d}.{ext}' for name,ext in
                    (('native','json'),('native','npz'),('update','pt')))
    return names


def baseline(config):
    old=read_json(Path(config['source']['baseline_manifest']['path']).parent/'report.json')
    return [dict(seed=a['seed'],arm='reused_LT42_joint',final=a['final']) for a in old['arms'] if a['arm']=='joint']


def decision(arms):
    if len(arms)!=6 or {(a['seed'],a['arm']) for a in arms}!={(s,a) for s in RECIPE['seeds'] for a in ARMS}:
        raise ValueError('six new arms required; never stitch seeds or arms')
    passed=all(a['final']['passed'] for a in arms if a['arm']=='AB')
    return dict(passed=passed,decision='ready_for_separate_fixed_and_rolling_plan' if passed else
        'stop_native_contract_recipe_diagnose_failed_component',candidate_arm='AB',
        diagnostic_passed={arm:all(a['final']['passed'] for a in arms if a['arm']==arm) for arm in ('A','B')},
        candidate_qualified=False,production_qualified=False)


def compare(arms,reused):
    key=lambda rows:{(r['source_index'],r['frame_index']) for r in rows if r['passed']}
    rows=[]
    for arm in arms:
        old=next(a for a in reused if a['seed']==arm['seed'])['final']
        before,after=key(old['rows']),key(arm['final']['rows'])
        rows.append(dict(seed=arm['seed'],arm=arm['arm'],initial_passed=arm['final']['initial_passed'],
            all_passed=arm['final']['all_frames_passed'],gained_from_LT42=[list(k) for k in sorted(after-before)],
            lost_from_LT42=[list(k) for k in sorted(before-after)],
            difficult_initial={str(i):next(r['passed'] for r in arm['final']['rows'] if r['source_index']==i and r['frame_index']==0)
                for i in (860,1290,2315)}))
    return rows


def _check_origin_outputs(config,seed,report,output):
    old=Path(config['source']['baseline_manifest']['path']).parent/f'{seed}-joint/native-0000'
    if report!=read_json(old.with_suffix('.json')):raise ValueError('initial native report changed')
    with np.load(Path(output).with_suffix('.npz')) as new:
        with np.load(old.with_suffix('.npz')) as previous:
            if set(new.files)!=set(previous.files):raise ValueError('initial native fields changed')
            for name in new.files:np.testing.assert_array_equal(new[name],previous[name])


def execute(config,samples,data,cases,output,budget,final_check):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',config_sha256=config['config_sha256'],arms=[],baseline=baseline(config),
        new_arms=6,reused_arms=2,training_performed=False,validation_used_for_fitting=False,
        closed_loop_evaluated=False,original_lt34_fit_unlocked=False,candidate_qualified=False,production_qualified=False)
    atomic_write_json(output/'config.json',config);atomic_write_json(output/'report.json',report)
    try:
        loaded=load_model_checkpoint(config['checkpoint']['path'])
        for seed in RECIPE['seeds']:
            plan=batch_plan(samples,seed,200);atomic_write_json(output/f'plan-{seed}.json',plan)
            for arm in ARMS:
                budget.check();torch.manual_seed(seed);model=deepcopy(loaded.model)
                optimizer=torch.optim.Adam(model.parameters(),lr=RECIPE['learning_rate'])
                folder=output/f'{seed}-{arm}';folder.mkdir();history=[];snapshots=[]
                for step in range(201):
                    budget.check()
                    if step:
                        history.append(update(model,optimizer,[samples[j] for j in plan[step-1]],cases,arm,seed,step))
                        report['training_performed']=True
                    if step in RECIPE['checkpoints']:
                        evaluation=native_evaluate(model,samples,data,cases,folder/f'native-{step:04d}')
                        if step==0:_check_origin_outputs(config,seed,evaluation,folder/'native-0000')
                        torch.save(dict(schema_version=1,model_name=loaded.model_name,model_config=loaded.model_config,
                            model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=-1,
                            experiment='NATIVE-FIT',experimental_only=True,seed=seed,arm=arm,updates=step,
                            config_sha256=config['config_sha256']),folder/f'update-{step:04d}.pt')
                        snapshots.append(dict(step=step,**evaluation));atomic_write_json(folder/'history.json',history)
                        print(f'FIT {seed} {arm} step={step}/200 initial={evaluation["initial_passed"]}/8 all={evaluation["all_frames_passed"]}/32',flush=True)
                report['arms'].append(dict(seed=seed,arm=arm,updates=200,checkpoints=snapshots,final=snapshots[-1]))
                atomic_write_json(output/'report.json',report)
        final_check();budget.check()
        report.update(status='completed',gate=decision(report['arms']),comparison=compare(report['arms'],report['baseline']),
            elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(output/'report.json',report)
        atomic_write_json(output/'artifacts.json',{str(p.resolve()):file_digest(p) for p in output.rglob('*') if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete',error=str(error));atomic_write_json(output/'report.json',report);raise


def inspect(config,samples,data,cases,output,budget):
    output=Path(output).resolve();report=read_json(output/'report.json');manifest=read_json(output/'artifacts.json')
    expected={str((output/n).resolve()) for n in expected_files()}
    actual={str(p.resolve()) for p in output.rglob('*') if p.is_file() and p!=output/'artifacts.json'}
    if set(manifest)!=expected or actual!=expected or read_json(output/'config.json')!=config:
        raise ValueError('complete exact 82 artifacts/config required')
    if any(file_digest(p)!=h for p,h in manifest.items()):raise ValueError('artifact drift')
    if (report['status']!='completed' or report['config_sha256']!=config['config_sha256'] or not report['training_performed']
            or report['baseline']!=baseline(config) or report['new_arms']!=6 or report['reused_arms']!=2
            or any(report[k] for k in ('validation_used_for_fitting','closed_loop_evaluated','original_lt34_fit_unlocked','candidate_qualified','production_qualified'))):
        raise ValueError('completion/qualification mismatch')
    parent=load_model_checkpoint(config['checkpoint']['path']).model.state_dict();rebuilt=[];evaluations=0
    with tempfile.TemporaryDirectory(prefix='native-contract-audit-') as temporary:
        for seed in RECIPE['seeds']:
            plan=batch_plan(samples,seed,200)
            if read_json(output/f'plan-{seed}.json')!=plan:raise ValueError('sample plan changed')
            for arm in ARMS:
                folder=output/f'{seed}-{arm}';history=read_json(folder/'history.json')
                if len(history)!=200:raise ValueError('fixed dose changed')
                for step,row in enumerate(history,1):
                    if row['step']!=step or row['samples']!=[dict(source_index=samples[j]['source_index'],frame_index=samples[j]['frame_index']) for j in plan[step-1]]:
                        raise ValueError('history/sample drift')
                    numeric=[v for k,v in row.items() if k.endswith('loss') or k=='unclipped_gradient_norm']
                    if not np.isfinite(numeric).all():raise ValueError('nonfinite history')
                gradients=history[0]['head_gradient_norms']
                if len(gradients)!=2 or not np.isfinite(gradients).all() or gradients[0]<=0 or (arm!='B' and gradients[1]<=0):
                    raise ValueError('missing required head gradients')
                snapshots=[]
                for step in RECIPE['checkpoints']:
                    budget.check();checkpoint=folder/f'update-{step:04d}.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if any(payload.get(k)!=v for k,v in dict(experiment='NATIVE-FIT',seed=seed,arm=arm,updates=step,config_sha256=config['config_sha256']).items()):
                        raise ValueError('checkpoint identity changed')
                    if not all(torch.isfinite(v).all() for v in payload['model_state'].values()):raise ValueError('nonfinite weights')
                    if step==0:
                        if set(parent)!=set(payload['model_state']) or payload['optimizer_state']['state']:raise ValueError('origin/reset mismatch')
                        for k,v in parent.items():torch.testing.assert_close(v,payload['model_state'][k],rtol=0,atol=0)
                    elif not payload['optimizer_state']['state'] or any(int(v['step'])!=step for v in payload['optimizer_state']['state'].values()):
                        raise ValueError('Adam update count mismatch')
                    model=load_model_checkpoint(checkpoint).model;target=Path(temporary)/'native'
                    evaluation=native_evaluate(model,samples,data,cases,target)
                    if evaluation!=read_json(folder/f'native-{step:04d}.json'):raise ValueError('hard evaluation mismatch')
                    with np.load(target.with_suffix('.npz')) as new:
                        with np.load(folder/f'native-{step:04d}.npz') as saved:
                            if set(new.files)!=set(saved.files):raise ValueError('native fields changed')
                            for k in new.files:np.testing.assert_array_equal(new[k],saved[k])
                    if step==0:_check_origin_outputs(config,seed,evaluation,target)
                    snapshots.append(dict(step=step,**evaluation));evaluations+=32
                    print(f'AUDIT {seed} {arm} step={step} inputs=32',flush=True)
                rebuilt.append(dict(seed=seed,arm=arm,updates=200,checkpoints=snapshots,final=snapshots[-1]))
    if (report['arms']!=rebuilt or report['gate']!=decision(rebuilt)
            or report['comparison']!=compare(rebuilt,report['baseline'])):raise ValueError('final candidate decision mismatch')
    return dict(status='verified',tasks=8,samples=32,new_arms=6,reused_arms=2,updates=1200,
        checkpoints=24,native_evaluations=evaluations,artifacts=82,gate=report['gate'],closed_loop_evaluated=False)
