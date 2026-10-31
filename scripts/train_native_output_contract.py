"""LT43: bounded native-output A/B/AB study; formal jobs are human-run."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.emitted_output_data import RECIPE,file_digest,file_identity,read_json
from experiments.emitted_output_study import batch_plan
from scripts import train_recertified_emitted_joint as previous
from scripts.emitted_teacher_recertification import load_authorized
from scripts.recertified_emitted_fit import expected_files as previous_files,native_evaluate
from scripts.native_output_geometry import NativeGeometry
from scripts.native_output_objectives import ARMS,OBJECTIVE,update
from scripts import native_output_fit as fitting
from scripts.diagnose_fixed_reference import read_shell_json
from scripts.train_emitted_output_joint import StageBudget
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

TEACHER=ROOT/'runs/prepared/emitted-recert-teachers-20261030-v1'
OLD_PREPARED=ROOT/'runs/prepared/recertified-emitted-joint-20261030-v1'
OLD=ROOT/'runs/training/lt42-recertified-emitted-joint-20261030/fit'
PRECHECK=ROOT/'runs/verification/native-contract-20261030/preflight-v1'
FILES=('native_output_geometry.py','native_output_objectives.py','native_output_fit.py','train_native_output_contract.py')
PROTOCOL=dict(arms=list(ARMS),seeds=RECIPE['seeds'],updates_per_arm=200,new_updates=1200,
    checkpoints=[0,50,100,200],candidate='AB',reused_baseline='LT42_joint_only',
    objective=OBJECTIVE,preflight_seconds=90,fit_seconds=7200,inspect_seconds=7200,
    teacher_contract='LT39_zero_rescue_GT',original_lt34_fit_unlocked=False,
    lt40_rejection_preserved=True,validation_used_for_fitting=False)


def source_identity():
    if file_digest(OLD_PREPARED/'config.json')!='bfd133d6c993c88975f0e8cc7fdcadbb94f136f29b74c879ef441cc5c62cec32':
        raise ValueError('original LT42 prepared changed')
    old=previous.precheck(OLD_PREPARED,TEACHER)
    expected={str((OLD/n).resolve()) for n in previous_files()};manifest=read_json(OLD/'artifacts.json')
    actual={str(p.resolve()) for p in OLD.rglob('*') if p.is_file() and p!=OLD/'artifacts.json'}
    if set(manifest)!=expected or actual!=expected or any(file_digest(p)!=h for p,h in manifest.items()):
        raise ValueError('complete immutable LT42 baseline required')
    report=read_json(OLD/'report.json');audit=read_shell_json(OLD.parent/'inspect.log',last_line=True)
    execution=dict(old,config_sha256=file_digest(OLD_PREPARED/'config.json'))
    if (read_json(OLD/'config.json')!=execution or report['status']!='completed' or report['gate']['passed']
            or report['closed_loop_evaluated'] or report['validation_used_for_fitting']
            or not report['training_performed'] or audit.get('status')!='verified'
            or audit.get('gate')!=report['gate']
            or any(audit.get(k)!=v for k,v in dict(tasks=8,samples=32,arms=4,updates=800,checkpoints=16,native_evaluations=512,artifacts=56).items())):
        raise ValueError('verified LT42 negative decision required, never unlock old recipe')
    if read_shell_json(OLD.parent/'console.log',last_line=True)['gate']!=report['gate']:
        raise ValueError('baseline console mismatch')
    for stage,name in (('fit','exit.json'),('inspect','inspect-exit.json')):
        if read_shell_json(OLD.parent/name)!=dict(task='LT42',stage=stage,exit_code=0):raise ValueError('baseline exit mismatch')
    code={**old['context']['code'],**{str((ROOT/'scripts'/n).resolve()):file_digest(ROOT/'scripts'/n) for n in FILES}}
    return dict(schema='native_output_source_v1',code=code,protocol=PROTOCOL,previous_context=old['context'],
        previous_config=file_identity(OLD_PREPARED/'config.json'),baseline_manifest=file_identity(OLD/'artifacts.json'),
        baseline_inspect=file_identity(OLD.parent/'inspect.log'),
        baseline_exits=[file_identity(OLD.parent/n) for n in ('exit.json','inspect-exit.json')],
        teacher_manifest=file_identity(TEACHER/'artifacts.json'))


def supervision(source):
    return load_authorized(TEACHER,source['previous_context'])[0]


def verify_geometry(samples,cases,budget):
    teachers=[];old_rows=[]
    for sample in samples:
        budget.check();geo=NativeGeometry(cases[sample['source_index']]['planner']._collision_checker)
        points=sample['points'][:int(sample['mask'].sum())]
        value=float(geo.loss(points,sample['pose']))
        if not geo.safe_prefix(points,sample['pose']).all() or value>1e-8:
            raise ValueError('certified safe teacher rejected by geometry objective')
        teachers.append(dict(source_index=sample['source_index'],frame_index=sample['frame_index'],loss=value))
    for seed in RECIPE['seeds']:
        folder=OLD/f'{seed}-joint';rows=read_json(folder/'native-0200.json')['rows']
        with np.load(folder/'native-0200.npz') as saved:
            for j,(sample,row) in enumerate(zip(samples,rows)):
                budget.check();geo=NativeGeometry(cases[sample['source_index']]['planner']._collision_checker)
                points=torch.tensor(saved['points'][j,:row['native_length']],requires_grad=True)
                hard=bool(geo.safe_prefix(points,sample['pose'])[-1]);loss=geo.loss(points,sample['pose'])
                if hard!=row['margin_safe'] or (float(loss.detach())>1e-8)==hard:
                    raise ValueError('geometry penalty/hard oracle mismatch')
                gradient=0.
                if not hard:
                    loss.backward();gradient=float(points.grad.abs().sum())
                    if not torch.isfinite(points.grad).all() or gradient<=0:raise ValueError('unsafe geometry lacks finite gradient')
                old_rows.append(dict(seed=seed,source_index=sample['source_index'],frame_index=sample['frame_index'],
                    hard_safe=hard,penetration=float(loss.detach()),gradient_l1=gradient))
    return dict(passed=True,teacher_samples=32,old_native_samples=64,unsafe_samples=sum(not r['hard_safe'] for r in old_rows),teachers=teachers,old_rows=old_rows)


def preflight(output):
    source=source_identity();samples=supervision(source)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);budget=StageBudget(90)
    report=dict(status='running',source=source,paired_updates=[],formal_training=False,weights_reusable_as_formal_origin=False)
    atomic_write_json(output/'context.json',source);atomic_write_json(output/'report.json',report)
    try:
        data,cases=previous.training_cases();proof=verify_geometry(samples,cases,budget)
        atomic_write_json(output/'geometry.json',proof)
        loaded=load_model_checkpoint(source['previous_context']['checkpoint']['path'])
        if loaded.model.stop_threshold!=.5 or loaded.model.dt!=.2 or loaded.model.max_horizon!=320:raise ValueError('native output contract changed')
        report['parent']=native_evaluate(loaded.model,samples,data,cases,output/'parent-native')
        if report['parent']!=read_json(OLD/'20261028-joint/native-0000.json'):raise ValueError('parent native result changed')
        seed=RECIPE['seeds'][0];plan=batch_plan(samples,seed,1)
        for arm in ARMS:
            budget.check();torch.manual_seed(seed);model=deepcopy(loaded.model)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-5);began=time.perf_counter()
            row=update(model,optimizer,[samples[j] for j in plan[0]],cases,arm,seed,1)
            if arm=='B':
                old_first=read_json(OLD/f'{seed}-joint/history.json')[0]
                if row['samples']!=old_first['samples'] or not np.isclose(row['legacy_joint_loss'],old_first['total_loss'],rtol=0,atol=1e-10):
                    raise ValueError('B must reproduce the old objective before adding geometry')
            duration=time.perf_counter()-began
            evaluation=native_evaluate(model,samples,data,cases,output/f'{arm}-post-native')
            report['paired_updates'].append(dict(arm=arm,record=row,update_seconds=duration,evaluation=evaluation))
        if source_identity()!=source:raise ValueError('source changed during preflight')
        supervision(source);budget.check()
        report.update(status='passed',elapsed_seconds=time.perf_counter()-budget.started,geometry_summary={k:proof[k] for k in ('passed','teacher_samples','old_native_samples','unsafe_samples')})
        atomic_write_json(output/'report.json',report)
        atomic_write_json(output/'artifacts.json',{str(p.resolve()):file_digest(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete',error=str(error));atomic_write_json(output/'report.json',report);raise


def frozen_config():
    source=source_identity();supervision(source)
    names=['context.json','report.json','geometry.json']+[f'{n}.{ext}' for n in ('parent-native','A-post-native','B-post-native','AB-post-native') for ext in ('json','npz')]
    manifest=read_json(PRECHECK/'artifacts.json');expected={str((PRECHECK/n).resolve()) for n in names}
    actual={str(p.resolve()) for p in PRECHECK.iterdir() if p.is_file() and p.name!='artifacts.json'}
    if set(manifest)!=expected or actual!=expected or any(file_digest(p)!=h for p,h in manifest.items()):raise ValueError('current complete preflight required')
    report=read_json(PRECHECK/'report.json');proof=read_json(PRECHECK/'geometry.json')
    if (report['status']!='passed' or report['source']!=source or read_json(PRECHECK/'context.json')!=source or report['formal_training']
            or report['weights_reusable_as_formal_origin'] or not proof['passed']
            or [r['arm'] for r in report['paired_updates']]!=list(ARMS)
            or any(r['record']['step']!=1 for r in report['paired_updates'])):raise ValueError('preflight authorization failed')
    return dict(schema='native_output_fit_v1',source=source,checkpoint=source['previous_context']['checkpoint'],
        preflight_manifest=file_identity(PRECHECK/'artifacts.json'),candidate_qualified=False,automatic_training=False)


def prepare(directory):
    directory=Path(directory).resolve()
    if directory.exists():raise FileExistsError('new prepared directory required')
    config=frozen_config();directory.mkdir(parents=True,exist_ok=False);atomic_write_json(directory/'config.json',config);return config


def precheck(directory):
    config=read_json(Path(directory)/'config.json')
    if config!=frozen_config():raise ValueError('source/preflight/data drift; never edit hashes')
    return config


def fit(directory,output, *, audit=False):
    config=precheck(directory);samples=supervision(config['source'])  # before output or optimizer creation
    config=dict(config,config_sha256=file_digest(Path(directory)/'config.json'));data,cases=previous.training_cases()
    if audit:
        result=fitting.inspect(config,samples,data,cases,output,StageBudget(7200));precheck(directory);return result
    return fitting.execute(config,samples,data,cases,output,StageBudget(7200),lambda:(precheck(directory),supervision(config['source'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('preflight','prepare','precheck','fit','inspect'))
    parser.add_argument('--prepared',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='preflight':
        if args.output is None:parser.error('--output required')
        r=preflight(args.output);result=dict(status=r['status'],updates=3,geometry=r['geometry_summary'],elapsed_seconds=r['elapsed_seconds'])
    else:
        if args.prepared is None:parser.error('--prepared required')
        if args.phase in ('prepare','precheck'):
            r=prepare(args.prepared) if args.phase=='prepare' else precheck(args.prepared)
            result=dict(status='prepared' if args.phase=='prepare' else 'passed',tasks=8,samples=32,new_arms=6,reused_arms=2,updates=1200,source_files=len(r['source']['code']))
        else:
            if args.output is None:parser.error('--output required')
            r=fit(args.prepared,args.output,audit=args.phase=='inspect')
            result=r if args.phase=='inspect' else {k:r[k] for k in ('status','gate','elapsed_seconds')}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
