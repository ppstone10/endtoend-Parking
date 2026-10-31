"""Frozen eight-task teacher/fixed-network/rolling-network execution diagnosis."""
from collections import Counter
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from experiments.emitted_output_data import (
    INDICES, precheck as parent_precheck, load_cohort, case_at, read_json,
    file_identity, execution_identity, file_digest,
)
from scripts.emitted_execution_capture import record_branch, paired_prefix
from scripts.train_emitted_output_joint import StageBudget
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PARENT=ROOT/'runs/prepared/emitted-joint-20261028-v1'
TEACHER=ROOT/'runs/training/lt34-emitted-joint-20261028/teacher'
PARENT_SHA='86cbf4c55664de4cd51a9c4098d4edf2b7ec26a7302a269c654244917d3cad6f'


def classify_triplet(teacher,fixed,rolling):
    if not teacher['success'] or teacher['collision']:
        return 'teacher_execution_task_failure'
    if not fixed['success']:
        return ('replanning_recovers_fixed_failure' if rolling['success'] else
                'network_fixed_reference_execution_failure')
    return 'all_three_reached' if rolling['success'] else 'replanning_associated_failure'


def metric_summary(rows):
    if not rows:
        raise ValueError('empty task denominator')
    return dict(tasks=len(rows),successes=sum(r['success'] for r in rows),
        collisions=sum(r['collision'] for r in rows),
        margin_affected_tasks=sum(r['margin_violation_steps']>0 for r in rows),
        margin_violation_steps=sum(r['margin_violation_steps'] for r in rows),
        arrival_rate=sum(r['success'] for r in rows)/len(rows),
        collision_rate=sum(r['collision'] for r in rows)/len(rows))


def current_code():
    return {**execution_identity(),**{str(p):file_digest(p) for p in
        [Path(__file__).resolve(),ROOT/'scripts/emitted_execution_capture.py']}}


def teacher_snapshot():
    parent=parent_precheck(PARENT)
    if file_digest(PARENT/'config.json')!=PARENT_SHA:
        raise ValueError('LT34 parent config identity changed')
    artifacts=read_json(TEACHER/'artifacts.json')
    expected={str((TEACHER/name).resolve()) for name in ['report.json','samples.pt',
        *[f'teacher-{i}.npz' for i in INDICES]]}
    if set(artifacts)!=expected:
        raise ValueError('teacher evidence denominator incomplete')
    for path,digest in artifacts.items():
        if file_digest(path)!=digest:
            raise ValueError(f'teacher evidence changed: {path}')
    teacher=read_json(TEACHER/'report.json')
    if (teacher['status']!='completed' or not teacher['formal'] or teacher['training_performed']
            or teacher['config_sha256']!=PARENT_SHA
            or [r['dataset_index'] for r in teacher['rows']]!=INDICES
            or [r['task_id'] for r in teacher['rows']]!=[s['task_id'] for s in parent['sources']]):
        raise ValueError('teacher provenance mismatch')
    # Failed margin qualification is retained, never relabeled as a pass or used for training.
    return parent,teacher,artifacts


def prepare(destination):
    destination=Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('new prepared directory required')
    parent,teacher,artifacts=teacher_snapshot()
    config=dict(schema='emitted_execution_diagnosis_v1',code=current_code(),
        parent_config=file_identity(PARENT/'config.json'),
        teacher_artifacts=file_identity(TEACHER/'artifacts.json'),teacher_files=artifacts,
        indices=INDICES,sources=parent['sources'],checkpoint=parent['inputs']['parent'],
        protocol=dict(safety_mode='none',max_steps=600,replan_every=10,control_seed=0,
            mpc_dt=.1,mpc_horizon=10,stop_threshold=.5,trajectory_dt=.2,horizon=320),
        teacher_strict_gate=teacher['gate'],training_authorized=False,
        attribution_only=True,budget_seconds=1800,smoke_seconds=90)
    destination.mkdir(parents=True)
    atomic_write_json(destination/'config.json',config)
    return config


def precheck(prepared):
    config=read_json(Path(prepared)/'config.json')
    parent,teacher,artifacts=teacher_snapshot()
    if (config['schema']!='emitted_execution_diagnosis_v1' or config['code']!=current_code()
            or config['indices']!=INDICES or config['sources']!=parent['sources']
            or config['checkpoint']!=parent['inputs']['parent']
            or config['teacher_files']!=artifacts or config['teacher_strict_gate']!=teacher['gate']
            or config['training_authorized'] or not config['attribution_only']
            or config['protocol']!=dict(safety_mode='none',max_steps=600,replan_every=10,control_seed=0,
                mpc_dt=.1,mpc_horizon=10,stop_threshold=.5,trajectory_dt=.2,horizon=320)
            or config['budget_seconds']!=1800 or config['smoke_seconds']!=90):
        raise ValueError('diagnosis identity/protocol drift; do not edit hashes to bypass')
    for item in (config['parent_config'],config['teacher_artifacts']):
        if file_identity(item['path'])!=item:
            raise ValueError('frozen evidence changed')
    return config


def run(prepared,destination,*,smoke=False):
    config=precheck(prepared)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',formal=not smoke,config_sha256=file_digest(Path(prepared)/'config.json'),
        indices=INDICES[:1] if smoke else INDICES,rows=[],training_performed=False,
        candidate_qualified=False,teacher_source=str(TEACHER))
    atomic_write_json(destination/'report.json',report)
    atomic_write_json(destination/'config.json',config)
    budget=StageBudget(config['smoke_seconds'] if smoke else config['budget_seconds'])
    try:
        data,manifest,vehicle=load_cohort(PARENT)
        model=load_model_checkpoint(config['checkpoint']['path']).model
        if model.stop_threshold!=.5 or model.max_horizon!=320 or model.dt!=.2:
            raise ValueError('native model contract mismatch')
        teacher=read_json(TEACHER/'report.json')
        for j,index in enumerate(report['indices']):
            budget.check();case=case_at(data,manifest,vehicle,j);branch_rows={};captured={}
            for name,fixed in [('network_fixed',True),('network_rolling',False)]:
                print(f'#{index} {name} BEGIN',flush=True)
                row,raw=record_branch(case,model,destination/f'{index:04d}-{name}',budget,fixed=fixed)
                branch_rows[name]=row;captured[name]=raw
                np.testing.assert_array_equal(raw['first_raw']['bev'],data['bevs'][j])
                print(f'#{index} {name} success={row["success"]} collision={row["collision"]} '
                      f'margin_steps={row["margin_violation_steps"]} controls={row["steps"]}',flush=True)
            fixed=captured['network_fixed'];rolling=captured['network_rolling']
            for key in ('points','logits','bev','goal','motion','pose','length'):
                np.testing.assert_array_equal(fixed['first_raw'][key],rolling['first_raw'][key])
            np.testing.assert_array_equal(fixed['first_global'],rolling['first_global'])
            prefix=paired_prefix(fixed,rolling)
            teacher_row=dict(dataset_index=index,task_id=case['task_id'],scene_name=case['scene_name'],
                branch='teacher_fixed',reused=True,source_file=str(TEACHER/f'teacher-{index}.npz'),
                **teacher['rows'][j]['execution'])
            report['rows'].append(dict(dataset_index=index,task_id=case['task_id'],scene_name=case['scene_name'],
                teacher=teacher_row,**branch_rows,paired_prefix=prefix,
                pattern=classify_triplet(teacher_row,branch_rows['network_fixed'],branch_rows['network_rolling'])))
            atomic_write_json(destination/'report.json',report)
        precheck(prepared);budget.check()
        report.update(status='completed',elapsed_seconds=time.perf_counter()-budget.started,
            metrics={name:metric_summary([r[name] for r in report['rows']])
                     for name in ('teacher','network_fixed','network_rolling')},
            patterns=dict(Counter(r['pattern'] for r in report['rows'])),
            limits='八训练任务机制诊断；非全数据因果占比/泛化/训练收益；余量侵入与碰撞分别统计')
        atomic_write_json(destination/'report.json',report)
        atomic_write_json(destination/'artifacts.json',
            {str(p.resolve()):file_digest(p) for p in destination.iterdir() if p.is_file()})
        return report
    except BaseException as exc:
        report.update(status='incomplete',error=str(exc),elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(destination/'report.json',report)
        raise


def inspect(prepared,destination):
    precheck(prepared);destination=Path(destination).resolve()
    artifacts=read_json(destination/'artifacts.json')
    actual={str(p.resolve()) for p in destination.iterdir() if p.is_file() and p.name!='artifacts.json'}
    if set(artifacts)!=actual:
        raise ValueError('diagnosis artifact set incomplete')
    for path,digest in artifacts.items():
        if file_digest(path)!=digest:
            raise ValueError(f'diagnosis artifact drift: {path}')
    report=read_json(destination/'report.json')
    expected=INDICES if report['formal'] else INDICES[:1]
    if (report['status']!='completed' or report['indices']!=expected
            or [r['dataset_index'] for r in report['rows']]!=expected
            or report['config_sha256']!=file_digest(Path(prepared)/'config.json')
            or file_digest(destination/'config.json')!=report['config_sha256']):
        raise ValueError('diagnosis completion/config/denominator mismatch')
    required={'config.json','report.json'}
    for index in expected:
        for branch in ('network_fixed','network_rolling'):
            stem=f'{index:04d}-{branch}'
            required.update((f'{stem}.json',f'{stem}.npz',f'{stem}-predictions.npz'))
    if {Path(p).name for p in artifacts}!=required:
        raise ValueError('required per-branch evidence missing')
    for row in report['rows']:
        if row['pattern']!=classify_triplet(row['teacher'],row['network_fixed'],row['network_rolling']):
            raise ValueError('attribution pattern mismatch')
        branches={}
        for branch in ('network_fixed','network_rolling'):
            path=destination/f'{row["dataset_index"]:04d}-{branch}'
            detail=read_json(path.with_suffix('.json'))
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
                states=archive['states_after']
            branches[branch]=dict(states=states,commands=np.array([f['command'] for f in detail['frames']]))
        if row['paired_prefix']!=paired_prefix(branches['network_fixed'],branches['network_rolling']):
            raise ValueError('saved paired execution prefix mismatch')
    for name in ('teacher','network_fixed','network_rolling'):
        if report['metrics'][name]!=metric_summary([r[name] for r in report['rows']]):
            raise ValueError('metric denominator mismatch')
    return dict(status='verified',formal=report['formal'],tasks=len(expected),artifacts=len(artifacts),
                metrics=report['metrics'],patterns=report['patterns'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','precheck','smoke','run','inspect'],required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='prepare':
        config=prepare(args.prepared);result=dict(status='prepared',indices=config['indices'])
    elif args.phase=='precheck':
        config=precheck(args.prepared);result=dict(status='passed',indices=config['indices'])
    else:
        if args.output is None:parser.error('--output required')
        if args.phase=='inspect':result=inspect(args.prepared,args.output)
        else:
            report=run(args.prepared,args.output,smoke=args.phase=='smoke')
            result={key:report[key] for key in ('status','formal','elapsed_seconds','metrics','patterns')}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
