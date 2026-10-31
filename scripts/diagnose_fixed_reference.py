"""LT41: fixed native reference vs frozen LT40 rolling failures; no training."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.emitted_output_data import read_json,file_digest,file_identity
from scripts import validate_zero_motion_retention as previous
from scripts.diagnose_emitted_execution import metric_summary
from scripts.feasible_cem_capture import audit_execution,audit_predictions
from scripts.zero_hold_motion_capture import record_rescue_branch,audit_rescue_execution
from scripts.zero_motion_retention_capture import audit_forward
from scripts.fixed_reference_analysis import diagnose_pair
from scripts.train_emitted_output_joint import StageBudget
from training.reporting import atomic_write_json

PREVIOUS=ROOT/'runs/prepared/zero-motion-retention-20261029-v1'
PREVIOUS_SHA='443ee0150933fc0c2a9f3e9db58f8899ca3677a98106d2c4aaad66208a708115'
BASELINE=ROOT/'runs/diagnostics/lt40-zero-motion-retention-20261029/results'
INDICES=[2,4,87,104,127,132,133,134,135,138,139,148,151,156,183,185,186,188,189,191,207]
SMOKE=[2,133]
PROTOCOL=dict(max_steps=600,smoke_steps=12,smoke_seconds=90,budget_seconds=7200,
    inspect_seconds=7200,branch='network_fixed',rolling='reuse_LT40_candidate_only',
    only_change='freeze_first_native_reference',seed='validation_dataset_index',
    stop_threshold=.5,network_horizon=320,trajectory_dt=.2,dt=.1,horizon=10,
    population=80,iterations=5,replan_every=10,required_margin_m=.2,control_limit_ms=100,
    geometry_source='simulator_ground_truth',training_authorized=False)


def read_shell_json(path, *, last_line=False):
    blob=Path(path).read_bytes()
    text=blob.decode('utf-16' if blob[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig')
    return json.loads(text.strip().splitlines()[-1] if last_line else text)


def code_identity():
    files=[Path(__file__),ROOT/'scripts/fixed_reference_analysis.py']
    return {**previous.code_identity(),**{str(p.resolve()):file_digest(p) for p in files}}


def frozen_config():
    if file_digest(PREVIOUS/'config.json')!=PREVIOUS_SHA:raise ValueError('LT40 prepared identity changed')
    old=previous.precheck(PREVIOUS)
    previous.verify_artifacts(BASELINE,previous.required_files(old))
    report=read_json(BASELINE/'report.json')
    if (report['status']!='completed' or not report['formal'] or report['jobs']!=previous.jobs(old)
            or report['config_sha256']!=PREVIOUS_SHA or read_json(BASELINE/'config.json')!=old
            or len(report['rows'])!=250 or report['training_performed']):
        raise ValueError('complete original LT40 required')
    calculated=previous.summarize(report['rows'],old,formal=True)
    if any(report.get(k)!=v for k,v in calculated.items()):raise ValueError('LT40 decision mismatch')
    failures=[r for r in report['rows'] if r['arm']=='candidate' and not r['success']]
    if ([r['dataset_index'] for r in failures]!=INDICES
            or len({r['task_id'] for r in failures})!=21
            or any(r['steps']!=600 or r['failure']!='oscillation' or r['zero_hold_steps']
                   or r['controller_stats'].get('rescue_attempts',0) or r['collision']
                   or r['margin_violation_steps'] or not r['initial_margin_safe'] for r in failures)):
        raise ValueError('all 21 predeclared LT40 nonzero failures required')
    shell=BASELINE.parent
    for stage,name in (('run','exit.json'),('inspect','inspect-exit.json')):
        if read_shell_json(shell/name)!=dict(task='LT40',stage=stage,exit_code=0):
            raise ValueError('LT40 successful run and full inspect required')
    audit=read_shell_json(shell/'inspect.log',last_line=True)
    if (audit['status']!='verified' or not audit['formal'] or audit['tasks']!=125
            or audit['episodes']!=250 or audit['artifacts']!=752
            or any(audit.get(k)!=v for k,v in calculated.items())):
        raise ValueError('LT40 complete audit record mismatch')
    return dict(schema='fixed_reference_diagnosis_v1',code=code_identity(),runtime=old['runtime'],
        protocol=PROTOCOL,indices=INDICES,smoke_indices=SMOKE,
        sources=[s for s in old['sources'] if s['index'] in INDICES],checkpoint=old['checkpoint'],
        previous_config=file_identity(PREVIOUS/'config.json'),baseline_artifacts=file_identity(BASELINE/'artifacts.json'),
        baseline_report=file_identity(BASELINE/'report.json'),
        baseline_inspect_log=file_identity(shell/'inspect.log'),
        baseline_exits={name:file_identity(shell/name) for name in ('exit.json','inspect-exit.json')},
        training_performed=False,candidate_qualified=False,production_qualified=False)


def prepare(path):
    path=Path(path).resolve()
    if path.exists():raise FileExistsError('new prepared directory required; no overwrite')
    config=frozen_config()
    path.mkdir(parents=True,exist_ok=False)
    atomic_write_json(path/'config.json',config)
    return config


def precheck(path):
    config=read_json(Path(path)/'config.json')
    if config!=frozen_config():raise ValueError('frozen code/input/protocol drift; never rewrite hashes')
    return config


def required_files(smoke=False):
    names=['config.json','report.json']
    for index in SMOKE if smoke else INDICES:
        stem=f'{index:04d}-network_fixed'
        names.extend([stem+'.json',stem+'.npz',stem+'-predictions.npz',f'{index:04d}-diagnosis.json'])
    return names


def summarize(rows, diagnoses, *, formal):
    indices=INDICES if formal else SMOKE
    if ([r['dataset_index'] for r in rows]!=indices or [d['dataset_index'] for d in diagnoses]!=indices
            or len({r['task_id'] for r in rows})!=len(indices)
            or any(r['branch']!='network_fixed' or r['profile']!='zero_rescue' for r in rows)
            or any(d['task_id']!=r['task_id'] for d,r in zip(diagnoses,rows))):
        raise ValueError('fixed 21-task diagnosis denominator/order/identity required')
    kinds=dict(Counter(d['classification'] for d in diagnoses))
    good=[d['dataset_index'] for d in diagnoses if d['classification']=='fixed_reference_recovers_rolling_failure']
    failed=[d['dataset_index'] for d in diagnoses if d['classification']=='fixed_reference_execution_also_failed']
    unqualified=[d['dataset_index'] for d in diagnoses if d['classification']=='fixed_arrival_unqualified_risk_or_latency']
    if not formal:decision='truncated_smoke_only'
    elif len(failed)==21:decision='all_fixed_also_fail_check_reference_control_contract'
    elif len(good)==21:decision='fixed_recovers_all_prioritize_replanning_continuity'
    else:decision='split_by_task_no_global_fix'
    return dict(metrics=metric_summary(rows),classification_counts=kinds,
        decision=dict(status=decision,rolling_related_indices=good,fixed_also_failed_indices=failed,
            fixed_arrival_unqualified_indices=unqualified,diagnosis_complete=bool(formal),
            lt40_rejection_preserved=True),training_unlocked=False,
        candidate_qualified=False,production_qualified=False)


def case_data():
    config=read_json(PREVIOUS/'config.json')
    return (*previous.load_cases(config),config)


def audit_fixed(output,row,case,model,budget):
    output=Path(output)
    detail=read_json(output.with_suffix('.json'))
    if (any(detail.get(k)!=v for k,v in row.items()) or detail['task_id']!=case['task_id']
            or detail['execution_npz_sha256']!=file_digest(output.with_suffix('.npz'))
            or detail['candidate_qualified'] or detail['training_performed']
            or detail['geometry_source']!='simulator_ground_truth' or detail['required_margin_m']!=.2):
        raise ValueError('new fixed evidence/identity mismatch')
    transitions=audit_execution(output,detail,case)['selected_h10_transitions']
    audit_rescue_execution(output,detail,case)
    audit_predictions(output,detail)
    predictions=audit_forward(output,model,budget)
    return transitions,predictions


def run(prepared,destination, *, smoke=False):
    config=precheck(prepared)
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    budget=StageBudget(PROTOCOL['smoke_seconds'] if smoke else PROTOCOL['budget_seconds'])
    report=dict(status='running',formal=not smoke,indices=SMOKE if smoke else INDICES,rows=[],diagnoses=[],
        config_sha256=file_digest(Path(prepared)/'config.json'),training_performed=False,
        reused_rolling_episodes=2 if smoke else 21)
    atomic_write_json(destination/'config.json',config);atomic_write_json(destination/'report.json',report)
    try:
        data,manifest,vehicle,old=case_data();model=previous.native_model(old)
        for index in report['indices']:
            budget.check()
            case=previous.case_at(data,manifest,vehicle,old,index)
            output=destination/f'{index:04d}-network_fixed'
            print(f'FIXED #{index} BEGIN',flush=True)
            row=record_rescue_branch(case,model,output,budget,branch='network_fixed',
                max_steps=PROTOCOL['smoke_steps'] if smoke else PROTOCOL['max_steps'])
            diagnosis=diagnose_pair(output,BASELINE/f'{index:04d}-candidate',case,formal=not smoke)
            atomic_write_json(destination/f'{index:04d}-diagnosis.json',diagnosis)
            report['rows'].append(row);report['diagnoses'].append(diagnosis)
            print(f'FIXED #{index} success={row["success"]} collision={row["collision"]} '
                f'margin={row["margin_violation_steps"]} steps={row["steps"]} '
                f'class={diagnosis["classification"]}',flush=True)
            atomic_write_json(destination/'report.json',report)
        precheck(prepared);budget.check()
        report.update(status='completed',elapsed_seconds=time.perf_counter()-budget.started,
            **summarize(report['rows'],report['diagnoses'],formal=not smoke))
        atomic_write_json(destination/'report.json',report)
        atomic_write_json(destination/'artifacts.json',
            {str(p.resolve()):file_digest(p) for p in destination.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete',error=str(error),elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(destination/'report.json',report)
        raise


def inspect(prepared,destination):
    config=precheck(prepared);destination=Path(destination).resolve()
    report=read_json(destination/'report.json');smoke=not report['formal']
    indices=SMOKE if smoke else INDICES
    if (report['status']!='completed' or report['indices']!=indices or len(report['rows'])!=len(indices)
            or len(report['diagnoses'])!=len(indices) or report['reused_rolling_episodes']!=len(indices)
            or report['config_sha256']!=file_digest(Path(prepared)/'config.json')
            or read_json(destination/'config.json')!=config or report['training_performed']):
        raise ValueError('completion/config/denominator mismatch')
    artifacts=previous.verify_artifacts(destination,required_files(smoke))
    data,manifest,vehicle,old=case_data();model=previous.native_model(old)
    budget=StageBudget(PROTOCOL['smoke_seconds'] if smoke else PROTOCOL['inspect_seconds'])
    new_transitions,old_transitions,predictions=0,0,0
    diagnoses=[]
    for index,row in zip(indices,report['rows']):
        budget.check();case=previous.case_at(data,manifest,vehicle,old,index)
        if bool(row['truncated_preflight'])!=smoke:raise ValueError('short/formal evidence mixed')
        output=destination/f'{index:04d}-network_fixed';rolling=BASELINE/f'{index:04d}-candidate'
        t,p=audit_fixed(output,row,case,model,budget);new_transitions+=t;predictions+=p
        if not smoke:
            detail=read_json(rolling.with_suffix('.json'))
            old_transitions+=audit_execution(rolling,detail,case)['selected_h10_transitions']
            audit_rescue_execution(rolling,detail,case);audit_predictions(rolling,detail)
            predictions+=audit_forward(rolling,model,budget)
        diagnosis=diagnose_pair(output,rolling,case,formal=not smoke)
        if diagnosis!=read_json(destination/f'{index:04d}-diagnosis.json'):
            raise ValueError('window/convergence diagnosis changed')
        diagnoses.append(diagnosis)
        print(f'AUDIT #{index} new_H10={new_transitions} reused_H10={old_transitions} predictions={predictions}',flush=True)
    if report['diagnoses']!=diagnoses:raise ValueError('diagnosis report mismatch')
    calculated=summarize(report['rows'],diagnoses,formal=not smoke)
    if any(report.get(k)!=v for k,v in calculated.items()):raise ValueError('classification summary mismatch')
    return dict(status='verified',formal=not smoke,tasks=len(indices),new_fixed_episodes=len(indices),
        reused_rolling_episodes=len(indices),artifacts=artifacts,new_h10_transitions=new_transitions,
        reused_rolling_h10_transitions_reaudited=old_transitions,full_network_forward_replays=predictions,**calculated)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['prepare','precheck','smoke','run','inspect'])
    parser.add_argument('--prepared',required=True,type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase in {'prepare','precheck'}:
        config=prepare(args.prepared) if args.phase=='prepare' else precheck(args.prepared)
        result=dict(status='prepared' if args.phase=='prepare' else 'passed',tasks=len(config['indices']),
            new_fixed_episodes=21,reused_rolling_episodes=21,source_files=len(config['code']))
    else:
        if args.output is None:parser.error('--output required')
        if args.phase=='inspect':result=inspect(args.prepared,args.output)
        else:
            report=run(args.prepared,args.output,smoke=args.phase=='smoke')
            result={k:report[k] for k in ('status','formal','elapsed_seconds','metrics','classification_counts','decision')}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
