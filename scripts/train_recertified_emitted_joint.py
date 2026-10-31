"""LT42: explicit experimental teacher recertification, then one bounded paired fit."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.emitted_output_data import (
    INDICES,RECIPE,load_cohort,case_at,read_json,file_digest,file_identity)
from experiments.emitted_output_study import batch_plan,update
from scripts import diagnose_fixed_reference as previous
from scripts import compare_zero_hold_motion as teacher_study
from scripts.emitted_teacher_recertification import (
    certify,load_authorized,audit_authorization)
from scripts.recertified_emitted_fit import native_evaluate,execute_fit,inspect_fit
from scripts.train_emitted_output_joint import StageBudget
from scripts.diagnose_emitted_execution import PARENT
from scripts.prepare_recovery_balance_ab import runtime_identity
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

LT39=ROOT/'runs/diagnostics/lt39-zero-motion-rescue-20261029/results'
LT41=ROOT/'runs/diagnostics/lt41-fixed-reference-diagnosis-20261029/results'
PRECHECK=ROOT/'runs/verification/recertified-emitted-joint-20261030/preflight-v1'
PROTOCOL=dict(teacher_contract='LT39_zero_rescue_GT',frames_per_task=4,samples=32,
    stop_threshold=.5,network_horizon=320,trajectory_dt=.2,required_margin_m=.2,
    strict_position_yaw=True,certify_seconds=90,preflight_seconds=90,fit_seconds=7200,
    inspect_seconds=7200,automatic_next_stage=False,original_mpc_gate_replaced=False,
    lt40_rejection_preserved=True,validation_used_for_fitting=False)


def code_identity():
    files=[Path(__file__),ROOT/'scripts/emitted_teacher_recertification.py',ROOT/'scripts/recertified_emitted_fit.py']
    return {**previous.code_identity(),**{str(p.resolve()):file_digest(p) for p in files}}


def context():
    previous_config=previous.precheck(ROOT/'runs/prepared/fixed-reference-diagnosis-20261029-v1')
    previous.previous.verify_artifacts(LT41,previous.required_files(False))
    report=read_json(LT41/'report.json')
    calculated=previous.summarize(report['rows'],report['diagnoses'],formal=True)
    if (report['status']!='completed' or not report['formal'] or not report['decision']['diagnosis_complete']
            or read_json(LT41/'config.json')!=previous_config
            or any(report.get(k)!=v for k,v in calculated.items())):
        raise ValueError('complete frozen LT41 diagnosis required')
    audit=previous.read_shell_json(LT41.parent/'inspect.log',last_line=True)
    if (audit['status']!='verified' or not audit['formal'] or audit['tasks']!=21
            or audit['full_network_forward_replays']!=1281
            or any(audit.get(k)!=v for k,v in calculated.items())):
        raise ValueError('verified LT41 inspection required')
    for stage,name in (('run','exit.json'),('inspect','inspect-exit.json')):
        if previous.read_shell_json(LT41.parent/name)!=dict(task='LT41',stage=stage,exit_code=0):
            raise ValueError('LT41 exit identity mismatch')
    teacher_config=teacher_study.precheck(ROOT/'runs/prepared/zero-motion-rescue-20261029-v1')
    previous.previous.verify_artifacts(LT39,teacher_study.required_files(INDICES))
    teacher_report=read_json(LT39/'report.json')
    if (teacher_report['status']!='completed' or not teacher_report['formal']
            or read_json(LT39/'config.json')!=teacher_config
            or teacher_report['metrics']['teacher_fixed']['successes']!=8):
        raise ValueError('all eight complete experimental teacher successes required')
    original=read_json(PARENT/'config.json')
    return dict(schema='recertified_emitted_context_v1',code=code_identity(),runtime=runtime_identity(),
        recipe=RECIPE,protocol=PROTOCOL,indices=INDICES,sources=original['sources'],
        checkpoint=teacher_config['checkpoint'],training_inputs={k:original[k] for k in ('cohort','manifest')},
        original_training_source=original['inputs']['train'],
        original_lt34_config=file_identity(PARENT/'config.json'),
        experimental_teacher_config=file_identity(ROOT/'runs/prepared/zero-motion-rescue-20261029-v1/config.json'),
        teacher_artifacts=file_identity(LT39/'artifacts.json'),
        lt41_artifacts=file_identity(LT41/'artifacts.json'),
        lt41_inspect=file_identity(LT41.parent/'inspect.log'))


def training_cases():
    data,manifest,vehicle=load_cohort(PARENT)
    cases={index:case_at(data,manifest,vehicle,j) for j,index in enumerate(INDICES)}
    return data,cases


def preflight(teacher,output):
    source=context();samples,_=load_authorized(teacher,source)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    budget=StageBudget(PROTOCOL['preflight_seconds'])
    report=dict(status='running',context=source,teacher_artifacts=file_identity(Path(teacher)/'artifacts.json'),
        paired_updates=[],formal_training=False,weights_reusable_as_formal_origin=False)
    atomic_write_json(output/'context.json',source);atomic_write_json(output/'report.json',report)
    try:
        data,cases=training_cases();loaded=load_model_checkpoint(source['checkpoint']['path'])
        if loaded.model.stop_threshold!=.5 or loaded.model.dt!=.2 or loaded.model.max_horizon!=320:
            raise ValueError('native contract changed')
        report['baseline']=native_evaluate(loaded.model,samples,data,cases,output/'parent-native')
        seed=RECIPE['seeds'][0];plan=batch_plan(samples,seed,1)
        for arm in ('control','joint'):
            budget.check();torch.manual_seed(seed);model=deepcopy(loaded.model)
            optimizer=torch.optim.Adam(model.parameters(),lr=RECIPE['learning_rate'])
            record=update(model,optimizer,[samples[j] for j in plan[0]],seed=seed,step=1,
                joint_weight=RECIPE['joint_weight'] if arm=='joint' else 0.,inspect_gradients=arm=='joint')
            evaluation=native_evaluate(model,samples,data,cases,output/f'{arm}-post-native')
            report['paired_updates'].append(dict(arm=arm,record=record,evaluation=evaluation))
        if context()!=source:raise ValueError('source changed during preflight')
        load_authorized(teacher,source);budget.check()
        report.update(status='passed',elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(output/'report.json',report)
        atomic_write_json(output/'artifacts.json',
            {str(p.resolve()):file_digest(p) for p in output.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete',error=str(error));atomic_write_json(output/'report.json',report)
        raise


def frozen_config(teacher):
    source=context();load_authorized(teacher,source)
    expected=['context.json','report.json']+[f'{name}.{ext}' for name in
        ('parent-native','control-post-native','joint-post-native') for ext in ('json','npz')]
    previous.previous.verify_artifacts(PRECHECK,expected)
    probe=read_json(PRECHECK/'report.json')
    if (probe['status']!='passed' or probe['context']!=source
            or probe['teacher_artifacts']!=file_identity(Path(teacher)/'artifacts.json')
            or probe['formal_training'] or probe['weights_reusable_as_formal_origin']
            or [r['arm'] for r in probe['paired_updates']]!=['control','joint']
            or any(r['record']['step']!=1 for r in probe['paired_updates'])):
        raise ValueError('current-code single-batch gradient preflight required')
    return dict(schema='recertified_emitted_fit_v1',context=source,checkpoint=source['checkpoint'],
        teacher_artifacts=file_identity(Path(teacher)/'artifacts.json'),
        preflight_artifacts=file_identity(PRECHECK/'artifacts.json'),
        candidate_qualified=False,automatic_training=False)


def prepare(prepared,teacher):
    prepared=Path(prepared).resolve()
    if prepared.exists():raise FileExistsError('new prepared directory required; no overwrite')
    config=frozen_config(teacher);prepared.mkdir(parents=True,exist_ok=False)
    atomic_write_json(prepared/'config.json',config)
    return config


def precheck(prepared,teacher):
    config=read_json(Path(prepared)/'config.json')
    if config!=frozen_config(teacher):raise ValueError('frozen source/authorization/preflight drift; never edit hashes')
    return config


def fit(prepared,teacher,output):
    config=precheck(prepared,teacher)
    samples,_=load_authorized(teacher,config['context'])  # BEFORE output/optimizer creation.
    config=dict(config,config_sha256=file_digest(Path(prepared)/'config.json'))
    data,cases=training_cases()
    return execute_fit(config,samples,data,cases,output,StageBudget(PROTOCOL['fit_seconds']),
        final_check=lambda:(precheck(prepared,teacher),load_authorized(teacher,config['context'])))


def inspect(prepared,teacher,output):
    config=precheck(prepared,teacher);samples,_=load_authorized(teacher,config['context'])
    config=dict(config,config_sha256=file_digest(Path(prepared)/'config.json'))
    data,cases=training_cases()
    result=inspect_fit(config,samples,data,cases,output,StageBudget(PROTOCOL['inspect_seconds']))
    precheck(prepared,teacher)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['certify','audit-teacher','preflight','prepare','precheck','fit','inspect'])
    parser.add_argument('--teacher',required=True,type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='certify':
        if args.output is not None:parser.error('certify writes only --teacher')
        source=context();r=certify(args.teacher,source,StageBudget(PROTOCOL['certify_seconds']))
        result=dict(status=r['status'],gate=r['gate'],samples=r['selected_samples'],h10_transitions=r['scalar_h10_transitions_reaudited'])
    elif args.phase=='audit-teacher':
        result=audit_authorization(args.teacher,context(),StageBudget(PROTOCOL['certify_seconds']))
    elif args.phase=='preflight':
        if args.output is None:parser.error('--output required')
        r=preflight(args.teacher,args.output)
        result=dict(status=r['status'],paired_updates=len(r['paired_updates']),
            head_gradients=r['paired_updates'][1]['record']['joint_head_gradient_norms'],
            elapsed_seconds=r['elapsed_seconds'])
    else:
        if args.prepared is None:parser.error('--prepared required')
        if args.phase in {'prepare','precheck'}:
            r=prepare(args.prepared,args.teacher) if args.phase=='prepare' else precheck(args.prepared,args.teacher)
            result=dict(status='prepared' if args.phase=='prepare' else 'passed',tasks=8,samples=32,
                arms=4,updates=800,source_files=len(r['context']['code']))
        else:
            if args.output is None:parser.error('--output required')
            if args.phase=='inspect':result=inspect(args.prepared,args.teacher,args.output)
            else:
                r=fit(args.prepared,args.teacher,args.output);result={k:r[k] for k in ('status','gate','elapsed_seconds')}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
