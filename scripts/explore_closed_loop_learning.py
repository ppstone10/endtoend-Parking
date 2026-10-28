"""连续多对照闭环探索：prepare/precheck/smoke/run；正式任务由用户启动。"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration import (TRAINING_SEEDS,PRIMARY_ARMS,arm_samples,fit_exploration_arm,
                                               fixed_frame_metrics,paired_counts,rank_directions,contrast_report)
from experiments.behavior_exploration_data import collect_initial,collect_learner_states,evaluate_cases
from experiments.exploration_store import ExplorationBudget,ExplorationStore,exploration_lock,verify_unit
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.preflight_behavior_repair import FIT,PROBE,INPUTS,frozen_config,build_case,matrix
from scripts.prepare_recovery_balance_ab import file_digest,metadata_only
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


class IdentityMismatch(ValueError):pass


def confirmation_indices(metadata):
    forbidden=collect_metadata_ids([metadata[i] for i in FIT+PROBE]);selected=[]
    for number in (1,2,3,4,5,6,8):
        prefix=f'S{number}_';quota=15 if number in (4,8) else 6
        pool=[i for i,m in enumerate(metadata) if m['scene_name'].startswith(prefix)]
        pool.sort(key=lambda i:hashlib.sha256(('learning-explore-v1:'+metadata[i]['task_id']).encode()).hexdigest())
        taken=[]
        for index in pool:
            identities=collect_metadata_ids(metadata[index])
            if identities & forbidden:continue
            taken.append(index);forbidden.update(identities)
            if len(taken)==quota:break
        if len(taken)!=quota:raise ValueError(f'{prefix}确认集来源不足')
        selected.extend(taken)
    return selected


def make_config(*,smoke=False):
    config=frozen_config(refine_teacher=True,rolling_s4=True)
    metadata=metadata_only(INPUTS['train']);confirmation=confirmation_indices(metadata)
    config.update(kind='learning_exploration_smoke' if smoke else 'learning_exploration',
                  confirmation_indices=confirmation,training_seeds=list(TRAINING_SEEDS),
                  primary_arms=list(PRIMARY_ARMS),snapshots=[2] if smoke else [12,60,240],
                  steps=2 if smoke else 240,budget_seconds=180 if smoke else 5400,
                  optimizer_steps=2 if smoke else 240,
                  control_seeds=[0] if smoke else [0,1],
                  maximum_confirmation_directions=2,automatic_training=False,
                  purpose='诊断探索，非正式模型晋升或部署',smoke=smoke)
    config['code'][str(Path(__file__).resolve())]=file_digest(__file__)
    config['source_tasks'].update({str(i):{'task_id':metadata[i]['task_id'],'scene_name':metadata[i]['scene_name']}
                                   for i in confirmation})
    old=ROOT/'runs/diagnostics/behavior-repair-preflight-20261021-v2'
    config['expected_R0_arrays']={str(i):{'path':str(old/f'R0-{i}.npz'),'sha256':file_digest(old/f'R0-{i}.npz')}
                                  for i in FIT+PROBE}
    return config


def prepare(output,*,smoke=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('探索目录必须不存在，不覆盖已有产物')
    config=make_config(smoke=smoke);output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/'config.json',config)
    atomic_write_json(output/'identity.json',{'config_sha256':file_digest(output/'config.json')})
    print(f'PREPARED {output}; NO FORMAL ROLLOUT STARTED',flush=True)
    return config


def verify_identity(output):
    output=Path(output);config=json.loads((output/'config.json').read_text(encoding='utf-8'))
    identity=json.loads((output/'identity.json').read_text(encoding='utf-8'))
    if file_digest(output/'config.json')!=identity['config_sha256'] or make_config(smoke=config['smoke'])!=config:
        raise IdentityMismatch('配置/输入/执行代码/环境发生漂移，不在旧目录续跑')
    return config


def precheck(output):
    config=verify_identity(output);units=Path(output)/'units'
    if units.exists():
        for directory in units.iterdir():
            if directory.is_dir():verify_unit(directory)
    print(f'IDENTITY AND COMPLETED UNITS VERIFIED; smoke={config["smoke"]}; NO ROLLOUT STARTED',flush=True)


def load_candidate(path,parent):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['threshold']!=parent.stop_threshold:raise ValueError('候选阈值漂移')
    model=deepcopy(parent);model.load_state_dict(state['model_state']);model.eval()
    return model,state['optimizer_state']


def checkpoint_arm(directory,arm,seed,parent,samples,common,cases,collected,config,budget,*,initial_path=None):
    optimizer_state=None;start_step=0
    if initial_path is not None:
        parent,optimizer_state=load_candidate(initial_path,parent);start_step=config['steps']
    rows={}
    def checkpoint(step,model,optimizer,history):
        budget.check();print(f'CHECKPOINT {arm} seed={seed} step={step}',flush=True)
        name=f'step{step:04d}';point=directory/name;point.mkdir(exist_ok=False)
        torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'threshold':model.stop_threshold,'purpose':'diagnostic_not_deployment','step':step},point/'state.pt')
        fit_indices=FIT[:1] if config['smoke'] else FIT
        probes=PROBE[:1] if config['smoke'] else PROBE
        fit=evaluate_cases(fit_indices,cases,model,budget,point,'fit')
        probe=evaluate_cases(probes,cases,model,budget,point,'probe')
        row={'arm':arm,'training_seed':seed,'step':step,'fit':fit,'probe':probe,
             'fixed_metrics':fixed_frame_metrics(model,common,budget.check),
             'checkpoint':str(point/'state.pt'),'loss_history':history}
        atomic_write_json(point/'report.json',row);rows[step]=row
    fit_exploration_arm(parent,samples,seed=seed,steps=config['steps'],
                        snapshots=([config['steps']] if initial_path is not None else config['snapshots']),
                        check_budget=budget.check,checkpoint=checkpoint,
                        optimizer_state=optimizer_state,start_step=start_step)
    return {'snapshots':rows,'final':rows[start_step+config['steps']]}


def confirmation_report(directory,selected,finals,cases,parent,config,budget):
    indices=config['confirmation_indices'][:2] if config['smoke'] else config['confirmation_indices']
    baselines={control:evaluate_cases(indices,cases,parent,budget,directory,'R0',control) for control in config['control_seeds']}
    results=[]
    for direction in selected:
        name=direction['direction'];pairs=[]
        for seed in TRAINING_SEEDS:
            model,_=load_candidate(finals[name][seed]['checkpoint'],parent)
            for control in config['control_seeds']:
                rows=evaluate_cases(indices,cases,model,budget,directory,f'{name}-training{seed}',control)
                pair=paired_counts(baselines[control],rows)
                pairs.append({'training_seed':seed,'control_seed':control,'rows':rows,'paired':pair})
        results.append({'direction':name,'discovery_eligible':direction['discovery_eligible'],'pairs':pairs,
                        'robust_stage_benefit':not config['smoke'] and direction['discovery_eligible']
                                              and len(pairs)==4 and all(p['paired']['confirmation_passed'] for p in pairs)})
    report={'baseline':baselines,'directions':results,'confirmation_count':len(indices),
            'main_protocol_passed':False,'automatic_deployment':False}
    atomic_write_json(directory/'report.json',report);return report


def run(output,*,expect_smoke=False):
    output=Path(output).resolve();config=verify_identity(output)
    if config['smoke']!=expect_smoke:raise ValueError('smoke与正式目录不能混跑')
    torch.set_num_threads(1)
    progress=output/'progress.json'
    used=json.loads(progress.read_text(encoding='utf-8'))['active_elapsed_seconds'] if progress.exists() else 0.
    budget=ExplorationBudget(config['budget_seconds'],used=used,progress_path=progress)
    with exploration_lock(output):
        store=ExplorationStore(output,lambda:verify_identity(output),budget)
        report={'status':'running','smoke':config['smoke'],'branch_failures':[],
                'main_protocol_passed':False,'automatic_deployment':False,
                'formal_training_recipe_validated':False,'config_sha256':file_digest(output/'config.json')}
        def save():
            report['active_elapsed_seconds']=budget.elapsed();atomic_write_json(output/'report.json',report)
        save()
        try:
            data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train'])
            vehicle=VehicleConfig(**manifest['vehicle_model']);parent=load_model_checkpoint(INPUTS['R0']).model
            indices=FIT+PROBE+(config['confirmation_indices'][:2] if config['smoke'] else config['confirmation_indices'])
            cases={i:build_case(data,manifest,i,vehicle) for i in indices}
            expected={int(i):v['path'] for i,v in config['expected_R0_arrays'].items()}
            collected,_=store.execute('collect',lambda p:collect_initial(cases,data,parent,budget,p,expected))
            report.update(teacher_coverage=collected['coverage'],sparse_frames=len(collected['sparse']),dense_frames=len(collected['dense']))
            finals={};primary={}
            for arm in PRIMARY_ARMS:
                for seed in TRAINING_SEEDS:
                    key=f'{arm}-{seed}'
                    try:
                        payload,_=store.execute(key,lambda p:checkpoint_arm(p,arm,seed,parent,
                                  arm_samples(collected['sparse'],collected['dense'],arm),collected['dense'],
                                  cases,collected,config,budget))
                        primary[(arm,seed)]=payload;finals.setdefault(arm,{})[seed]=payload['final']
                    except IdentityMismatch:raise
                    except ValueError as exc:report['branch_failures'].append({'unit':key,'error':str(exc)})
                    save()
            aggregation=[]
            for seed in TRAINING_SEEDS:
                if ('dense_joint',seed) not in primary:continue
                initial=primary[('dense_joint',seed)]['final']['checkpoint']
                learner,_=load_candidate(initial,parent)
                aggregation_failed=False
                try:
                    extra,_=store.execute(f'aggregate-labels-{seed}',lambda p:collect_learner_states(cases,data,learner,parent,budget,p))
                except IdentityMismatch:raise
                except ValueError as exc:
                    report['branch_failures'].append({'unit':f'aggregate-labels-{seed}','error':str(exc)})
                    extra={'extra':[],'sources':[],'added_frames':0,'aggregation_had_new_labels':False}
                    aggregation_failed=True
                aggregation.append({'training_seed':seed,'added_frames':extra['added_frames'],
                                    'sources':extra['sources'],'aggregation_had_new_labels':extra['aggregation_had_new_labels']})
                for arm,samples in [('continued_dense',collected['dense']),('aggregated_dense',collected['dense']+extra['extra'])]:
                    if arm=='aggregated_dense' and aggregation_failed:continue
                    key=f'{arm}-{seed}'
                    try:
                        payload,_=store.execute(key,lambda p:checkpoint_arm(p,arm,seed,parent,samples,collected['dense'],
                                                   cases,collected,config,budget,initial_path=initial))
                        finals.setdefault(arm,{})[seed]=payload['final']
                    except IdentityMismatch:raise
                    except ValueError as exc:report['branch_failures'].append({'unit':key,'error':str(exc)})
                    save()
            report['aggregation']=aggregation
            selected=rank_directions(finals,collected['baseline_probe'][:1] if config['smoke'] else collected['baseline_probe'])
            report['selected_directions']=selected
            confirmed,_=store.execute('confirmation',lambda p:confirmation_report(p,selected,finals,cases,parent,config,budget))
            dose=[]
            for seed in TRAINING_SEEDS:
                payload=primary.get(('sparse_joint',seed))
                if payload and not config['smoke']:
                    left,right=payload['snapshots'][12],payload['snapshots'][240]
                    dose.append({'training_seed':seed,'fit':paired_counts(left['fit'],right['fit']),
                                 'probe':paired_counts(left['probe'],right['probe']),
                                 'fixed_metrics_12':left['fixed_metrics'],'fixed_metrics_240':right['fixed_metrics']})
            report.update(status='completed_with_branch_failures' if report['branch_failures'] else 'completed',
                          all_predeclared_branches_completed=not report['branch_failures'] and len(finals)==6,
                          dose_contrast=dose,contrasts=contrast_report(finals,collected['baseline_fit'],collected['baseline_probe']),
                          confirmation=confirmed,final_candidates=finals,
                          decision='smoke_only_no_benefit_claim' if config['smoke'] else
                                   'robust_direction_found_for_agent_review' if any(d['robust_stage_benefit'] for d in confirmed['directions']) else
                                   'matrix_completed_no_qualified_direction_review_contrasts')
            verify_identity(output);save();print(f'FINISHED {report["decision"]}',flush=True)
            return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                          error=str(exc),decision='incomplete_do_not_infer_benefit')
            save();raise
        finally:budget.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','precheck','smoke','run'));parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.output)
    elif args.action=='precheck':precheck(args.output)
    elif args.action=='smoke':prepare(args.output,smoke=True);run(args.output,expect_smoke=True)
    else:run(args.output)
