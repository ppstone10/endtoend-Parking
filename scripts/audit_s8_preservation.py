"""180秒零更新：LT18实际S8目标、局部梯度和训练来源成功教师可用性。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration_data import run_case, save_record
from experiments.behavior_preservation import behavior_loss
from experiments.effective_recovery_data import input_id, label_id
from experiments.executable_behavior import PreflightBudget, prepare_supervision_frames
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.recovery_execution import record_arrays
from experiments.retention_quality import audit_source_windows
from experiments.s6_recovery_learning import batch_identity
from experiments.s8_preservation_audit import comparison_metrics, gradient_relation, temporal_clean_queries
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix, result_row
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_s6_policy_transfer import prepare_context as transfer_context
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest, metadata_only
from sim import VehicleConfig
from training.reporting import atomic_write_json


PREVIOUS=ROOT/'runs/diagnostics/s6-policy-transfer-preflight-20261022-v1'
HASHES={'config.json':'cd1d25e0cb8bbbcf191a7ef0f7b95eedb22a62ee420936547a9e8f981627e2c1',
    'report.json':'6e463aa2732d2780f07b04106ed1b026637e307dc6e9e792dab0116fed962b3e'}


def prepare_context(budget):
    verify_files({str(PREVIOUS/k):v for k,v in HASHES.items()})
    prior=json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'))
    report=json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    verify_config(prior,budget)
    if report['status']!='completed' or report['qualification']['passed']: raise ValueError('不是完整未放行LT18')
    for name,digest in report['artifacts_sha256'].items():
        budget.check();verify_files({str(artifact_path(PREVIOUS,name)):digest})
    context=transfer_context(budget)
    if json.loads(json.dumps(context['config']))!=prior: raise ValueError('LT18配置无法精确重建')
    metadata=metadata_only(INPUTS['train']);sources={};visits={};gradient_batches={}
    for seed in prior['training_seeds']:
        schedule=context['plans']['policy_targets'][seed]
        if [batch_identity(b) for b in schedule]!=report['completed_arms']['policy_targets'][str(seed)]['presented_batch_identities']:
            raise ValueError('实际训练批次不符')
        selected=[s for b in schedule for s in b[2:4] if metadata[s['source_index']]['scene_name'].startswith('S8_')]
        sources[seed]=sorted({s['source_index'] for s in selected});visits[seed]=selected
        gradient_batches[seed]=next(i for i,b in enumerate(schedule) if any(metadata[s['source_index']]['scene_name'].startswith('S8_') for s in b[2:4]))
    declared=collect_metadata_ids([metadata[i] for values in sources.values() for i in values])
    forbidden=collect_metadata_ids(metadata_only(INPUTS['val']))|collect_metadata_ids(metadata_only(INPUTS['selection']))
    forbidden |= collect_metadata_ids([metadata[i] for i in prior['probe_indices']])
    if not declared or declared&forbidden: raise ValueError('审计来源与保留集交叉')
    cfg=dict(kind='s8_preservation_audit_v1',budget_seconds=180,training_updates=0,runtime=prior['runtime'],
        code=dict(prior['code']),frozen_files=dict(prior['frozen_files']),sources=sources,
        fixed_gradient_batches=gradient_batches,probe_labels_collected=False,automatic_deployment=False,
        main_protocol_passed=False,allowed_to_prepare_training_plan=False)
    cfg['frozen_files'].update({str(PREVIOUS/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(PREVIOUS,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s8_preservation_audit
    for p in (Path(__file__).resolve(),Path(s8_preservation_audit.__file__).resolve()):cfg['code'][str(p)]=file_digest(p)
    verify_config(cfg,budget)
    context.update(config=cfg,prior=prior,previous_report=report,metadata=metadata,visits=visits)
    return context


@torch.no_grad()
def label_comparisons(context,seed,budget):
    cfg=context['prior'];parent,_=load_candidate(cfg['parent_checkpoints'][str(seed)],context['parent'])
    models={'C240':parent}
    for arm in ('self_targets','policy_targets'):
        models[arm],_=load_candidate(PREVIOUS/f'{arm}-{seed}/state.pt',context['parent'])
    visits=context['visits'][seed];counts=Counter(input_id(s) for s in visits)
    unique={input_id(s):s for s in visits};rows=[]
    for identity,s in unique.items():
        budget.check()
        base=parent.forward_with_stop(s['bev'][None],s['goal'][None],s['state'][None])
        target=dict(s,points=base.points[0].detach(),stop=base.stop_logits[0].sigmoid().detach())
        row=dict(source_index=s['source_index'],frame_index=s['frame_index'],input_sha256=identity,
            label_sha256=label_id(s),visits=counts[identity],role=s['role'],origin=s.get('origin','not_recorded'),models={})
        for name,model in models.items():
            out=base if name=='C240' else model.forward_with_stop(s['bev'][None],s['goal'][None],s['state'][None])
            row['models'][name]=dict(to_old=comparison_metrics(out,s,model.stop_threshold),
                to_C240=comparison_metrics(out,target,model.stop_threshold))
        rows.append(row)
    aggregate={}
    for name in models:
        aggregate[name]={}
        for target in ('to_old','to_C240'):
            aggregate[name][target]={key:sum(r['visits']*r['models'][name][target][key] for r in rows)/len(visits)
                for key in ('pose_loss','stop_kl','total_loss','mask_beyond_prediction')}
    return dict(visits=len(visits),unique_inputs=len(unique),rows=rows,weighted_metrics=aggregate)


def local_gradient(context,seed,budget):
    budget.check();index=context['config']['fixed_gradient_batches'][seed]
    batch=context['plans']['policy_targets'][seed][index]
    model,_=load_candidate(context['prior']['parent_checkpoints'][str(seed)],context['parent'])
    model.eval();stack=lambda key:torch.stack([s[key] for s in batch])
    out=model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
    s8=[i for i in (2,3) if context['metadata'][batch[i]['source_index']]['scene_name'].startswith('S8_')]
    def loss(indices):
        return behavior_loss(out.points[indices],out.stop_logits[indices],stack('points')[indices],
            stack('stop')[indices],stack('mask')[indices],['recovery']*len(indices))
    result=gradient_relation(loss(s8)*(.25*len(s8)/2),loss([4,5])*.25,model.parameters())
    return dict(batch_index=index,S8_slots=s8,batch_identity=batch_identity(batch),
        actual_contribution_weights=[.25*len(s8)/2,.25],relation=result,
        interpretation='single_initial_batch_only_not_full_training_causality')


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();budget=PreflightBudget(180)
    report=dict(status='running',stage='identity',training_updates=0,source_rows=[],label_comparisons={},gradients={},
        allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-started;atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget);cfg=context['config']
            atomic_write_json(output/'config.json',cfg);report['config_sha256']=file_digest(output/'config.json')
            for seed in context['prior']['training_seeds']:
                report['stage']=f'labels-{seed}';save()
                report['label_comparisons'][seed]=label_comparisons(context,seed,budget)
                report['gradients'][seed]=local_gradient(context,seed,budget);save()
            data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train'])
            vehicle=VehicleConfig(**manifest['vehicle_model'])
            cases={i:build_case(data,manifest,i,vehicle) for values in cfg['sources'].values() for i in values}
            bank={};eligible=0
            for seed in context['prior']['training_seeds']:
                model,_=load_candidate(context['prior']['parent_checkpoints'][str(seed)],context['parent'])
                bank[seed]={}
                for index in cfg['sources'][seed]:
                    budget.check();report['stage']=f'source-{seed}-{index}';save()
                    print('COLLECT C240 TRAINING SOURCE',seed,index,flush=True)
                    case=cases[index];result,engine,frames=run_case(case,model,budget)
                    save_record(output,f'C240-{seed}-{index}',result)
                    row=dict(seed=seed,**result_row(case,result,engine))
                    if result.success:
                        samples=prepare_supervision_frames(frames,model,'anchor',index,frame_limit=None)
                        states,_=record_arrays(result);windows=audit_source_windows(samples,states,case,budget.check)
                        queries,gate=temporal_clean_queries(samples,windows,source_success=True)
                        for q in queries:
                            q.update(origin='validated_C240_success',query_policy=f'C240-{seed}',
                                teacher_checkpoint_sha256=file_digest(context['prior']['parent_checkpoints'][str(seed)]))
                        row.update(windows=windows,query_gate=gate)
                        if gate['passed']:bank[seed][index]=queries;eligible+=1
                    else:row['query_gate']=dict(passed=False,reason='full_C240_source_failed')
                    report['source_rows'].append(row);save()
            torch.save(dict(purpose='training_sources_only_potential_retention_not_training_authorization',
                queries_by_seed=bank),output/'queries.pt')
            verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',complete_sources=len(report['source_rows']),
                eligible_source_policies=eligible,decision='audit_only_choose_separate_retention_experiment' if eligible
                else 'no_verified_C240_retention_source_do_not_train')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p.name not in ('report.json','run.lock','console.log')}
            save();print('FINISHED',report['decision'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def precheck():
    context=prepare_context(PreflightBudget(60));print('S8 AUDIT INPUTS/FIXED SOURCES VERIFIED; NO GRADIENTS/ROLLOUTS',flush=True)
    return context


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precheck',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.precheck:
        if a.output:p.error('precheck不写output')
        precheck()
    elif a.output is None:p.error('需要不存在的新output')
    else:run(a.output)
