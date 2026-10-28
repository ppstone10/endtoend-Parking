"""150秒双学生短检：原LT18恢复目标加当前成功C240行为保持，禁止扫权重。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration import first_stop,paired_counts
from experiments.behavior_exploration_data import evaluate_cases
from experiments.effective_recovery_data import input_id,label_id
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path,exploration_lock
from experiments.s6_recovery_learning import batch_identity,recovery_qualification
from experiments.s8_preservation_learning import PRESERVATION_WEIGHT,preservation_schedule,fit_preservation
from scripts.audit_s8_preservation import prepare_context as audit_context,PREVIOUS
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.reporting import atomic_write_json


AUDIT=ROOT/'runs/diagnostics/s8-preservation-audit-20261022-v1'
HASHES={'config.json':'5609fbc81abdf412e65e162edca0f5c2cfcab3ef11d6cd30c152874093bdf77a',
    'report.json':'982d288c67eb515d3a829f3bcba38921503440e73f3eae77e7af90dbebc93b6c'}


def prepare_context(budget):
    verify_files({str(AUDIT/k):v for k,v in HASHES.items()})
    report=json.loads((AUDIT/'report.json').read_text(encoding='utf-8'))
    old=json.loads((AUDIT/'config.json').read_text(encoding='utf-8'))
    if report['status']!='completed' or report['complete_sources']!=13 or report['training_updates']!=0:
        raise ValueError('非完整无更新S8审计')
    verify_config(old,budget)
    for name,digest in report['artifacts_sha256'].items():verify_files({str(artifact_path(AUDIT,name)):digest})
    context=audit_context(budget)
    if json.loads(json.dumps(context['config']))!=old:raise ValueError('旧S8审计配置无法精确重建')
    bank=torch.load(AUDIT/'queries.pt',map_location='cpu',weights_only=False)['queries_by_seed']
    cfg=deepcopy(context['config']);cfg.update(kind='s8_preservation_learning_preflight_v1',budget_seconds=150,
        steps=28,start_step=240,end_step=268,preservation_weight=PRESERVATION_WEIGHT,
        comparison='unchanged LT18 primary objective plus .25 verified C240 preservation; cached self/policy controls',
        training_updates=28,planned_primary={},planned_preservation={},teacher_identities={})
    cfg['frozen_files'].update({str(AUDIT/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(AUDIT,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s8_preservation_learning
    for p in (Path(__file__).resolve(),Path(s8_preservation_learning.__file__).resolve()):cfg['code'][str(p)]=file_digest(p)
    schedules={}
    for seed in context['prior']['training_seeds']:
        model,_=load_candidate(context['prior']['parent_checkpoints'][str(seed)],context['parent'])
        eligible={row['dataset_index']:row for row in report['source_rows'] if row['seed']==seed and row['query_gate']['passed']}
        if set(bank[seed])!=set(eligible) or len(eligible)!=(3 if seed==20261020 else 6):raise ValueError('成功来源集合漂移')
        for index,samples in bank[seed].items():
            row=eligible[index];windows={w['frame_index']:w for w in row['windows']}
            if not row['success'] or [s['frame_index'] for s in samples]!=row['query_gate']['selected_frames']:
                raise ValueError('不是预选完整成功来源保持帧')
            for s in samples:
                budget.check();w=windows[s['frame_index']]
                if (not w['clean'] or w['input_sha256']!=input_id(s) or w['label_sha256']!=label_id(s) or
                        s['teacher_checkpoint_sha256']!=file_digest(context['prior']['parent_checkpoints'][str(seed)])):
                    raise ValueError('成功保持标签/窗口身份漂移')
                with torch.no_grad():out=model.forward_with_stop(s['bev'][None],s['goal'][None],s['state'][None])
                if (not torch.equal(out.points[0],s['points']) or not torch.equal(out.stop_logits[0].sigmoid(),s['stop']) or
                        int(torch.nonzero(s['mask']).max())>=first_stop(s['stop'],model.stop_threshold)):
                    raise ValueError('保持教师输出/首停支持未复现')
        schedules[seed]=preservation_schedule(bank[seed],seed)
        cfg['planned_primary'][seed]=[batch_identity(b) for b in context['plans']['policy_targets'][seed]]
        cfg['planned_preservation'][seed]=[batch_identity(b) for b in schedules[seed]]
        cfg['teacher_identities'][seed]=[batch_identity(v) for _,v in sorted(bank[seed].items())]
    verify_config(cfg,budget);context.update(config=cfg,preservation_schedules=schedules)
    return context


def qualification(context,points):
    previous=context['previous'];intkeys=lambda d:{int(k):v for k,v in d.items()}
    cached=context['previous_report']['completed_arms']
    rows={s:p['probe'] for s,p in points.items()}
    verdict=recovery_qualification(previous['r0_probe'],intkeys(previous['parent_probe']),
        {int(s):p['probe'] for s,p in previous['completed_arms']['flat_old'].items()},
        {int(s):p['probe'] for s,p in context['consumption']['completed_seeds'].items()},
        {int(s):p['probe'] for s,p in cached['policy_targets'].items()},rows,context['parent_fit'],
        {s:p['fit'] for s,p in points.items()})
    pairs=[dict(seed=s,**paired_counts(cached['self_targets'][str(s)]['probe'],p['probe'])) for s,p in points.items()]
    gate=(len(pairs)==2 and all(p['success_delta']>=0 and p['collision_delta']<=0 and not p['lost_success_indices'] for p in pairs)
        and sum(p['success_delta'] for p in pairs)>=1)
    return dict(passed=verdict['passed'] and gate,policy_control_gate=verdict,self_control_pairs=pairs,
        self_control_passed=gate,main_protocol_passed=False,automatic_deployment=False)


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    budget=PreflightBudget(150);start=time.perf_counter()
    report=dict(status='running',stage='identity',completed_seeds={},actual_updates={},training_performed=False,
        allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-start;atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget);cfg=context['config'];atomic_write_json(output/'config.json',cfg)
            report['config_sha256']=file_digest(output/'config.json');save()
            data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train']);vehicle=VehicleConfig(**manifest['vehicle_model'])
            indices=context['prior']['probe_indices']+context['prior']['fit_indices']
            cases={i:build_case(data,manifest,i,vehicle) for i in indices}
            for seed in context['prior']['training_seeds']:
                budget.check();directory=output/str(seed);directory.mkdir()
                report['stage']=f'fit-{seed}';save();print('FIT PRESERVATION',seed,'updates=28',flush=True)
                parent,state=load_candidate(context['prior']['parent_checkpoints'][str(seed)],context['parent'])
                def on_update(count):report['actual_updates'][seed]=count;report['training_performed']=True
                model,opt,history,presented=fit_preservation(parent,state,context['plans']['policy_targets'][seed],
                    context['preservation_schedules'][seed],cfg['planned_primary'][seed],cfg['planned_preservation'][seed],seed,budget.check,on_update)
                if model.stop_threshold!=parent.stop_threshold:raise ValueError('学生停止阈值漂移')
                torch.save(dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),step=268,
                    threshold=model.stop_threshold,purpose='bounded_preservation_diagnostic_not_deployment'),directory/'state.pt')
                point=dict(seed=seed,step=268,loss_history=history,presented_batches=presented,checkpoint=str(directory/'state.pt'))
                atomic_write_json(directory/'report.json',point);report['stage']=f'evaluate-{seed}';save()
                point['probe']=evaluate_cases(context['prior']['probe_indices'],cases,model,budget,directory,'probe')
                point['fit']=evaluate_cases(context['prior']['fit_indices'],cases,model,budget,directory,'fit')
                atomic_write_json(directory/'report.json',point);report['completed_seeds'][seed]=point;save()
            verdict=qualification(context,report['completed_seeds']);verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',qualification=verdict,allowed_to_prepare_training_plan=verdict['passed'],
                decision='preservation_signal_prepare_separate_manual_plan' if verdict['passed'] else 'no_qualified_preservation_signal_do_not_expand')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p.name not in ('report.json','run.lock','console.log')}
            save();print('FINISHED',report['decision'],report['elapsed_seconds'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


def precheck():
    context=prepare_context(PreflightBudget(60));print('PRESERVATION INPUTS/28-STEP PLANS VERIFIED; NO TRAINING/ROLLOUT',flush=True)
    return context


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precheck',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.precheck:
        if a.output:p.error('precheck不写output')
        precheck()
    elif a.output is None:p.error('需要不存在的新output')
    else:run(a.output)
