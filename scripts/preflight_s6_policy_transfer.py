"""用户运行的离线成功策略迁移/保持配对预检；Agent只运行precheck。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration_data import evaluate_cases
from experiments.effective_recovery_data import input_id, label_id, novel_samples
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import ExplorationBudget, exploration_lock
from experiments.retention_sampling import TRAINING_SEEDS, sampling_audit
from experiments.s6_policy_transfer import ARMS, STEPS, build_transfer_schedules, fit_transfer, validate_policy_samples
from experiments.s6_recovery_learning import batch_identity, recovery_qualification
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix
from scripts.preflight_learner_failure_teacher import verify_config, LT17, STRICT
from scripts.preflight_retention_consumption import prepare_context as retention_context
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s6_chain_learning import prepare_context as chain_context
from scripts.preflight_s6_recovery_learning import CONSUMPTION, CONSUMPTION_HASHES, FIT, QUERIES
from scripts.prepare_recovery_balance_ab import file_digest, metadata_only
from sim import VehicleConfig
from training.reporting import atomic_write_json


BANK=ROOT/'runs/diagnostics/s6-chain-learning-preflight-20261022-v2/success-teacher-bank'
BANK_HASHES={'report.json':'ed281e7b2ebb58ac6995b3895dd62a4c67cf2844f8af9c12d09fb3e559116825',
    'bank.pt':'f4fc58d3beca9d2cddb616290a8b092c3b5adfcc62ade2d5c870376b6ca13095',
    'teacher-2149.npz':'7a8dd0a32fa3bf82a46c1b5f9cc5b98980698f61f1ef31568b998783b39b7f8b'}


def prepare_context(budget):
    torch.set_num_threads(1)
    chain=chain_context(budget); old=retention_context(budget)
    verify_files({str(BANK/k):v for k,v in BANK_HASHES.items()})
    report=json.loads((BANK/'report.json').read_text(encoding='utf-8'))
    if (report['status']!='completed' or report['training_performed'] or
            report['probe_inputs_used_for_gradient']!=0 or not report['execution']['reached_with_margin'] or
            report['execution']['steps']!=539 or report['clean_nonzero_frames']!=53 or
            not report['ready_for_preservation_data_design']):
        raise ValueError('非完整严格成功策略银行')
    for name,digest in report['artifacts_sha256'].items():
        if BANK_HASHES.get(name)!=digest: raise ValueError('银行登记产物身份不符')
    bank=torch.load(BANK/'bank.pt',map_location='cpu',weights_only=False)
    teacher_sha=file_digest(bank['teacher_checkpoint'])
    if teacher_sha!='5d8e127c6cec63cefe00d143b052b4f37e28a2ef24300c5b4a38ca8ad3996cd1':
        raise ValueError('成功教师模型漂移')
    samples=bank['samples']; validate_policy_samples(samples,teacher_sha)
    identities=[dict(frame_index=s['frame_index'],input_sha256=input_id(s),label_sha256=label_id(s)) for s in samples]
    if identities!=report['sample_identities']: raise ValueError('银行样本身份与报告不符')
    windows={w['frame_index']:w for w in report['windows']}
    for s in samples:
        w=windows[s['frame_index']]
        if not w['clean'] or w['input_sha256']!=input_id(s) or w['label_sha256']!=label_id(s):
            raise ValueError('真实消费窗口未通过')
    legacy=torch.load(LT17/'data.pt',map_location='cpu',weights_only=False)
    query_path=QUERIES/'queries.pt'
    if str(query_path) not in chain['config']['frozen_files']: raise ValueError('旧三个查询缓存未冻结')
    old_queries=torch.load(query_path,map_location='cpu',weights_only=False)['samples']
    accepted,novelty=novel_samples(legacy['old']+legacy['novel']+old_queries+chain['samples'],samples)
    if len(accepted)!=53: raise ValueError('53个输入新增性未复现')
    metadata=old['metadata']; probes=old['config']['probe_indices']
    if set(bank['probe_indices_forbidden'])!=set(probes): raise ValueError('探针隔离列表漂移')
    gradient=collect_metadata_ids([metadata[2149]])
    held=collect_metadata_ids([metadata[i] for i in probes])
    held |= collect_metadata_ids(metadata_only(INPUTS['val']))|collect_metadata_ids(metadata_only(INPUTS['selection']))
    if gradient & held or 2149 in probes: raise ValueError('新增梯度来源与保留集交叉')
    cfg=dict(kind='s6_policy_transfer_preflight_v1',runtime=chain['config']['runtime'],
        code={**chain['config']['code'],**old['config']['code']},
        frozen_files={**chain['config']['frozen_files'],**old['config']['frozen_files']},
        parent_checkpoints=old['config']['parent_checkpoints'],training_seeds=list(TRAINING_SEEDS),
        arms=list(ARMS),steps=STEPS,start_step=240,end_step=268,budget_seconds=300,
        probe_indices=probes,fit_indices=list(FIT),batch_roles=['anchor']*2+['recovery']*4,
        comparison='only 56 novel-slot targets differ; 112 legacy slots unchanged per arm',
        manual_bias=False,novelty=novelty,planned_batch_identities={},sampling={},pairing={},
        automatic_deployment=False,main_protocol_passed=False)
    from experiments import s6_policy_transfer
    for path in (Path(__file__).resolve(),Path(s6_policy_transfer.__file__).resolve()): cfg['code'][str(path)]=file_digest(path)
    cfg['frozen_files'].update({str(BANK/k):v for k,v in BANK_HASHES.items()})
    cfg['frozen_files'][bank['teacher_checkpoint']]=teacher_sha
    cfg['frozen_files'].update({str(CONSUMPTION/k):v for k,v in CONSUMPTION_HASHES.items()})
    if chain['config']['parent_checkpoints']!=cfg['parent_checkpoints']: raise ValueError('未校准C240起点不同')
    plans={a:{} for a in ARMS}
    for seed in TRAINING_SEEDS:
        model,_=load_candidate(cfg['parent_checkpoints'][str(seed)],chain['parent'])
        paired,audit=build_transfer_schedules(old['schedules'][seed],samples,model,seed,budget.check)
        if audit['unique_policy_inputs']!=53 or audit['policy_slots']!=56 or audit['self_label_changes']!=53:
            raise ValueError('新增消费/目标干预不足')
        cfg['pairing'][seed]=audit
        for arm in ARMS:
            plans[arm][seed]=paired[arm]
            cfg['planned_batch_identities'].setdefault(arm,{})[seed]=[batch_identity(b) for b in paired[arm]]
            cfg['sampling'].setdefault(arm,{})[seed]=sampling_audit(paired[arm],metadata)
    verify_config(cfg,budget)
    consumption=json.loads((CONSUMPTION/'report.json').read_text(encoding='utf-8'))
    strict=json.loads((STRICT/'report.json').read_text(encoding='utf-8'))
    return dict(config=cfg,plans=plans,parent=chain['parent'],previous=old['previous'],consumption=consumption,
        parent_fit={s:[r for r in strict['source_rows'] if r['seed']==s] for s in TRAINING_SEEDS})


def precheck(prepared=None):
    context=prepare_context(PreflightBudget(60))
    if prepared is not None:
        prepared=Path(prepared).resolve()
        if prepared.exists(): raise FileExistsError(prepared)
        prepared.parent.mkdir(parents=True,exist_ok=True)
        atomic_write_json(prepared,context['config'])
    print('TRANSFER INPUTS/PAIRED 28-STEP PLANS VERIFIED; ZERO UPDATES/ROLLOUTS',flush=True)
    return context


def run(config_path,output):
    config_path=Path(config_path).resolve(); expected=json.loads(config_path.read_text(encoding='utf-8'))
    budget=ExplorationBudget(300); started=time.perf_counter()
    verify_config(expected,budget)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',stage='identity',completed_arms={},training_performed=False,actual_updates={},
        allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-started; atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget)
            cfg=json.loads(json.dumps(context['config']))
            if cfg!=expected: raise ValueError('准备配置/排程与当前上下文不同，禁止训练')
            atomic_write_json(output/'config.json',cfg); report['config_sha256']=file_digest(output/'config.json')
            report['prepared_config_sha256']=file_digest(config_path); save()
            data=DatasetGenerator.load(INPUTS['train']); manifest=matrix.load_dataset_manifest(INPUTS['train'])
            vehicle=VehicleConfig(**manifest['vehicle_model'])
            cases={i:build_case(data,manifest,i,vehicle) for i in sorted(set(cfg['probe_indices']+cfg['fit_indices']))}
            for arm in ARMS:
                for seed in TRAINING_SEEDS:
                    budget.check(); directory=output/f'{arm}-{seed}'; directory.mkdir()
                    report.update(stage=f'fit-{arm}-{seed}'); save()
                    print(f'FIT {arm} seed={seed} updates={STEPS}',flush=True)
                    model,state=load_candidate(cfg['parent_checkpoints'][str(seed)],context['parent'])
                    threshold=model.stop_threshold
                    def on_update(count):
                        report['training_performed']=True
                        report['actual_updates'][f'{arm}-{seed}']=count
                    model,opt,history,presented=fit_transfer(model,state,context['plans'][arm][seed],
                        cfg['planned_batch_identities'][arm][str(seed)],seed,budget.check,on_update)
                    if model.stop_threshold!=threshold: raise ValueError('学生阈值漂移')
                    torch.save(dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),step=268,
                        threshold=model.stop_threshold,purpose='offline_policy_transfer_diagnostic_not_deployment'),directory/'state.pt')
                    point=dict(seed=seed,arm=arm,step=268,threshold=threshold,loss_history=history,
                        presented_batch_identities=presented,checkpoint=str(directory/'state.pt'))
                    atomic_write_json(directory/'report.json',point)
                    report['training_performed']=True; report.update(stage=f'evaluate-{arm}-{seed}'); save()
                    point['probe']=evaluate_cases(cfg['probe_indices'],cases,model,budget,directory,'probe')
                    point['fit']=evaluate_cases(cfg['fit_indices'],cases,model,budget,directory,'fit')
                    atomic_write_json(directory/'report.json',point)
                    report['completed_arms'].setdefault(arm,{})[seed]=point; save()
            previous=context['previous']; int_keys=lambda d:{int(k):v for k,v in d.items()}
            rows=lambda arm,field:{s:p[field] for s,p in report['completed_arms'][arm].items()}
            verdict=recovery_qualification(previous['r0_probe'],int_keys(previous['parent_probe']),
                {int(s):p['probe'] for s,p in previous['completed_arms']['flat_old'].items()},
                {int(s):p['probe'] for s,p in context['consumption']['completed_seeds'].items()},
                rows('self_targets','probe'),rows('policy_targets','probe'),context['parent_fit'],rows('policy_targets','fit'))
            verify_config(cfg,budget); budget.check()
            report.update(status='completed',stage='decision',qualification=verdict,
                allowed_to_prepare_training_plan=verdict['passed'],
                decision='native_transfer_signal_prepare_separate_plan' if verdict['passed'] else 'no_qualified_native_transfer_do_not_expand')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p.name not in ('run.lock','console.log') and p!=output/'report.json'}
            save(); print('FINISHED',report['decision'],flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),
                allowed_to_prepare_training_plan=False,decision='incomplete_do_not_infer_transfer')
            save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precheck',action='store_true'); parser.add_argument('--prepare',type=Path)
    parser.add_argument('--config',type=Path); parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.precheck:
        if args.config or args.output: parser.error('precheck只支持可选prepare，不运行正式任务')
        precheck(args.prepare)
    elif args.config is None or args.output is None or args.prepare is not None:
        parser.error('正式执行需要config和不存在的新output')
    else: run(args.config,args.output)
