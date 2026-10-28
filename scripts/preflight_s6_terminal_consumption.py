"""原S6首窗strict控制与已执行完整终段mask对照；120秒、两新模型。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from dataset import DatasetGenerator
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.retention_sampling import TRAINING_SEEDS, sampling_audit
from experiments.s6_recovery_learning import batch_identity, recovery_qualification
from experiments.s6_terminal_consumption import terminal_schedule, terminal_qualification
from interfaces import VehicleState
from runtime.sources import NetworkSource
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s6_recovery_learning import prepare_context as prepare_learning_context, fit_recovery_point, FIT
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.reporting import atomic_write_json


PREVIOUS=ROOT/'runs/diagnostics/s6-recovery-learning-preflight-20261022-v1'
HASHES={
    'config.json':'eb6e10b9f2f28bd7e59593252518ae89521393897f91f57a40cdb5cc0285a67d',
    'report.json':'22653712bcdf5ed7501022988016a8c9be5a0b440d514b780da3e35feb6ef665',
    'independent-transfer/audit.json':'34d823f064fc9ade107ad56872d1e332e3aae86086719c4e4981853a88114018',
    'independent-transfer/terminal-teacher-audit.json':'164bb6a51dae916f42b767068debfe44bf44dab788f31c478ebe9ab5c321413e',
    'independent-transfer/terminal-teacher-fixed.npz':'12f598e640213f7e7a7caa435324c211fa82ba1ae4f869f16678d62388f4a751',
}
ARM='terminal_consumed'


def prepare_context(budget):
    torch.set_num_threads(1)
    verify_files({str(PREVIOUS/k):v for k,v in HASHES.items()})
    old=json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'))
    report=json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    audit=json.loads((PREVIOUS/'independent-transfer/audit.json').read_text(encoding='utf-8'))
    proof=json.loads((PREVIOUS/'independent-transfer/terminal-teacher-audit.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed' or report['allowed_to_prepare_training_plan'] or audit['status'] != 'verified':
        raise ValueError('不是完整已复核且未放行的首窗对照')
    files=dict(old['frozen_files']); files.update({str(PREVIOUS/k):v for k,v in HASHES.items()})
    files.update({str(artifact_path(PREVIOUS,k)):v for k,v in report['artifacts_sha256'].items()})
    files.update({str(artifact_path(PREVIOUS/'independent-transfer',k)):v for k,v in audit['artifacts_sha256'].items()})
    config=dict(old); config['frozen_files']=files; config['code']=dict(old['code'])
    verify_config(config,budget)
    if (proof['status'] != 'verified' or not proof['reference_connected_safe'] or
            not proof['execution']['reached_with_margin'] or proof['execution']['collision'] or
            proof['execution']['margin_violation_steps'] or proof['execution']['steps'] != 31 or
            proof['source_index'] != 2149 or proof['context_step'] != 400 or
            proof['actual_consumed_raw_indices'] != list(range(19))):
        raise ValueError('完整终段缺原严格执行证据')
    context=prepare_learning_context(budget)
    # 整份原配置必须重现，不只比较新干预所在的一个样本。
    if json.loads(json.dumps(context['config'])) != old:
        raise ValueError('首窗对照原排程/身份未逐字段复现')
    plans={}; sampling={}; identities={}; mask_audits={}
    with np.load(PREVIOUS/'independent-transfer/terminal-teacher-fixed.npz') as record:
        reference=record['reference'].copy(); union=record['union_consumed_mask'].copy()
        state=VehicleState.from_array(record['context_state'])
        if len(record['states']) != 31 or len(record['commands']) != 31:
            raise ValueError('终段执行记录分母不符')
    for seed in TRAINING_SEEDS:
        budget.check(); base=context['plans']['strict_targets'][seed]
        expected=old['planned_batch_identities']['strict_targets'][str(seed)]
        if ([batch_identity(batch) for batch in base] != expected or
                report['completed_arms']['strict_targets'][str(seed)]['presented_batch_identities'] != expected):
            raise ValueError('缓存控制的实际梯度批次未复现')
        sample=base[16][2]
        restored=NetworkSource._to_global(np.asarray(sample['points'])[:19],state)
        if (not np.array_equal(restored,reference) or not np.array_equal(state.to_array()[:3],sample['global_pose']) or
                not np.array_equal(state.to_array()[3:].astype(np.float32),np.asarray(sample['state']))):
            raise ValueError('完整参考不属于同一实际查询/float32标签')
        plans[seed],mask_audits[seed]=terminal_schedule(base,union)
        sampling[seed]=sampling_audit(plans[seed],context['metadata'])
        identities[seed]=[batch_identity(batch) for batch in plans[seed]]
    from experiments import s6_terminal_consumption
    for path in (Path(s6_terminal_consumption.__file__).resolve(),Path(__file__).resolve()):
        config['code'][str(path)]=file_digest(path)
    config.update(kind='s6_terminal_consumption_preflight_v1',budget_seconds=120,arms=[ARM],
        comparison='same query/points/stop/order/24 updates; only step400 actual consumed mask 0..10 versus 0..18',
        sampling={ARM:sampling},planned_batch_identities={ARM:identities},mask_audits=mask_audits,
        reused_window_control=str(PREVIOUS),terminal_execution_proof=proof)
    context.update(config=config,plans={ARM:plans},window_report=report)
    budget.check(); return context


def precheck():
    context=prepare_context(PreflightBudget(120))
    print('ORIGINAL PLAN AND EXECUTED TERMINAL MASK VERIFIED; NO TRAINING OR ROLLOUT',flush=True)
    return context


def run(output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); budget=PreflightBudget(120)
    report=dict(status='running',stage='identity_and_exact_plan',completed_seeds={},training_started=False,
        training_performed=False,allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)

    def save():
        report['elapsed_seconds']=time.perf_counter()-started
        atomic_write_json(output/'report.json',report)

    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget); config=context['config']; old=context['window_report']
            atomic_write_json(output/'config.json',config)
            report.update(config_sha256=file_digest(output/'config.json'),mask_audits=config['mask_audits'],
                reused_window_control=old['completed_arms']['strict_targets'],
                reused_self_control=old['completed_arms']['self_targets']); save()
            data=DatasetGenerator.load(INPUTS['train']); manifest=matrix.load_dataset_manifest(INPUTS['train'])
            vehicle=VehicleConfig(**manifest['vehicle_model'])
            context['cases']={i:build_case(data,manifest,i,vehicle) for i in sorted(set(config['probe_indices']+list(FIT)))}
            for seed in TRAINING_SEEDS:
                budget.check(); directory=output/f'{ARM}-{seed}'; directory.mkdir()
                report.update(stage=f'fit-{seed}',training_started=True); save()
                print(f'FIT {ARM} seed={seed} updates=24',flush=True)
                report['completed_seeds'][seed]=fit_recovery_point(context,ARM,seed,budget,directory)
                report['training_performed']=True; save()
            cast=lambda values:{int(k):v for k,v in values.items()}
            candidates={s:p['probe'] for s,p in report['completed_seeds'].items()}
            previous=context['previous']
            original=recovery_qualification(previous['r0_probe'],cast(old['reused_parent_probe']),
                {int(s):p['probe'] for s,p in previous['completed_arms']['flat_old'].items()},
                cast(old['reused_consumption_probe']),
                {int(s):p['probe'] for s,p in old['completed_arms']['self_targets'].items()},
                candidates,cast(old['reused_parent_fit']),{s:p['fit'] for s,p in report['completed_seeds'].items()})
            verdict=terminal_qualification(original,
                {int(s):p['probe'] for s,p in old['completed_arms']['strict_targets'].items()},candidates)
            verify_config(config,budget)
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p != output/'report.json' and p.name not in ('run.lock','console.log')}
            budget.check()
            report.update(status='completed',stage='decision',qualification=verdict,
                allowed_to_prepare_training_plan=verdict['passed'],decision='terminal_consumption_signal_prepare_manual_plan'
                if verdict['passed'] else 'no_qualified_terminal_consumption_signal_do_not_expand')
            save(); print(f'FINISHED {report["decision"]}',flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),
                allowed_to_prepare_training_plan=False,decision='incomplete_do_not_infer_terminal_benefit')
            save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path); parser.add_argument('--precheck',action='store_true')
    args=parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output目录')
    else: run(args.output)
