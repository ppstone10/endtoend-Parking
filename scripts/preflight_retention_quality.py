"""固定两S3槽位的干净/风险换样双臂；不续接264点、不扩大短检预算。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration_data import evaluate_cases
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.retention_quality import audit_source_windows, build_quality_schedules, quality_qualification
from experiments.retention_sampling import fit_continuation, sampling_audit, TRAINING_SEEDS
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix
from scripts.preflight_retention_consumption import LT17, prepare_context as prepare_consumption_context
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


CONSUMPTION = ROOT / 'runs/diagnostics/retention-consumption-preflight-20261021-v1'
CONSUMPTION_HASHES = {
    'config.json': 'b69652b11de886cdc7997f588db93663d677482788f6991c1af1c777799f704c',
    'report.json': '54f88af74e79c20fe9fd6e5dbd4b1759a861a99b821964e3c0e029804dac7c52',
}
ARMS = ('clean_windows', 'risky_windows')


def prepare_context(budget):
    budget.check()
    verify_files({str(CONSUMPTION / k): v for k, v in CONSUMPTION_HASHES.items()})
    old_config = json.loads((CONSUMPTION / 'config.json').read_text(encoding='utf-8'))
    previous = json.loads((CONSUMPTION / 'report.json').read_text(encoding='utf-8'))
    verify_files(old_config['code'])
    if previous['status'] != 'completed' or set(previous['completed_seeds']) != set(map(str, TRAINING_SEEDS)):
        raise ValueError('消费基线不完整')
    for relative, digest in previous['artifacts_sha256'].items():
        budget.check()
        if file_digest(artifact_path(CONSUMPTION, relative)) != digest:
            raise ValueError(f'消费基线产物漂移：{relative}')
    original = prepare_consumption_context(budget)
    for seed in TRAINING_SEEDS:
        if sampling_audit(original['schedules'][seed], original['metadata']) != old_config['sampling'][str(seed)]:
            raise ValueError('消费基线批次未精确复现')
    # The upstream check verifies data.pt before this trusted local deserialization.
    bank = torch.load(LT17 / 'data.pt', weights_only=False, map_location='cpu')
    novel = bank['novel']
    data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    source = build_case(data, manifest, 1196, vehicle)
    with np.load(LT17 / 'collect-1196/R0.npz') as archive:
        quality = audit_source_windows([s for s in novel if s['source_index'] == 1196],
                                       archive['states'], source, budget.check)
    plans = {arm: {} for arm in ARMS}; audits = {}; samplings = {arm: {} for arm in ARMS}
    for seed in TRAINING_SEEDS:
        budget.check()
        by_arm, audits[seed] = build_quality_schedules(original['schedules'][seed], novel, quality)
        for arm in ARMS:
            plans[arm][seed] = by_arm[arm]
            samplings[arm][seed] = sampling_audit(by_arm[arm], original['metadata'])
    code = dict(old_config['code'])
    from experiments import retention_quality
    for path in (Path(retention_quality.__file__).resolve(), Path(__file__).resolve()):
        code[str(path)] = file_digest(path)
    frozen = dict(original['config']['frozen_files'])
    frozen.update({str(CONSUMPTION / k): v for k, v in CONSUMPTION_HASHES.items()})
    frozen.update({str(artifact_path(CONSUMPTION, k)): v for k, v in previous['artifacts_sha256'].items()})
    config = dict(kind='retention_quality_preflight_v1', budget_seconds=180, code=code, frozen_files=frozen,
                  runtime=old_config['runtime'], parent_checkpoints=old_config['parent_checkpoints'],
                  start_step=240, steps=24, training_seeds=list(TRAINING_SEEDS), arms=list(ARMS),
                  probe_indices=old_config['probe_indices'], fit_indices=old_config['fit_indices'],
                  source_quality_rows=quality, pairing_audits=audits, sampling=samplings,
                  reused_controls='LT17 R0/C240/flat_old and consumption-v1; no new control training',
                  automatic_deployment=False, main_protocol_passed=False)
    cases = {i: build_case(data, manifest, i, vehicle) for i in config['probe_indices'] + config['fit_indices']}
    budget.check()
    return dict(config=config, plans=plans, metadata=original['metadata'], cases=cases,
                lt17=original['previous'], consumption=previous)


def precheck():
    context = prepare_context(PreflightBudget(180))
    print('QUALITY PAIRING VERIFIED; NO TRAINING OR ROLLOUT STARTED', flush=True)
    return context


def fit_point(context, arm, seed, parent, budget, directory):
    config = context['config']; schedule = context['plans'][arm][seed]
    model, optimizer_state = load_candidate(config['parent_checkpoints'][str(seed)], parent)
    presented = []; planned = config['sampling'][arm][seed]['batches']

    def check_training_batch():
        budget.check(); position = len(presented)
        if position >= 24:
            raise ValueError('超过24更新')
        actual = sampling_audit([schedule[position]], context['metadata'])['batches'][0]
        if actual != planned[position]:
            raise ValueError('实际批次身份漂移')
        presented.append(actual)

    model, optimizer, history = fit_continuation(model, optimizer_state, schedule, seed=seed,
                                                 check_budget=check_training_batch)
    if len(presented) != 24 or {int(s['step']) for s in optimizer.state.values()} != {264}:
        raise ValueError('批次/Adam计数不符')
    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), step=264,
                    threshold=model.stop_threshold, purpose='diagnostic_not_deployment'), directory / 'state.pt')
    point = dict(arm=arm, seed=seed, step=264, checkpoint=str(directory / 'state.pt'),
                 presented_batches=presented, loss_history=history)
    atomic_write_json(directory / 'report.json', point)
    point['probe'] = evaluate_cases(config['probe_indices'], context['cases'], model, budget, directory, 'probe')
    point['fit'] = evaluate_cases(config['fit_indices'], context['cases'], model, budget, directory, 'fit')
    atomic_write_json(directory / 'report.json', point)
    return point


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(180); torch.set_num_threads(1)
    report = dict(status='running', stage='identity_quality_and_pairing', completed_arms={}, training_started=False,
                  training_performed=False, allowed_to_prepare_training_plan=False,
                  main_protocol_passed=False, automatic_deployment=False)

    def save():
        report['elapsed_seconds'] = time.perf_counter() - started
        atomic_write_json(output / 'report.json', report)

    with exploration_lock(output):
        save()
        try:
            context = prepare_context(budget); config = context['config']
            atomic_write_json(output / 'config.json', config)
            report.update(config_sha256=file_digest(output / 'config.json'), quality_gate_passed=True,
                          source_quality_rows=config['source_quality_rows'], pairing_audits=config['pairing_audits'],
                          reused_controls=config['reused_controls']); save()
            parent = load_model_checkpoint(INPUTS['R0']).model
            for arm in ARMS:
                for seed in TRAINING_SEEDS:
                    budget.check(); directory = output / f'{arm}-{seed}'; directory.mkdir()
                    report.update(stage=f'{arm}-{seed}', training_started=True); save()
                    print(f'FIT {arm} seed={seed} updates=24', flush=True)
                    point = fit_point(context, arm, seed, parent, budget, directory)
                    report['completed_arms'].setdefault(arm, {})[seed] = point
                    report['training_performed'] = True; save()
            lt17 = context['lt17']; previous = context['consumption']
            int_keys = lambda values: {int(k): v for k, v in values.items()}
            new = {arm: {seed: p['probe'] for seed, p in values.items()}
                   for arm, values in report['completed_arms'].items()}
            verdict = quality_qualification(lt17['r0_probe'], int_keys(lt17['parent_probe']),
                {int(s): p['probe'] for s, p in lt17['completed_arms']['flat_old'].items()},
                {int(s): p['probe'] for s, p in previous['completed_seeds'].items()},
                new['clean_windows'], new['risky_windows'])
            verify_files(config['code']); verify_files(config['frozen_files']); budget.check()
            report.update(status='completed', stage='decision', qualification=verdict,
                          allowed_to_prepare_training_plan=verdict['passed'],
                          decision='quality_signal_prepare_manual_plan' if verdict['passed'] else
                                   'no_qualified_quality_signal_do_not_expand')
            report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                                         if p.is_file() and p != output / 'report.json' and
                                         p.name not in ('run.lock', 'console.log')}
            budget.check(); save(); print(f'FINISHED {report["decision"]}', flush=True)
            return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else
                          'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc),
                          allowed_to_prepare_training_plan=False, decision='incomplete_do_not_infer_benefit')
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    if args.precheck:
        precheck()
    elif args.output is None:
        parser.error('短检需要不存在的新output目录')
    else:
        run(args.output)
