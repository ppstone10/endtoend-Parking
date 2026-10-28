"""只改变关键新增保持帧的实际消费；固定24更新，单次总预算180秒。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration_data import evaluate_cases
from experiments.effective_recovery_data import input_id
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.retention_consumption import build_consumption_schedule, consumption_qualification
from experiments.retention_sampling import fit_continuation, sampling_audit, TRAINING_SEEDS
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix
from scripts.preflight_retention_sampling import verify_prepared, verify_files
from scripts.prepare_recovery_balance_ab import file_digest, metadata_only
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


LT17 = ROOT / 'runs/diagnostics/retention-sampling-diagnostic-20261021-v2'
LT17_HASHES = {
    'config.json': 'f4d08a8fed191d22dff443116e13dc70247fe099c128b27347a5173491864db0',
    'report.json': '0dda9859a681fbe6b8364d295fc3114aab0937efabdbeb6fe015fa749add6da9',
}


def prepare_context(budget):
    """在梯度前核验缓存来源，并生成两个完整且过门禁的批次计划。"""
    budget.check()
    verify_files({str(LT17 / name): digest for name, digest in LT17_HASHES.items()})
    previous_config = verify_prepared(LT17)
    report = json.loads((LT17 / 'report.json').read_text(encoding='utf-8'))
    if (report['status'] != 'completed' or report['branch_failures'] or
            not report['data_gate_passed'] or not report['baseline_gate']['passed'] or
            len(report['completed_arms']) != 4):
        raise ValueError('LT17不是完整可信对照')
    for relative, digest in report['artifacts_sha256'].items():
        budget.check()
        if file_digest(artifact_path(LT17, relative)) != digest:
            raise ValueError(f'LT17产物漂移：{relative}')
    if 'data.pt' not in report['artifacts_sha256']:
        raise ValueError('训练缓存未登记SHA，禁止加载')
    bank = torch.load(LT17 / 'data.pt', weights_only=False, map_location='cpu')
    old, novel = bank['old'], bank['novel']
    if len(old) != 390 or len(novel) != 113:
        raise ValueError('不是冻结的390＋113监督')
    if {input_id(s) for s in novel} != {
            d['input_sha256'] for d in report['novelty']['decisions'] if d['decision'] == 'accepted_new_input'}:
        raise ValueError('新增输入身份不一致')
    metadata = metadata_only(INPUTS['train'])
    gradient_ids = collect_metadata_ids([metadata[i] for i in sorted({s['source_index'] for s in old + novel})])
    held_out = collect_metadata_ids(metadata_only(INPUTS['val'])) | collect_metadata_ids(metadata_only(INPUTS['selection']))
    held_out |= collect_metadata_ids([metadata[i] for i in previous_config['groups']['probes'] +
                                      previous_config['groups']['regression_candidates']])
    if gradient_ids & held_out:
        raise ValueError('梯度来源与探针/保留集交叉')
    schedules = {}; audits = {}; actual_sampling = {}
    for seed in TRAINING_SEEDS:
        budget.check()
        schedules[seed], audits[seed] = build_consumption_schedule(old, novel, metadata, seed=seed)
        actual_sampling[seed] = sampling_audit(schedules[seed], metadata)
    code = dict(previous_config['code'])
    from experiments import retention_consumption
    for path in (Path(retention_consumption.__file__).resolve(), Path(__file__).resolve()):
        code[str(path)] = file_digest(path)
    frozen = {str(LT17 / name): digest for name, digest in LT17_HASHES.items()}
    frozen.update({str(LT17 / name): digest for name, digest in report['artifacts_sha256'].items()})
    # Explicitly pin start tensors as well as their predecessor unit manifests.
    frozen.update({path: file_digest(path) for path in previous_config['parent_checkpoints'].values()})
    config = dict(kind='retention_consumption_preflight_v1', code=code, frozen_files=frozen,
                  runtime=previous_config['runtime'], budget_seconds=180, start_step=240, steps=24,
                  parent_checkpoints=previous_config['parent_checkpoints'], training_seeds=list(TRAINING_SEEDS),
                  probe_indices=report['evaluation_probes'], fit_indices=report['fit_indices'],
                  consumption_audits=audits, sampling=actual_sampling,
                  reused_control='LT17 balanced_augmented: same pool and 24 updates; no new control training',
                  original_control='LT17 flat_old', reused_baseline='LT17 R0 and unmodified C240',
                  automatic_deployment=False, main_protocol_passed=False)
    budget.check()
    return dict(config=config, previous=report, schedules=schedules, metadata=metadata)


def precheck():
    context = prepare_context(PreflightBudget(180))
    print('CONSUMPTION PLAN VERIFIED; NO TRAINING OR ROLLOUT STARTED', flush=True)
    return context


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(180)
    torch.set_num_threads(1)
    report = dict(status='running', stage='identity_and_consumption_gate', completed_seeds={},
                  training_performed=False, allowed_to_prepare_training_plan=False,
                  main_protocol_passed=False, automatic_deployment=False)

    def save():
        report['elapsed_seconds'] = time.perf_counter() - started
        atomic_write_json(output / 'report.json', report)

    with exploration_lock(output):
        save()
        try:
            context = prepare_context(budget); config = context['config']; previous = context['previous']
            atomic_write_json(output / 'config.json', config)
            report.update(config_sha256=file_digest(output / 'config.json'), consumption_gate_passed=True,
                          consumption_audits=config['consumption_audits'],
                          reused_baseline=previous['parent_probe'], reused_R0=previous['r0_probe'],
                          reused_flat_control={s: a['probe'] for s, a in previous['completed_arms']['flat_old'].items()},
                          reused_same_pool_control={s: a['probe'] for s, a in previous['completed_arms']['balanced_augmented'].items()})
            save()
            data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
            vehicle = VehicleConfig(**manifest['vehicle_model'])
            cases = {i: build_case(data, manifest, i, vehicle) for i in config['probe_indices'] + config['fit_indices']}
            parent = load_model_checkpoint(INPUTS['R0']).model
            for seed in TRAINING_SEEDS:
                budget.check(); directory = output / str(seed); directory.mkdir()
                model, optimizer_state = load_candidate(config['parent_checkpoints'][str(seed)], parent)
                schedule = context['schedules'][seed]; planned = config['sampling'][seed]['batches']; presented = []

                def check_training_batch():
                    budget.check()
                    position = len(presented)
                    if position >= 24:
                        raise ValueError('实际训练超过24次')
                    batch = sampling_audit([schedule[position]], context['metadata'])['batches'][0]
                    if batch != planned[position]:
                        raise ValueError('实际输入身份偏离预声明计划')
                    presented.append(batch)

                report.update(stage=f'fit-{seed}'); save(); print(f'FIT seed={seed} updates=24', flush=True)
                model, optimizer, history = fit_continuation(model, optimizer_state, schedule, seed=seed,
                                                             check_budget=check_training_batch)
                if len(presented) != 24 or {int(v['step']) for v in optimizer.state.values()} != {264}:
                    raise ValueError('实际批次/Adam步数不符')
                report['training_performed'] = True
                torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), step=264,
                                threshold=model.stop_threshold, purpose='diagnostic_not_deployment'), directory / 'state.pt')
                point = dict(seed=seed, step=264, checkpoint=str(directory / 'state.pt'),
                             presented_batches=presented, loss_history=history)
                atomic_write_json(directory / 'report.json', point)
                report.update(stage=f'probe-{seed}'); save()
                probes = evaluate_cases(config['probe_indices'], cases, model, budget, directory, 'probe')
                fit = evaluate_cases(config['fit_indices'], cases, model, budget, directory, 'fit')
                point.update(probe=probes, fit=fit)
                atomic_write_json(directory / 'report.json', point)
                report['completed_seeds'][seed] = point; save()
            candidates = {seed: point['probe'] for seed, point in report['completed_seeds'].items()}
            int_keys = lambda rows: {int(seed): value for seed, value in rows.items()}
            verdict = consumption_qualification(previous['r0_probe'], int_keys(previous['parent_probe']),
                int_keys(report['reused_flat_control']), int_keys(report['reused_same_pool_control']), candidates)
            verify_files(config['code']); verify_files(config['frozen_files']); budget.check()
            report.update(status='completed', stage='decision', qualification=verdict,
                          allowed_to_prepare_training_plan=verdict['passed'],
                          decision='retention_consumption_signal_prepare_manual_plan' if verdict['passed'] else
                                   'no_qualified_consumption_signal_do_not_expand')
            report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                                         if p.is_file() and p != output / 'report.json' and
                                         p.name not in ('run.lock', 'console.log')}
            budget.check()
            save(); print(f'FINISHED {report["decision"]}', flush=True)
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
