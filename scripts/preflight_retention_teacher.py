"""共同C240访问输入上的R0/C240保持教师对照；默认总预算180秒。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration_data import run_case, save_record
from experiments.executable_behavior import PreflightBudget, prepare_supervision_frames
from experiments.exploration_store import exploration_lock, artifact_path
from experiments.recovery_execution import record_arrays
from experiments.retention_quality import audit_source_windows
from experiments.retention_teacher import select_teacher_pairs, build_teacher_schedules, teacher_qualification
from experiments.retention_consumption import KEY_SOURCES
from experiments.retention_sampling import sampling_audit, TRAINING_SEEDS
from scripts.preflight_retention_consumption import prepare_context as prepare_consumption_context
from scripts.preflight_retention_quality import fit_point
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix, result_row
from scripts.explore_closed_loop_learning import load_candidate
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


QUALITY = ROOT / 'runs/diagnostics/retention-quality-preflight-20261021-v1'
QUALITY_HASHES = {'config.json': 'b36ecfee20f4cb8c8a32a36ece156ec52171900efe925e53684aa1b60940bf04',
                  'report.json': 'e3fe2f7dd3ea9270919769e02c720df7bdca1d7daf19eb1e400cc891e42900dc'}
ARMS = ('R0_targets', 'C240_targets')


def prepare_context(budget):
    budget.check(); verify_files({str(QUALITY / k): h for k, h in QUALITY_HASHES.items()})
    old = json.loads((QUALITY / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((QUALITY / 'report.json').read_text(encoding='utf-8'))
    verify_files(old['code'])
    if report['status'] != 'completed':
        raise ValueError('上轮质量对照不是完整存档')
    for relative, digest in report['artifacts_sha256'].items():
        budget.check()
        if file_digest(artifact_path(QUALITY, relative)) != digest:
            raise ValueError('质量对照产物漂移')
    context = prepare_consumption_context(budget)
    config = dict(context['config']); config['code'] = dict(old['code'])
    from experiments import retention_teacher
    for path in (Path(retention_teacher.__file__).resolve(), Path(__file__).resolve()):
        config['code'][str(path)] = file_digest(path)
    config['frozen_files'] = {**config['frozen_files'], **{str(QUALITY / k): h for k, h in QUALITY_HASHES.items()},
        **{str(artifact_path(QUALITY, k)): h for k, h in report['artifacts_sha256'].items()}}
    config.update(kind='retention_teacher_preflight_v1', source_indices=list(KEY_SOURCES), arms=list(ARMS),
                  fit_indices=list(KEY_SOURCES), source_gate_passed=False,
                  comparison='same C240 visited inputs/masks; only 12 anchor point/soft-stop targets differ')
    context['config'] = config
    return context


def precheck():
    context = prepare_context(PreflightBudget(180))
    print('INPUT IDENTITY VERIFIED; TEACHER DATA GATE NOT RUN; NO ROLLOUT OR TRAINING', flush=True)
    return context


def collect_sources(context, budget, output, report, save):
    config = context['config']; parent = load_model_checkpoint(INPUTS['R0']).model
    context['parent_model'] = parent
    data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    context['cases'] = {i: build_case(data, manifest, i, vehicle)
                        for i in sorted(set(config['probe_indices'] + list(KEY_SOURCES)))}
    all_pairs = {}; source_rows = []; complete = True
    for seed in TRAINING_SEEDS:
        teacher, _ = load_candidate(config['parent_checkpoints'][str(seed)], parent)
        all_pairs[seed] = {}
        for index in KEY_SOURCES:
            budget.check(); directory = output / f'source-{seed}-{index}'; directory.mkdir()
            report['stage'] = f'collect-{seed}-{index}'; save()
            print(f'COLLECT C240 seed={seed} source={index}', flush=True)
            case = context['cases'][index]; result, engine, frames = run_case(case, teacher, budget)
            save_record(directory, 'C240', result)
            row = dict(seed=seed, **result_row(case, result, engine))
            if result.success:
                samples = prepare_supervision_frames(frames, teacher, 'anchor', index, frame_limit=None)
                states, _ = record_arrays(result)
                quality = audit_source_windows(samples, states, case, budget.check)
                pairs, audit = select_teacher_pairs(samples, quality, parent, budget.check)
                row.update(window_audit=quality, query_gate=audit)
                if audit['passed']:
                    all_pairs[seed][index] = pairs
                else:
                    complete = False
            else:
                row['query_gate'] = dict(passed=False, reason='C240_full_source_failed'); complete = False
            source_rows.append(row); atomic_write_json(directory / 'report.json', row)
            report['source_rows'] = source_rows; save()
    torch.save(dict(purpose='paired_training_sources_only', pairs_by_seed=all_pairs), output / 'pairs.pt')
    if complete:
        plans = {arm: {} for arm in ARMS}; sampling = {arm: {} for arm in ARMS}; pairing = {}
        for seed in TRAINING_SEEDS:
            by_arm, pairing[seed] = build_teacher_schedules(context['schedules'][seed], all_pairs[seed])
            for arm in ARMS:
                plans[arm][seed] = by_arm[arm]
                sampling[arm][seed] = sampling_audit(by_arm[arm], context['metadata'])
        context['plans'] = plans
        config.update(sampling=sampling, pairing_audits=pairing)
    config.update(source_gate_passed=complete, source_rows=source_rows)
    context['source_gate_passed'] = complete
    return complete


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(180); torch.set_num_threads(1)
    report = dict(status='running', stage='identity', completed_arms={}, training_started=False,
                  training_performed=False, allowed_to_prepare_training_plan=False,
                  main_protocol_passed=False, automatic_deployment=False)

    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_write_json(output / 'report.json', report)

    def finalize(context):
        verify_files(context['config']['code']); verify_files(context['config']['frozen_files']); budget.check()
        report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
            if p.is_file() and p != output / 'report.json' and p.name not in ('run.lock', 'console.log')}
        budget.check(); save(); print(f'FINISHED {report["decision"]}', flush=True)

    with exploration_lock(output):
        save()
        try:
            context = prepare_context(budget)
            atomic_write_json(output / 'input-config.json', context['config'])
            report['input_config_sha256'] = file_digest(output / 'input-config.json'); save()
            passed = collect_sources(context, budget, output, report, save)
            atomic_write_json(output / 'config.json', context['config'])
            report.update(config_sha256=file_digest(output / 'config.json'), source_gate_passed=passed)
            if not passed:
                report.update(status='completed_teacher_gate_failed', decision='no_training_untrusted_teacher_sources')
                finalize(context); return report
            for arm in ARMS:
                for seed in TRAINING_SEEDS:
                    budget.check(); directory = output / f'{arm}-{seed}'; directory.mkdir()
                    report.update(stage=f'{arm}-{seed}', training_started=True); save()
                    print(f'FIT {arm} seed={seed} updates=24', flush=True)
                    point = fit_point(context, arm, seed, context['parent_model'], budget, directory)
                    report['completed_arms'].setdefault(arm, {})[seed] = point
                    report['training_performed'] = True; save()
            previous = context['previous']
            candidates = {arm: {s: p['probe'] for s, p in values.items()}
                          for arm, values in report['completed_arms'].items()}
            # Consumption baseline is already frozen and verified upstream.
            from scripts.preflight_retention_quality import CONSUMPTION
            consumption = json.loads((CONSUMPTION / 'report.json').read_text(encoding='utf-8'))
            verdict = teacher_qualification(previous['r0_probe'], {int(s): p for s, p in previous['parent_probe'].items()},
                {int(s): p['probe'] for s, p in previous['completed_arms']['flat_old'].items()},
                {int(s): p['probe'] for s, p in consumption['completed_seeds'].items()},
                candidates['R0_targets'], candidates['C240_targets'],
                {s: p['fit'] for s, p in report['completed_arms']['C240_targets'].items()})
            report.update(status='completed', stage='decision', qualification=verdict,
                          allowed_to_prepare_training_plan=verdict['passed'],
                          decision='teacher_retention_signal_prepare_manual_plan' if verdict['passed'] else
                                   'no_qualified_teacher_signal_do_not_expand')
            finalize(context); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else
                          'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc),
                          allowed_to_prepare_training_plan=False, decision='incomplete_do_not_infer_benefit')
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path); parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output目录')
    else: run(args.output)
