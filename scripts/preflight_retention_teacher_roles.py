"""严格教师门禁之后独立验证来源分工；120秒，不恢复或修改先前结果。"""
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
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.retention_quality import audit_source_windows
from experiments.retention_sampling import sampling_audit, TRAINING_SEEDS
from experiments.retention_consumption import KEY_SOURCES
from experiments.retention_teacher import teacher_qualification
from experiments.retention_teacher_roles import source_role, fallback_pairs, build_role_schedules
from scripts.preflight_retention_teacher import prepare_context as prepare_strict_context
from scripts.preflight_retention_quality import fit_point, CONSUMPTION
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix
from scripts.prepare_recovery_balance_ab import file_digest
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json
from sim import VehicleConfig

STRICT = ROOT / 'runs/diagnostics/retention-teacher-preflight-20261022-v1'
LT17 = ROOT / 'runs/diagnostics/retention-sampling-diagnostic-20261021-v2'
HASHES = {'config.json': '91270d645e84a4abd04e3b40aaad4350d729573e1c11da161cb5d91835e1267a',
          'report.json': '6282585bc877b46892ca09c9cbd314e55560502fabc3e66c7842ce1654300357'}
ARMS = ('R0_targets', 'verified_targets')


def prepare_context(budget):
    verify_files({str(STRICT / k): h for k, h in HASHES.items()})
    stored = json.loads((STRICT / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((STRICT / 'report.json').read_text(encoding='utf-8'))
    verify_files(stored['code'])
    if report['status'] != 'completed_teacher_gate_failed' or report['training_started']:
        raise ValueError('不是已停止且未训练的严格教师证据')
    for relative, digest in report['artifacts_sha256'].items():
        budget.check()
        if file_digest(artifact_path(STRICT, relative)) != digest:
            raise ValueError('严格教师采集产物漂移')
    context = prepare_strict_context(budget)
    bank = torch.load(LT17 / 'data.pt', weights_only=False, map_location='cpu')
    saved = torch.load(STRICT / 'pairs.pt', weights_only=False, map_location='cpu')['pairs_by_seed']
    data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    config = dict(context['config']); config['code'] = dict(stored['code'])
    from experiments import retention_teacher_roles
    for path in (Path(retention_teacher_roles.__file__).resolve(), Path(__file__).resolve()):
        config['code'][str(path)] = file_digest(path)
    config['frozen_files'] = {**config['frozen_files'], **{str(STRICT / k): h for k, h in HASHES.items()},
        **{str(artifact_path(STRICT, k)): h for k, h in report['artifacts_sha256'].items()}}
    cases = {i: build_case(data, manifest, i, vehicle) for i in sorted(set(config['probe_indices'] + list(KEY_SOURCES)))}
    c240_rows = {(r['seed'], r['dataset_index']): r for r in report['source_rows']}
    r0_rows = {r['dataset_index']: r for r in context['previous']['collection']}
    plans = {a: {} for a in ARMS}; sampling = {a: {} for a in ARMS}; audits = {}; fallback_audits = {}
    for seed in TRAINING_SEEDS:
        pairs = {}; roles = {}; fallback_audits[seed] = {}
        for index in KEY_SOURCES:
            budget.check(); role = source_role(c240_rows[seed, index], r0_rows[index]); roles[index] = role
            if role == 'C240_preserve':
                pairs[index] = saved[seed][index]
            else:
                samples = [s for s in bank['old'] + bank['novel'] if s['source_index'] == index and s['role'] == 'anchor']
                with np.load(LT17 / f'collect-{index}/R0.npz') as archive:
                    quality = audit_source_windows(samples, archive['states'], cases[index], budget.check)
                pairs[index] = fallback_pairs(samples, quality)
                fallback_audits[seed][index] = dict(window_audit=quality,
                    selected_frames=[p['r0']['frame_index'] for p in pairs[index]])
        by_arm, audits[seed] = build_role_schedules(context['schedules'][seed], pairs, roles)
        for arm in ARMS:
            plans[arm][seed] = by_arm[arm]; sampling[arm][seed] = sampling_audit(by_arm[arm], context['metadata'])
    config.update(kind='retention_teacher_role_split_v1', budget_seconds=120, arms=list(ARMS),
                  fit_indices=list(KEY_SOURCES), sampling=sampling, pairing_audits=audits,
                  R0_restore_audits=fallback_audits,
                  source_gate_passed=True, comparison='same inputs/masks; C240 targets only on verified successful sources')
    context.update(config=config, plans=plans, cases=cases)
    budget.check(); return context


def precheck():
    context = prepare_context(PreflightBudget(120))
    print('SOURCE ROLES AND SHARED QUERIES VERIFIED; NO TRAINING OR ROLLOUT', flush=True)
    return context


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(120); torch.set_num_threads(1)
    report = dict(status='running', stage='identity_and_role_gate', completed_arms={}, training_started=False,
                  allowed_to_prepare_training_plan=False, main_protocol_passed=False, automatic_deployment=False)
    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_write_json(output / 'report.json', report)
    with exploration_lock(output):
        save()
        try:
            context = prepare_context(budget); config = context['config']
            atomic_write_json(output / 'config.json', config)
            report.update(config_sha256=file_digest(output / 'config.json'), pairing_audits=config['pairing_audits']); save()
            parent = load_model_checkpoint(INPUTS['R0']).model
            for arm in ARMS:
                for seed in TRAINING_SEEDS:
                    budget.check(); directory = output / f'{arm}-{seed}'; directory.mkdir()
                    report.update(stage=f'{arm}-{seed}', training_started=True); save()
                    print(f'FIT {arm} seed={seed} updates=24', flush=True)
                    point = fit_point(context, arm, seed, parent, budget, directory)
                    report['completed_arms'].setdefault(arm, {})[seed] = point; save()
            old = context['previous']; consumption = json.loads((CONSUMPTION / 'report.json').read_text(encoding='utf-8'))
            rows = {a: {s: p['probe'] for s, p in values.items()} for a, values in report['completed_arms'].items()}
            verdict = teacher_qualification(old['r0_probe'], {int(s): p for s, p in old['parent_probe'].items()},
                {int(s): p['probe'] for s, p in old['completed_arms']['flat_old'].items()},
                {int(s): p['probe'] for s, p in consumption['completed_seeds'].items()},
                rows['R0_targets'], rows['verified_targets'],
                {s: p['fit'] for s, p in report['completed_arms']['verified_targets'].items()})
            verdict['verified_teacher_gate'] = verdict.pop('C240_gate')
            verify_files(config['code']); verify_files(config['frozen_files']); budget.check()
            report.update(status='completed', stage='decision', qualification=verdict,
                          allowed_to_prepare_training_plan=verdict['passed'],
                          decision='verified_teacher_signal_prepare_manual_plan' if verdict['passed'] else
                                   'no_qualified_role_split_signal_do_not_expand')
            report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                if p.is_file() and p != output / 'report.json' and p.name not in ('run.lock', 'console.log')}
            budget.check(); save(); print(f'FINISHED {report["decision"]}', flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else 'failed', error=str(exc),
                          allowed_to_prepare_training_plan=False, decision='incomplete_do_not_infer_benefit')
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path); parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output目录')
    else: run(args.output)
