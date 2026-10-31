"""Frozen LT37 refinement/ablation study. Formal execution is human-operated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.emitted_output_data import INDICES, case_at, load_cohort, file_digest, file_identity, read_json
from scripts import compare_feasible_cem as legacy
from scripts.diagnose_emitted_execution import PARENT, TEACHER, metric_summary
from scripts.feasible_cem_capture import audit_execution, audit_predictions
from scripts.runtime_refinement_capture import record_refined_branch, audit_refinement
from scripts.runtime_refinement_mpc import PROFILES
from scripts.runtime_refinement_preflight import run_probes, probe_code, PROBES
from scripts.train_emitted_output_joint import StageBudget
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

OLD_PREPARED = ROOT / 'runs/prepared/feasible-cem-20261029-v3'
OLD_SHA = 'ef300846382c6f495600d69f925ed72091ca3bf6c3c9925d7ee08a4744ac8107'
BASELINE = ROOT / 'runs/diagnostics/lt36-feasible-cem-20261029/results'
PREFLIGHT = ROOT / 'runs/verification/runtime-refinement-20261029/probes-v2.json'
PROTOCOL = dict(max_steps=600, smoke_steps=12, budget_seconds=7200, smoke_seconds=90,
    dt=.1, horizon=10, population=80, iterations=5, replan_every=10, seed='dataset_index',
    stop_threshold=.5, trajectory_dt=.2, network_horizon=320, required_margin_m=.2,
    alpha=.1, std_floor_fraction=.1, min_fit_elites=2, progress_position=.15,
    progress_yaw=.1, catchup_window=4, profiles=PROFILES,
    geometry_source='simulator_ground_truth', training_authorized=False,
    reference_repair=False, expert_fallback=False)


def jobs(smoke=False):
    combined = [dict(profile='combined', index=i, branch=b)
                for i in (INDICES[:1] if smoke else INDICES) for b in legacy.BRANCHES]
    if smoke:
        return combined
    return combined + [dict(profile=p, index=860, branch='teacher_fixed')
                       for p in ('fast', 'variance', 'selection', 'progress')] + [
                           dict(profile=p, index=1290, branch='network_rolling') for p in ('fast', 'terminal')]


def code_identity():
    files = [Path(__file__), ROOT / 'scripts/batched_footprint_check.py',
        ROOT / 'scripts/runtime_refinement_mpc.py', ROOT / 'scripts/runtime_refinement_capture.py',
        ROOT / 'scripts/runtime_refinement_preflight.py']
    return {**legacy.code_identity(), **{str(p.resolve()): file_digest(p) for p in files}}


def frozen_config():
    if file_digest(OLD_PREPARED / 'config.json') != OLD_SHA:
        raise ValueError('LT36 frozen identity changed')
    old = legacy.precheck(OLD_PREPARED)
    probe = read_json(PREFLIGHT)
    if (probe['status'] != 'passed' or probe['exact_probes'] != 19 or probe['candidate_qualified']
            or probe['code'] != probe_code()):
        raise ValueError('bounded exact-equivalence preflight required')
    # Hash the LT36 manifest/report and verify every registered file. The original
    # complete scalar audit was already performed; do not rerun it per precheck.
    report = read_json(BASELINE / 'report.json')
    if not report['formal'] or report['status'] != 'completed' or report['indices'] != INDICES:
        raise ValueError('complete LT36 baseline required')
    artifacts = read_json(BASELINE / 'artifacts.json')
    expected = {str((BASELINE / name).resolve()) for name in legacy.required_files(INDICES)}
    actual = {str(p.resolve()) for p in BASELINE.iterdir() if p.is_file() and p.name != 'artifacts.json'}
    if set(artifacts) != expected or actual != expected:
        raise ValueError('LT36 baseline artifact denominator mismatch')
    for path, digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError('LT36 baseline artifact changed')
    return dict(schema='runtime_refinement_v1', protocol=PROTOCOL, jobs=jobs(), indices=INDICES,
        code=code_identity(), checkpoint=old['checkpoint'], sources=old['sources'],
        old_config=file_identity(OLD_PREPARED / 'config.json'),
        baseline_artifacts=file_identity(BASELINE / 'artifacts.json'),
        probes=file_identity(PROBES), preflight=file_identity(PREFLIGHT),
        candidate_qualified=False, training_performed=False)


def prepare(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('new prepared directory required')
    config = frozen_config()
    destination.mkdir(parents=True)
    atomic_write_json(destination / 'config.json', config)
    return config


def precheck(prepared):
    config = read_json(Path(prepared) / 'config.json')
    if config != frozen_config():
        raise ValueError('frozen code/protocol/input drift; never edit hashes to bypass')
    return config


def summaries(rows, formal):
    selected = [r for r in rows if r['profile'] == 'combined']
    metrics = {b: metric_summary([r for r in selected if r['branch'] == b]) for b in legacy.BRANCHES}
    if not formal:
        return dict(metrics=metrics, decision=dict(status='truncated_smoke_only',
            ready_for_next_validation=False), ablations=[], production_qualified=False,
            lt34_training_unlocked=False)
    baseline = read_json(BASELINE / 'report.json')['rows']
    comparisons = {b: legacy.compare_rows([r[b] for r in baseline],
                    [r for r in selected if r['branch'] == b]) for b in legacy.BRANCHES}
    teacher = metrics['teacher_fixed']['successes'] == 8
    restored = all(next(r for r in selected if r['dataset_index'] == 860 and r['branch'] == b)['success']
                   for b in ('network_fixed', 'teacher_fixed'))
    retained = all(not c['lost_successes'] and not c['risky_tasks'] for c in comparisons.values())
    timely = all(r['control_timing']['over_100ms'] == 0 for r in selected)
    ready = bool(teacher and restored and retained and timely)
    ablations = []
    for index, branch, profiles in [(860, 'teacher_fixed', ['variance', 'selection', 'progress']),
                                     (1290, 'network_rolling', ['terminal'])]:
        fast = next(r for r in rows if r['dataset_index'] == index and r['branch'] == branch and r['profile'] == 'fast')
        for profile in profiles:
            new = next(r for r in rows if r['dataset_index'] == index and r['branch'] == branch and r['profile'] == profile)
            ablations.append(dict(index=index, branch=branch, profile=profile,
                comparison=legacy.compare_rows([fast], [new]), scope='one_fixed_witness_only'))
    if not retained:
        status = 'reject_candidate_regression_or_risk'
    elif not teacher or not restored:
        status = 'stop_refinement_control_contract_unresolved'
    elif not timely:
        status = 'stop_real_time_budget_failed'
    else:
        status = 'bounded_gates_passed_prepare_only_next_validation'
    return dict(metrics=metrics, decision=dict(status=status, comparisons=comparisons,
        teacher_strict_pass=teacher, restored_860_fixed_and_teacher=restored,
        all_old_successes_retained=retained, observed_within_100ms=timely,
        ready_for_next_validation=ready), ablations=ablations,
        production_qualified=False, lt34_training_unlocked=False)


def verify_fast_episode(output, detail):
    """Two fast-only complete episodes must reproduce the frozen LT36 arrays."""
    stem = f'{detail["dataset_index"]:04d}-{detail["branch"]}'
    with np.load(BASELINE / (stem+'.npz'), allow_pickle=False) as old:
        with np.load(Path(output).with_suffix('.npz'), allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                raise ValueError('fast-only archive fields changed')
            for key in old.files:
                np.testing.assert_array_equal(old[key], new[key])
    old = read_json(BASELINE / (stem+'.json'))
    if len(old['searches']) != len(detail['searches']):
        raise ValueError('fast-only search denominator changed')
    for a, b in zip(old['searches'], detail['searches']):
        for ga, gb in zip(a['generations'], b['generations']):
            for key in ('feasible_mask', 'elite_indices'):
                if ga[key] != gb[key]:
                    raise ValueError('fast-only CEM eligibility/elite changed')


def verify_initial(output, detail):
    if detail['branch'] == 'teacher_fixed':
        return
    stem = f'{detail["dataset_index"]:04d}-{detail["branch"]}-predictions.npz'
    with np.load(BASELINE / stem, allow_pickle=False) as old:
        with np.load(Path(output).parent / (Path(output).name+'-predictions.npz'), allow_pickle=False) as new:
            for key in ('points', 'logits', 'bev', 'goal', 'motion', 'pose', 'lengths'):
                np.testing.assert_array_equal(old[key][0], new[key][0])


def run(prepared, destination, smoke=False):
    config = precheck(prepared)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', formal=not smoke, jobs=jobs(smoke), rows=[],
        config_sha256=file_digest(Path(prepared) / 'config.json'), geometry_source='simulator_ground_truth',
        candidate_qualified=False, training_performed=False)
    atomic_write_json(destination / 'config.json', config)
    atomic_write_json(destination / 'report.json', report)
    budget = StageBudget(PROTOCOL['smoke_seconds'] if smoke else PROTOCOL['budget_seconds'])
    try:
        data, manifest, vehicle = load_cohort(PARENT)
        model = load_model_checkpoint(config['checkpoint']['path']).model
        if model.stop_threshold != .5 or model.max_horizon != 320 or model.dt != .2:
            raise ValueError('native prediction contract drift')
        for job in report['jobs']:
            budget.check()
            case = case_at(data, manifest, vehicle, INDICES.index(job['index']))
            sub = destination / job['profile']
            sub.mkdir(exist_ok=True)
            output = sub / f'{job["index"]:04d}-{job["branch"]}'
            print(f'{job["profile"]} #{job["index"]} {job["branch"]} BEGIN', flush=True)
            summary = record_refined_branch(case, model, output, budget, profile=job['profile'],
                branch=job['branch'], teacher_archive=TEACHER / f'teacher-{job["index"]}.npz',
                max_steps=PROTOCOL['smoke_steps'] if smoke else PROTOCOL['max_steps'])
            verify_initial(output, summary)
            if job['profile'] == 'fast':
                verify_fast_episode(output, read_json(output.with_suffix('.json')))
            report['rows'].append(summary)
            print(f'{job["profile"]} #{job["index"]} {job["branch"]} success={summary["success"]} '
                f'collision={summary["collision"]} margin={summary["margin_violation_steps"]} '
                f'steps={summary["steps"]} p95_ms={summary["control_timing"]["p95_ms"]}', flush=True)
            atomic_write_json(destination / 'report.json', report)
        for index in (INDICES[:1] if smoke else INDICES):
            legacy.verify_initial_predictions(destination / 'combined', index)
        precheck(prepared)
        budget.check()
        report.update(status='completed', elapsed_seconds=time.perf_counter()-budget.started,
                      **summaries(report['rows'], formal=not smoke))
        atomic_write_json(destination / 'report.json', report)
        atomic_write_json(destination / 'artifacts.json',
            {str(p.resolve()): file_digest(p) for p in destination.rglob('*') if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete', error=str(error), elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(destination / 'report.json', report)
        raise


def required_files(smoke):
    names = {'config.json', 'report.json'}
    for job in jobs(smoke):
        stem = f'{job["profile"]}/{job["index"]:04d}-{job["branch"]}'
        names.update((stem+'.json', stem+'.npz'))
        if job['branch'] != 'teacher_fixed':
            names.add(stem+'-predictions.npz')
    return names


def inspect(prepared, destination):
    config = precheck(prepared)
    destination = Path(destination).resolve()
    report = read_json(destination / 'report.json')
    smoke = not report['formal']
    if (report['status'] != 'completed' or report['jobs'] != jobs(smoke)
            or len(report['rows']) != len(jobs(smoke))
            or report['config_sha256'] != file_digest(Path(prepared) / 'config.json')
            or read_json(destination / 'config.json') != config
            or report['candidate_qualified'] or report['training_performed']
            or report['geometry_source'] != 'simulator_ground_truth'):
        raise ValueError('completion/config/denominator/qualification mismatch')
    artifacts = read_json(destination / 'artifacts.json')
    expected = {str((destination / name).resolve()) for name in required_files(smoke)}
    actual = {str(p.resolve()) for p in destination.rglob('*') if p.is_file() and p != destination / 'artifacts.json'}
    if set(artifacts) != expected or actual != expected:
        raise ValueError('missing or extra registered episode artifacts')
    for path, digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError('registered artifact changed')
    data, manifest, vehicle = load_cohort(PARENT)
    transitions = 0
    for job, row in zip(jobs(smoke), report['rows']):
        case = case_at(data, manifest, vehicle, INDICES.index(job['index']))
        if (row['profile'] != job['profile'] or row['branch'] != job['branch']
                or row['dataset_index'] != job['index'] or row['task_id'] != case['task_id']
                or config['sources'][INDICES.index(job['index'])]['task_id'] != row['task_id']):
            raise ValueError('paired task/profile identity mismatch')
        output = destination / job['profile'] / f'{job["index"]:04d}-{job["branch"]}'
        detail = read_json(output.with_suffix('.json'))
        if (any(detail[key] != value for key, value in row.items())
                or detail['execution_npz_sha256'] != file_digest(output.with_suffix('.npz'))
                or detail['geometry_source'] != 'simulator_ground_truth' or detail['required_margin_m'] != .2
                or detail['candidate_qualified'] or detail['training_performed']
                or bool(detail['truncated_preflight']) != smoke):
            raise ValueError('episode summary/identity/evidence mismatch')
        transitions += audit_execution(output, detail, case)['selected_h10_transitions']
        audit_refinement(output, detail, case)
        audit_predictions(output, detail, TEACHER / f'teacher-{job["index"]}.npz')
        verify_initial(output, detail)
        if job['profile'] == 'fast':
            verify_fast_episode(output, detail)
    for index in (INDICES[:1] if smoke else INDICES):
        legacy.verify_initial_predictions(destination / 'combined', index)
    calculated = summaries(report['rows'], formal=not smoke)
    if any(report[key] != value for key, value in calculated.items()):
        raise ValueError('recomputed metrics/decision mismatch')
    return dict(status='verified', formal=not smoke, tasks=1 if smoke else 8,
        episodes=len(report['rows']), artifacts=len(artifacts),
        independently_audited_h10_transitions=transitions, **calculated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['probe', 'prepare', 'precheck', 'smoke', 'run', 'inspect'], required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'probe':
        if args.output is None:
            parser.error('--output required for probe')
        r = run_probes(args.output)
        result = {k: r[k] for k in ('status', 'exact_probes', 'scalar_p50_ms', 'batch_p50_ms', 'batch_max_ms')}
    else:
        if args.prepared is None:
            parser.error('--prepared required')
        if args.phase in {'prepare', 'precheck'}:
            config = prepare(args.prepared) if args.phase == 'prepare' else precheck(args.prepared)
            result = dict(status='prepared' if args.phase == 'prepare' else 'passed', jobs=len(config['jobs']))
        else:
            if args.output is None:
                parser.error('--output required')
            if args.phase == 'inspect':
                result = inspect(args.prepared, args.output)
            else:
                r = run(args.prepared, args.output, smoke=args.phase == 'smoke')
                result = {k: r[k] for k in ('status', 'formal', 'elapsed_seconds', 'metrics', 'decision')}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
