"""Frozen eight-task feasible-CEM experiment; formal runs are human-operated."""
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

from experiments.emitted_output_data import INDICES, load_cohort, case_at, file_identity, file_digest, read_json
from scripts.diagnose_emitted_execution import (
    PARENT, TEACHER, current_code as baseline_code, inspect as inspect_baseline, metric_summary,
)
from scripts.feasible_cem_capture import record_feasible_branch, audit_execution, audit_predictions
from scripts.train_emitted_output_joint import StageBudget
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

BASELINE_PREPARED = ROOT / 'runs/prepared/emitted-execution-diagnosis-20261028-v1'
BASELINE = ROOT / 'runs/diagnostics/lt35-emitted-execution-20261028/results'
BASELINE_SHA = '0b5fe08e72e8d696493ba92c90e9cd3392cfa1d46a80b2625c2c5af0075c76dd'
BRANCHES = ('network_fixed', 'network_rolling', 'teacher_fixed')
PROTOCOL = dict(max_steps=600, smoke_steps=12, replan_every=10, mpc_dt=.1, mpc_horizon=10,
    population=80, iterations=5, stop_threshold=.5, trajectory_dt=.2, horizon=320,
    required_margin_m=.2, geometry_source='simulator_ground_truth',
    budget_seconds=7200, smoke_seconds=90, reference_repair=False, expert_fallback=False,
    terminal_takeover=False, training_authorized=False)


def code_identity():
    files = [Path(__file__).resolve(), ROOT / 'scripts/feasible_cem_capture.py',
             ROOT / 'scripts/feasible_cem_mpc.py']
    return {**baseline_code(), **{str(path): file_digest(path) for path in files}}


def frozen_config():
    if file_digest(BASELINE_PREPARED / 'config.json') != BASELINE_SHA:
        raise ValueError('LT35 baseline config identity changed')
    verified = inspect_baseline(BASELINE_PREPARED, BASELINE)
    if not verified['formal'] or verified['tasks'] != 8:
        raise ValueError('complete formal eight-task LT35 baseline required')
    old = read_json(BASELINE_PREPARED / 'config.json')
    return dict(schema='feasible_cem_study_v1', code=code_identity(), protocol=PROTOCOL,
        indices=INDICES, sources=old['sources'], checkpoint=old['checkpoint'],
        baseline_config=file_identity(BASELINE / 'config.json'),
        baseline_artifacts=file_identity(BASELINE / 'artifacts.json'),
        teacher_artifacts=file_identity(TEACHER / 'artifacts.json'),
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
        raise ValueError('frozen identity/protocol drift; do not edit SHA to bypass')
    return config


def compare_rows(original, candidate):
    """Arrival gains cannot offset losses; an all-stop result cannot pass."""
    def keyed(rows):
        keys = [r['task_id'] for r in rows]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError('nonempty unique task denominator required')
        return {r['task_id']: r for r in rows}
    old, new = keyed(original), keyed(candidate)
    if set(old) != set(new):
        raise ValueError('paired task denominator mismatch')
    for key in old:
        if old[key]['dataset_index'] != new[key]['dataset_index']:
            raise ValueError('paired source identity mismatch')
    lost = [old[k]['dataset_index'] for k in old if old[k]['success'] and not new[k]['success']]
    gained = [old[k]['dataset_index'] for k in old if not old[k]['success'] and new[k]['success']]
    risks = [new[k]['dataset_index'] for k in old
             if new[k]['collision'] or new[k]['margin_violation_steps'] > 0]
    return dict(tasks=len(old), lost_successes=lost, gained_successes=gained,
                risky_tasks=risks, mechanism_gain=bool(gained and not lost and not risks))


def summaries(rows, *, formal):
    metrics = {branch: metric_summary([r[branch] for r in rows]) for branch in BRANCHES}
    teacher_pass = bool(formal and len(rows) == 8 and metrics['teacher_fixed']['successes'] == 8
                        and metrics['teacher_fixed']['collisions'] == 0
                        and metrics['teacher_fixed']['margin_violation_steps'] == 0)
    if formal:
        baseline = read_json(BASELINE / 'report.json')
        indices = {r['dataset_index'] for r in rows}
        comparisons = {branch: compare_rows(
            [r[branch] for r in baseline['rows'] if r['dataset_index'] in indices],
            [r[branch] for r in rows]) for branch in BRANCHES[:2]}
        # A fixed-reference gain cannot stand in for a rolling-reference gain.
        decisions = {}
        for branch in BRANCHES[:2]:
            timely = all(r[branch]['control_timing']['over_100ms'] == 0 for r in rows)
            decisions[branch] = dict(**comparisons[branch], observed_within_100ms=timely,
                small_cohort_gain=comparisons[branch]['mechanism_gain'],
                ready_for_next_validation=bool(comparisons[branch]['mechanism_gain'] and timely and teacher_pass))
    else:
        # A twelve-step prefix has a different horizon from the completed baseline.
        decisions = {branch: dict(status='not_evaluated_truncated_smoke', tasks=len(rows),
            small_cohort_gain=False, ready_for_next_validation=False) for branch in BRANCHES[:2]}
    return dict(metrics=metrics, comparisons=decisions, new_contract_teacher_strict_pass=teacher_pass,
                lt34_training_unlocked=False, production_qualified=False)


def verify_initial_predictions(destination, index):
    """All arms share the original same-input initial free decode, without truncation."""
    original = BASELINE / f'{index:04d}-network_fixed-predictions.npz'
    with np.load(original, allow_pickle=False) as archived:
        for branch in BRANCHES[:2]:
            with np.load(destination / f'{index:04d}-{branch}-predictions.npz', allow_pickle=False) as new:
                for key in ('points', 'logits', 'bev', 'goal', 'motion', 'pose', 'lengths'):
                    np.testing.assert_array_equal(archived[key][0], new[key][0])
    with np.load(destination / f'{index:04d}-network_fixed.npz', allow_pickle=False) as fixed:
        with np.load(destination / f'{index:04d}-network_rolling.npz', allow_pickle=False) as rolling:
            shared = min(10, len(fixed['states_after']))
            if min(10, len(rolling['states_after'])) != shared:
                raise ValueError('paired feasible-controller execution prefix length mismatch')
            np.testing.assert_array_equal(fixed['global_plan_0000'], rolling['global_plan_0000'])
            for key in ('states_before', 'states_after', 'control_windows', 'selected_control_sequences'):
                np.testing.assert_array_equal(fixed[key][:shared], rolling[key][:shared])


def run(prepared, destination, *, smoke=False):
    config = precheck(prepared)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', formal=not smoke, indices=INDICES[:1] if smoke else INDICES,
        rows=[], config_sha256=file_digest(Path(prepared) / 'config.json'),
        geometry_source='simulator_ground_truth', candidate_qualified=False, training_performed=False)
    atomic_write_json(destination / 'config.json', config)
    atomic_write_json(destination / 'report.json', report)
    budget = StageBudget(PROTOCOL['smoke_seconds'] if smoke else PROTOCOL['budget_seconds'])
    try:
        data, manifest, vehicle = load_cohort(PARENT)
        model = load_model_checkpoint(config['checkpoint']['path']).model
        if model.stop_threshold != .5 or model.max_horizon != 320 or model.dt != .2:
            raise ValueError('native model contract drift')
        for ordinal, index in enumerate(report['indices']):
            budget.check()
            case = case_at(data, manifest, vehicle, ordinal)
            row = dict(dataset_index=index, task_id=case['task_id'], scene_name=case['scene_name'])
            for branch in BRANCHES:
                print(f'#{index} {branch} BEGIN', flush=True)
                row[branch] = record_feasible_branch(case, model, destination / f'{index:04d}-{branch}',
                    budget, branch=branch, teacher_archive=TEACHER / f'teacher-{index}.npz',
                    max_steps=PROTOCOL['smoke_steps'] if smoke else PROTOCOL['max_steps'])
                s = row[branch]
                print(f'#{index} {branch} success={s["success"]} collision={s["collision"]} '
                      f'margin={s["margin_violation_steps"]} holds={s["zero_hold_steps"]} '
                      f'controls={s["steps"]} p95_ms={s["control_timing"]["p95_ms"]}', flush=True)
                atomic_write_json(destination / 'report.json', report)
            verify_initial_predictions(destination, index)
            report['rows'].append(row)
            atomic_write_json(destination / 'report.json', report)
        precheck(prepared)
        budget.check()
        report.update(status='completed', elapsed_seconds=time.perf_counter() - budget.started,
            **summaries(report['rows'], formal=not smoke))
        atomic_write_json(destination / 'report.json', report)
        atomic_write_json(destination / 'artifacts.json',
            {str(p.resolve()): file_digest(p) for p in destination.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete', error=str(error), elapsed_seconds=time.perf_counter() - budget.started)
        atomic_write_json(destination / 'report.json', report)
        raise


def required_files(indices):
    names = {'config.json', 'report.json'}
    for index in indices:
        for branch in BRANCHES:
            stem = f'{index:04d}-{branch}'
            names.update((stem + '.json', stem + '.npz'))
            if branch != 'teacher_fixed':
                names.add(stem + '-predictions.npz')
    return names


def inspect(prepared, destination):
    config = precheck(prepared)
    destination = Path(destination).resolve()
    report = read_json(destination / 'report.json')
    indices = INDICES if report['formal'] else INDICES[:1]
    if (report['status'] != 'completed' or report['indices'] != indices
            or [r['dataset_index'] for r in report['rows']] != indices
            or report['config_sha256'] != file_digest(Path(prepared) / 'config.json')
            or read_json(destination / 'config.json') != config
            or report['training_performed'] or report['candidate_qualified']
            or report['geometry_source'] != 'simulator_ground_truth'):
        raise ValueError('completion/config/qualification/denominator mismatch')
    names = required_files(indices)
    artifacts = read_json(destination / 'artifacts.json')
    expected = {str((destination / name).resolve()) for name in names}
    actual = {str(p.resolve()) for p in destination.iterdir() if p.is_file() and p.name != 'artifacts.json'}
    if set(artifacts) != expected or actual != expected:
        raise ValueError('required branch artifacts missing or extra')
    for path, digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError(f'artifact changed: {path}')
    data, manifest, vehicle = load_cohort(PARENT)
    transitions = 0
    for ordinal, row in enumerate(report['rows']):
        case = case_at(data, manifest, vehicle, ordinal)
        if row['task_id'] != case['task_id'] or config['sources'][ordinal]['task_id'] != row['task_id']:
            raise ValueError('task identity drift')
        for branch in BRANCHES:
            output = destination / f'{row["dataset_index"]:04d}-{branch}'
            detail = read_json(output.with_suffix('.json'))
            if (detail['branch'] != branch or detail['task_id'] != row['task_id']
                    or detail['required_margin_m'] != .2 or detail['geometry_source'] != 'simulator_ground_truth'
                    or detail['candidate_qualified'] or detail['training_performed']
                    or any(detail[key] != value for key, value in row[branch].items())):
                raise ValueError('branch summary/identity drift')
            with np.load(output.with_suffix('.npz'), allow_pickle=False) as saved:
                if detail['execution_npz_sha256'] != file_digest(output.with_suffix('.npz')):
                    raise ValueError('branch execution digest mismatch')
                if int(len(saved['plan_issue_steps'])) != (1 if branch == 'teacher_fixed' else detail['predictions']):
                    raise ValueError('prediction denominator mismatch')
                if branch == 'network_fixed' and detail['predictions'] != 1:
                    raise ValueError('fixed reference must be predicted once')
            transitions += audit_execution(output, detail, case)['selected_h10_transitions']
            audit_predictions(output, detail, TEACHER / f'teacher-{row["dataset_index"]}.npz')
        verify_initial_predictions(destination, row['dataset_index'])
    calculated = summaries(report['rows'], formal=report['formal'])
    if any(report[key] != value for key, value in calculated.items()):
        raise ValueError('paired metrics/decisions mismatch')
    return dict(status='verified', formal=report['formal'], tasks=len(indices),
        artifacts=len(artifacts), independently_audited_h10_transitions=transitions, **calculated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'precheck', 'smoke', 'run', 'inspect'], required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'prepare':
        result = dict(status='prepared', indices=prepare(args.prepared)['indices'])
    elif args.phase == 'precheck':
        result = dict(status='passed', indices=precheck(args.prepared)['indices'])
    else:
        if args.output is None:
            parser.error('--output required')
        if args.phase == 'inspect':
            result = inspect(args.prepared, args.output)
        else:
            report = run(args.prepared, args.output, smoke=args.phase == 'smoke')
            result = {key: report[key] for key in ('status', 'formal', 'elapsed_seconds', 'metrics', 'comparisons')}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
