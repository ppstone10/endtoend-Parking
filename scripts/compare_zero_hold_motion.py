"""LT39: fixed motion candidates after zero hold; formal runs are human-operated."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from interfaces import VehicleState
from experiments.emitted_output_data import INDICES, case_at, load_cohort, read_json, file_digest, file_identity
from scripts import compare_ungated_refinement as prior
from scripts.compare_feasible_cem import BRANCHES, compare_rows, required_files, verify_initial_predictions
from scripts.diagnose_emitted_execution import PARENT, TEACHER, metric_summary
from scripts.feasible_cem_capture import audit_execution, audit_predictions
from scripts.zero_hold_motion_rescue import ZeroHoldMotionRescueController, PROFILE, OPTIONS, SPEEDS
from scripts.ungated_refinement_mpc import UngatedRefinedCEMController
from scripts.zero_hold_motion_capture import record_rescue_branch, audit_rescue_execution
from scripts.train_emitted_output_joint import StageBudget
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PREVIOUS_PREPARED = ROOT / 'runs/prepared/ungated-refinement-20261029-v1'
PREVIOUS_SHA = '3a146dbcf58096d7c109d59cf53b12a7b45b515a3ffb320958e65380f0d78bb6'
BASELINE = ROOT / 'runs/diagnostics/lt38-ungated-refinement-20261029/results'
PROBES = ROOT / 'runs/verification/zero-motion-rescue-20261029/probes-v1.json'
MOTION_PROBE_SOURCE = ROOT / 'runs/verification/lt38-independent-review-20261029/safe-motion-probes.json'
BASELINES = {
    'lt36': ROOT / 'runs/diagnostics/lt36-feasible-cem-20261029/results',
    'lt37': ROOT / 'runs/diagnostics/lt37-runtime-refinement-20261029/results',
    'lt38': BASELINE,
}
PROTOCOL = dict(max_steps=600, smoke_steps=12, budget_seconds=7200, smoke_seconds=90,
    profile=PROFILE, options=OPTIONS, dt=.1, horizon=10, population=80, iterations=5,
    replan_every=10, seed='dataset_index', required_margin_m=.2, stop_threshold=.5,
    trajectory_dt=.2, network_horizon=320, training_authorized=False,
    geometry_source='simulator_ground_truth', only_change='fixed_bank_after_full_zero',
    supplemental_speeds=list(SPEEDS), supplemental_candidate_limit=8)


def jobs(smoke=False):
    return [dict(index=i, branch=b) for i in (INDICES[:1] if smoke else INDICES) for b in BRANCHES]


def code_identity():
    files = [Path(__file__), ROOT / 'scripts/zero_hold_motion_rescue.py', ROOT / 'scripts/zero_hold_motion_capture.py']
    return {**prior.code_identity(), **{str(p.resolve()): file_digest(p) for p in files}}


def frozen_config():
    if file_digest(PREVIOUS_PREPARED / 'config.json') != PREVIOUS_SHA:
        raise ValueError('LT38 frozen config changed')
    previous = prior.precheck(PREVIOUS_PREPARED)
    report = read_json(BASELINE / 'report.json')
    if (not report['formal'] or report['status'] != 'completed'
            or report['jobs'] != prior.jobs() or len(report['rows']) != 24
            or read_json(BASELINE / 'config.json') != previous):
        raise ValueError('complete frozen LT38 baseline required')
    artifacts = read_json(BASELINE / 'artifacts.json')
    expected = {str((BASELINE / name).resolve()) for name in prior.required_files(INDICES)}
    actual = {str(p.resolve()) for p in BASELINE.iterdir() if p.is_file() and p.name != 'artifacts.json'}
    if set(artifacts) != expected or actual != expected:
        raise ValueError('LT38 baseline artifact denominator mismatch')
    for path, digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError('LT38 baseline artifact changed')
    probe = read_json(PROBES)
    if (probe['status'] != 'passed' or probe['code'] != code_identity()
            or len(probe['rows']) != 12 or probe['rescued_zero_inputs'] != 9):
        raise ValueError('current-code fixed saved-input probes required')
    return dict(schema='zero_motion_rescue_v1', code=code_identity(), protocol=PROTOCOL,
        jobs=jobs(), indices=INDICES, checkpoint=previous['checkpoint'], sources=previous['sources'],
        previous_config=file_identity(PREVIOUS_PREPARED / 'config.json'),
        baseline_artifacts={key:file_identity(path / 'artifacts.json') for key,path in BASELINES.items()},
        preflight=file_identity(PROBES), saved_probe_source=file_identity(MOTION_PROBE_SOURCE),
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
        raise ValueError('frozen code/input/protocol drift; never edit hashes to bypass')
    return config


def summaries(rows, formal):
    expected = jobs(not formal)
    if [(r['dataset_index'], r['branch']) for r in rows] != [(j['index'], j['branch']) for j in expected]:
        raise ValueError('fixed episode denominator/order required')
    metrics = {b:metric_summary([r for r in rows if r['branch']==b]) for b in BRANCHES}
    if not formal:
        return dict(metrics=metrics, decision=dict(status='truncated_smoke_only',
            ready_for_next_validation=False), production_qualified=False, lt34_training_unlocked=False)
    comparisons = {}
    for name,path in BASELINES.items():
        original = read_json(path / 'report.json')['rows']
        if name=='lt36':original=[r[b] for r in original for b in BRANCHES]
        elif name=='lt37':original=[r for r in original if r['profile']=='combined']
        comparisons[name]={b:compare_rows([r for r in original if r['branch']==b],
                                         [r for r in rows if r['branch']==b]) for b in BRANCHES}
    retained = all(not c['lost_successes'] and not c['risky_tasks'] for group in comparisons.values() for c in group.values())
    teacher = metrics['teacher_fixed']['successes']==8
    fixed = metrics['network_fixed']['successes']==8
    timely = all(r['control_timing']['over_100ms']==0 for r in rows)
    ready = bool(retained and teacher and fixed and timely)
    if not retained:status='reject_regression_or_risk'
    elif not teacher:status='stop_teacher_execution_contract_unresolved'
    elif not fixed:status='stop_native_reference_or_progress_contract_unresolved'
    elif not timely:status='stop_compute_budget_failed'
    else:status='bounded_execution_gates_passed_prepare_only_next_validation'
    return dict(metrics=metrics, decision=dict(status=status, comparisons=comparisons,
        all_old_successes_retained=retained, teacher_strict_pass=teacher,
        fixed_success_union_pass=fixed, observed_within_100ms=timely, ready_for_next_validation=ready),
        production_qualified=False, lt34_training_unlocked=False)

def verify_parent_prefix(output, detail):
    stem=f'{detail["dataset_index"]:04d}-{detail["branch"]}'
    saved_detail=read_json(Path(output).with_suffix('.json'))
    interventions=[n for n,s in enumerate(saved_detail['searches']) if s['rescue']['intervened']]
    with np.load(BASELINE / (stem+'.npz'),allow_pickle=False) as old:
        with np.load(Path(output).with_suffix('.npz'),allow_pickle=False) as new:
            for key in ('initial_state','goal','global_plan_0000','local_prediction_0000'):
                np.testing.assert_array_equal(old[key],new[key])
            if interventions:
                shared=interventions[0]
                if shared>=len(old['states_after']):raise ValueError('intervention occurred after baseline already terminated')
                for key in ('states_before','control_windows'):
                    np.testing.assert_array_equal(old[key][:shared+1],new[key][:shared+1])
            else:
                shared=len(new['states_after'])
                if not detail['truncated_preflight'] and shared!=len(old['states_after']):raise ValueError('no-intervention episode changed length')
            for key in ('states_before','states_after','control_windows','selected_control_sequences'):
                np.testing.assert_array_equal(old[key][:shared],new[key][:shared])
    if detail['branch']!='teacher_fixed':
        with np.load(BASELINE / (stem+'-predictions.npz'),allow_pickle=False) as old:
            with np.load(Path(output).parent / (Path(output).name+'-predictions.npz'),allow_pickle=False) as new:
                for key in ('points','logits','bev','goal','motion','pose','lengths'):
                    np.testing.assert_array_equal(old[key][0],new[key][0])

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
        if model.stop_threshold != .5 or model.dt != .2 or model.max_horizon != 320:
            raise ValueError('native decode contract changed')
        for job in report['jobs']:
            budget.check()
            case = case_at(data, manifest, vehicle, INDICES.index(job['index']))
            output = destination / f'{job["index"]:04d}-{job["branch"]}'
            print(f'#{job["index"]} {job["branch"]} BEGIN', flush=True)
            row = record_rescue_branch(case, model, output, budget, branch=job['branch'],
                teacher_archive=TEACHER / f'teacher-{job["index"]}.npz',
                max_steps=PROTOCOL['smoke_steps'] if smoke else PROTOCOL['max_steps'])
            verify_parent_prefix(output, row)
            report['rows'].append(row)
            print(f'#{job["index"]} {job["branch"]} success={row["success"]} collision={row["collision"]} '
                f'margin={row["margin_violation_steps"]} steps={row["steps"]} '
                f'rescues={row["controller_stats"].get("rescue_interventions",0)} p95_ms={row["control_timing"]["p95_ms"]}', flush=True)
            atomic_write_json(destination / 'report.json', report)
        for index in (INDICES[:1] if smoke else INDICES):
            verify_initial_predictions(destination, index)
        precheck(prepared)
        budget.check()
        report.update(status='completed', elapsed_seconds=time.perf_counter()-budget.started,
                      **summaries(report['rows'], formal=not smoke))
        atomic_write_json(destination / 'report.json', report)
        atomic_write_json(destination / 'artifacts.json',
            {str(p.resolve()): file_digest(p) for p in destination.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete', error=str(error), elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(destination / 'report.json', report)
        raise


def inspect(prepared, destination):
    config = precheck(prepared)
    destination = Path(destination).resolve()
    report = read_json(destination / 'report.json')
    smoke = not report['formal']
    if (report['status'] != 'completed' or report['jobs'] != jobs(smoke)
            or len(report['rows']) != len(jobs(smoke)) or read_json(destination / 'config.json') != config
            or report['config_sha256'] != file_digest(Path(prepared) / 'config.json')
            or report['candidate_qualified'] or report['training_performed']
            or report['geometry_source'] != 'simulator_ground_truth'):
        raise ValueError('completion/config/denominator mismatch')
    artifacts = read_json(destination / 'artifacts.json')
    expected = {str((destination / n).resolve()) for n in required_files(INDICES[:1] if smoke else INDICES)}
    actual = {str(p.resolve()) for p in destination.iterdir() if p.is_file() and p.name != 'artifacts.json'}
    if set(artifacts) != expected or actual != expected:
        raise ValueError('missing or extra episode artifacts')
    for path, digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError('episode artifact changed')
    data, manifest, vehicle = load_cohort(PARENT)
    transitions = 0
    for job, row in zip(jobs(smoke), report['rows']):
        case = case_at(data, manifest, vehicle, INDICES.index(job['index']))
        if (row['dataset_index'] != job['index'] or row['branch'] != job['branch']
                or row['task_id'] != case['task_id'] or row['profile'] != PROFILE
                or config['sources'][INDICES.index(job['index'])]['task_id'] != row['task_id']):
            raise ValueError('paired source identity/order mismatch')
        output = destination / f'{job["index"]:04d}-{job["branch"]}'
        detail = read_json(output.with_suffix('.json'))
        if (any(detail[k] != v for k, v in row.items())
                or detail['execution_npz_sha256'] != file_digest(output.with_suffix('.npz'))
                or detail['geometry_source'] != 'simulator_ground_truth' or detail['required_margin_m'] != .2
                or detail['candidate_qualified'] or detail['training_performed']
                or bool(detail['truncated_preflight']) != smoke):
            raise ValueError('episode evidence/summary mismatch')
        transitions += audit_execution(output, detail, case)['selected_h10_transitions']
        audit_rescue_execution(output, detail, case)
        audit_predictions(output, detail, TEACHER / f'teacher-{job["index"]}.npz')
        verify_parent_prefix(output, detail)
    for index in (INDICES[:1] if smoke else INDICES):
        verify_initial_predictions(destination, index)
    calculated = summaries(report['rows'], formal=not smoke)
    if any(report[k] != v for k, v in calculated.items()):
        raise ValueError('paired gate decisions mismatch')
    return dict(status='verified', formal=not smoke, tasks=1 if smoke else 8,
        episodes=len(report['rows']), artifacts=len(artifacts),
        scalar_audited_h10_transitions=transitions, frozen_parent_solve_replays=transitions//10, **calculated)


def run_probes(output):
    output=Path(output)
    if output.exists():raise FileExistsError('new probe path required')
    data,manifest,vehicle=load_cohort(PARENT)
    selection=read_json(MOTION_PROBE_SOURCE)['rows']
    if len(selection)!=12 or len({r['index'] for r in selection})!=2:
        raise ValueError('fixed saved-input denominator changed')
    rows=[]
    for item in selection:
        index,branch,step=item['index'],item['branch'],item['step']
        case=case_at(data,manifest,vehicle,INDICES.index(index))
        stem=BASELINE/f'{index:04d}-{branch}'
        kwargs=dict(goal=case['goal'],terminal=case['engine_kwargs']['terminal'],
            footprint_checker=case['planner']._collision_checker,
            vehicle_model=case['engine_kwargs']['vehicle_model'],dt=.1,horizon=10,
            seed=index,**case['vehicle'].mpc_kwargs())
        parent,new=UngatedRefinedCEMController(**kwargs),ZeroHoldMotionRescueController(**kwargs)
        with np.load(stem.with_suffix('.npz'),allow_pickle=False) as saved:
            state=VehicleState.from_array(saved['states_before'][step]);window=saved['control_windows'][step]
            previous=saved['selected_control_sequences'][step-1].reshape(-1) if step else np.zeros(20)
            warm=new._shift(previous) if step else previous
            archived=saved['selected_control_sequences'][step].reshape(-1)
        for _ in range(step*5):parent._rng.normal(size=(80,20))
        new._rng.bit_generator.state=deepcopy(parent._rng.bit_generator.state)
        base=parent._solve(warm,window,state);selected=new._solve(warm,window,state)
        np.testing.assert_array_equal(base,archived)
        if parent._rng.bit_generator.state!=new._rng.bit_generator.state:raise ValueError('rescue consumed search randomness')
        if new.safety_audit()['generations']!=parent.safety_audit()['generations']:raise ValueError('CEM generations changed')
        zero=bool(np.all(base==0.));audit=new.safety_audit()['rescue']
        if zero:
            if not audit['intervened'] or new.safety_audit()['selected_cost']>=parent.safety_audit()['selected_cost']:
                raise ValueError('known safe lower-cost zero-hold witness not rescued')
        else:
            np.testing.assert_array_equal(selected,base)
            if audit['attempted']:raise ValueError('nonzero parent output intercepted')
        if not new.sequence_safe(selected,state):raise ValueError('rescued H10 unsafe')
        rows.append(dict(index=index,branch=branch,step=step,base_zero=zero,intervened=audit['intervened'],
            parent_cost=parent.safety_audit()['selected_cost'],new_cost=new.safety_audit()['selected_cost'],
            first_command=selected[:2].tolist(),final_h10_safe=True,search_rng_exact=True))
    result=dict(status='passed',code=code_identity(),rows=rows,rescued_zero_inputs=sum(r['base_zero'] for r in rows),
        scope='12 saved inputs from 2 tasks/4 paired branches; counterfactual H10 only, never executed',
        candidate_qualified=False,training_performed=False)
    output.parent.mkdir(parents=True,exist_ok=True);atomic_write_json(output,result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['probe', 'prepare', 'precheck', 'smoke', 'run', 'inspect'], required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'probe':
        if args.output is None: parser.error('--output required')
        r = run_probes(args.output)
        result = dict(status=r['status'], rows=r['rows'])
    else:
        if args.prepared is None: parser.error('--prepared required')
        if args.phase in {'prepare', 'precheck'}:
            r = prepare(args.prepared) if args.phase == 'prepare' else precheck(args.prepared)
            result = dict(status='prepared' if args.phase == 'prepare' else 'passed', jobs=len(r['jobs']))
        else:
            if args.output is None: parser.error('--output required')
            if args.phase == 'inspect': result = inspect(args.prepared, args.output)
            else:
                r = run(args.prepared, args.output, smoke=args.phase == 'smoke')
                result = {k:r[k] for k in ('status','formal','elapsed_seconds','metrics','decision')}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
