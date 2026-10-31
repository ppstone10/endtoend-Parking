"""LT40: frozen 125-task rolling retention pair; formal execution is human-operated."""
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
from dataset import DatasetGenerator
from experiments.emitted_output_data import read_json, file_digest, file_identity, regression_cohort
from scripts import compare_zero_hold_motion as lt39
from scripts.diagnose_emitted_execution import metric_summary
from scripts.preflight_behavior_repair import build_case
from scripts.prepare_recovery_balance_ab import metadata_only, runtime_identity
from scripts.train_emitted_output_joint import StageBudget
from scripts.zero_motion_retention_capture import (
    ARMS, record_pair_arm, audit_pair_arm, audit_paired_prefix, audit_forward)
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PARENT = ROOT / 'runs/prepared/emitted-joint-20261028-v1'
LT39_PREPARED = ROOT / 'runs/prepared/zero-motion-rescue-20261029-v1'
LT39_SHA = 'c99e5054bc5e2367b531c2781aaebd442b88632ae59830eecd705db56c3fa12c'
LT39_RESULTS = ROOT / 'runs/diagnostics/lt39-zero-motion-rescue-20261029/results'
VAL_MANIFEST = ROOT / 'runs/verification/surround-target-rebuild-lineage-20261027/val/manifest.json'
LT32_CONFIG = ROOT / 'runs/verification/strict-footprint-formal-20261028/results/config.json'
SMOKE_INDICES = [0, 183]
PROTOCOL = dict(max_steps=600, smoke_steps=12, smoke_seconds=90, budget_seconds=14400,
    branch='network_rolling', replan_every=10, seed='validation_dataset_index', dt=.1,
    horizon=10, population=80, iterations=5, stop_threshold=.5, network_horizon=320,
    trajectory_dt=.2, required_margin_m=.2, geometry_source='simulator_ground_truth',
    only_algorithm_change='fixed_bank_after_full_zero', training_performed=False,
    historical_control_seed=0, historical_comparison='task_success_retention_only')


def validate_cohort(regression, metadata):
    """Retain overlapping group memberships; never count episodes as unique tasks."""
    expected_sizes = dict(regressions=22, recoveries=8, residual=5, native_success=93)
    groups = regression['groups']
    if (len(metadata) != 300 or set(groups) != set(expected_sizes)
            or any(len(groups[k]) != n or len(set(groups[k])) != n for k,n in expected_sizes.items())):
        raise ValueError('historical group denominators changed')
    union = sorted(set().union(*groups.values()))
    ids = [metadata[i]['task_id'] for i in union]
    if (len(union) != 125 or len(set(ids)) != 125 or regression['unique_tasks'] != 125
            or regression['indices'] != union or regression['task_ids'] != ids
            or not regression['evaluation_only'] or regression['independent_unseen_test']):
        raise ValueError('frozen validation identities/union changed')
    retained = set(groups['native_success']) | set(groups['regressions']) | set(groups['recoveries'])
    if len(retained) != 120 or retained & set(groups['residual']):
        raise ValueError('historical retention/residual partition changed')
    return [dict(index=i, task_id=metadata[i]['task_id'], scene_name=metadata[i]['scene_name'],
                 groups=[key for key,values in groups.items() if i in values]) for i in union]


def code_identity():
    files = [Path(__file__), ROOT/'scripts/zero_motion_retention_capture.py']
    return {**lt39.code_identity(), **{str(p.resolve()):file_digest(p) for p in files}}


def verify_artifacts(destination, expected):
    destination = Path(destination).resolve()
    artifacts = read_json(destination/'artifacts.json')
    paths = {str((destination/name).resolve()) for name in expected}
    actual = {str(p.resolve()) for p in destination.iterdir() if p.is_file() and p.name != 'artifacts.json'}
    if set(artifacts) != paths or actual != paths:
        raise ValueError('missing or extra artifacts')
    for path,digest in artifacts.items():
        if file_digest(path) != digest:
            raise ValueError('artifact digest drift')
    return len(artifacts)


def frozen_config(prepared):
    prepared = Path(prepared).resolve()
    if file_digest(LT39_PREPARED/'config.json') != LT39_SHA:
        raise ValueError('LT39 prepared identity drift')
    previous = lt39.precheck(LT39_PREPARED)
    verify_artifacts(LT39_RESULTS, lt39.required_files(lt39.INDICES))
    report = read_json(LT39_RESULTS/'report.json')
    if (report['status'] != 'completed' or not report['formal'] or report['jobs'] != lt39.jobs()
            or report['config_sha256'] != LT39_SHA or read_json(LT39_RESULTS/'config.json') != previous
            or len(report['rows']) != 24 or not report['decision']['ready_for_next_validation']
            or any(report[k] != v for k,v in lt39.summaries(report['rows'], True).items())):
        raise ValueError('verified complete passing LT39 required')
    original = read_json(PARENT/'config.json')
    for key in ('val','lt32','lt33','parent'):
        if file_identity(original['inputs'][key]['path']) != original['inputs'][key]:
            raise ValueError('original evaluation input drift')
    baseline = read_json(LT32_CONFIG)
    if (baseline['inputs']['data'] != original['inputs']['val']
            or baseline['inputs']['checkpoint'] != previous['checkpoint']
            or baseline['inputs']['manifest'] != file_identity(VAL_MANIFEST)):
        raise ValueError('historical task/model provenance mismatch')
    metadata = metadata_only(original['inputs']['val']['path'])
    regression = regression_cohort(read_json(original['inputs']['lt33']['path']),
        read_json(original['inputs']['lt32']['path']), metadata)
    if regression != original['regression']:
        raise ValueError('LT34 frozen cohort changed')
    sources = validate_cohort(regression, metadata)
    return dict(schema='zero_motion_retention_v1', code=code_identity(), runtime=runtime_identity(),
        protocol=PROTOCOL, regression=regression, sources=sources,
        checkpoint=previous['checkpoint'],
        inputs={key:original['inputs'][key] for key in ('val','lt32','lt33')},
        historical_config=file_identity(LT32_CONFIG), manifest=file_identity(VAL_MANIFEST),
        parent_config=file_identity(PARENT/'config.json'), previous_config=file_identity(LT39_PREPARED/'config.json'),
        lt39_artifacts=file_identity(LT39_RESULTS/'artifacts.json'),
        cohort=file_identity(prepared/'cohort.npz'),
        candidate_qualified=False, training_authorized=False, production_qualified=False)


def prepare(prepared):
    prepared = Path(prepared).resolve()
    if prepared.exists():
        raise FileExistsError('new prepared directory required; no overwrite')
    original = read_json(PARENT/'config.json')
    indices = original['regression']['indices']
    prepared.mkdir(parents=True, exist_ok=False)
    # Save only the already frozen validation union, preserving original indices separately.
    with np.load(original['inputs']['val']['path'], allow_pickle=False) as archive:
        subset = {k:archive[k][indices] if k in {'bevs','goals','states','trajs','masks','task_meta'}
                  else archive[k] for k in archive.files}
    np.savez_compressed(prepared/'cohort.npz', **subset)
    config = frozen_config(prepared)
    atomic_write_json(prepared/'config.json', config)
    return config


def precheck(prepared):
    if Path(sys.executable).resolve() != Path('D:/conda/envs/endtoend-parking/python.exe').resolve():
        raise ValueError('endtoend-parking environment required')
    config = read_json(Path(prepared)/'config.json')
    if config != frozen_config(prepared):
        raise ValueError('frozen code/runtime/data/protocol drift; never rewrite hashes to bypass')
    return config


def jobs(config, smoke=False):
    indices = SMOKE_INDICES if smoke else config['regression']['indices']
    return [dict(index=i, arm=arm) for i in indices for arm in ARMS]


def required_files(config, smoke=False):
    names = ['config.json','report.json']
    for job in jobs(config, smoke):
        stem = f'{job["index"]:04d}-{job["arm"]}'
        names.extend([stem+'.json', stem+'.npz', stem+'-predictions.npz'])
    return names


def summarize(rows, config, *, formal):
    expected = jobs(config, not formal)
    sources = {s['index']:s for s in config['sources']}
    if [(r['dataset_index'],r['arm']) for r in rows] != [(j['index'],j['arm']) for j in expected]:
        raise ValueError('fixed paired denominator/order required')
    if any(r['task_id'] != sources[r['dataset_index']]['task_id'] for r in rows):
        raise ValueError('task identity mismatch')
    metrics = {arm:metric_summary([r for r in rows if r['arm']==arm]) for arm in ARMS}
    flags = dict(production_qualified=False, lt34_training_unlocked=False, candidate_qualified=False)
    if not formal:
        return dict(metrics=metrics, decision=dict(status='truncated_smoke_only',
            ready_for_next_validation=False), **flags)
    parent = {r['dataset_index']:r for r in rows if r['arm']=='parent'}
    candidate = {r['dataset_index']:r for r in rows if r['arm']=='candidate'}
    lost = [i for i in parent if parent[i]['success'] and not candidate[i]['success']]
    gained = [i for i in parent if not parent[i]['success'] and candidate[i]['success']]
    historical = set().union(*(config['regression']['groups'][k] for k in
        ('native_success','regressions','recoveries')))
    historical_lost = sorted(i for i in historical if not candidate[i]['success'])
    risky = [i for i,r in candidate.items() if r['collision'] or r['margin_violation_steps']
             or not r['initial_contact_safe'] or not r['initial_margin_safe']]
    slow = [i for i,r in candidate.items() if r['control_timing']['over_100ms']]
    clean = not (lost or historical_lost or risky or slow)
    status = ('reject_regression_risk_or_latency' if not clean else
              'retention_passed_no_increment' if not gained else
              'bounded_retention_and_increment_passed_prepare_only_next_validation')
    groups = {}
    for name,indices in config['regression']['groups'].items():
        groups[name] = {arm:metric_summary([r for r in rows if r['arm']==arm and r['dataset_index'] in indices])
                        for arm in ARMS}
        groups[name]['candidate_failed'] = [i for i in indices if not candidate[i]['success']]
    scenes = {scene:{arm:metric_summary([r for r in rows if r['arm']==arm and r['scene_name']==scene])
                    for arm in ARMS} for scene in sorted({r['scene_name'] for r in rows})}
    return dict(metrics=metrics, groups=groups, scenes=scenes, decision=dict(status=status,
        lost_parent_successes=lost, gained_parent_failures=gained, lost_historical_successes=historical_lost,
        candidate_risky_tasks=risky, candidate_slow_tasks=slow, historical_success_union=len(historical),
        retention_passed=clean, ready_for_next_validation=bool(clean and gained)), **flags)


def load_cases(config):
    data = DatasetGenerator.load(config['cohort']['path'])
    manifest = read_json(config['manifest']['path'])
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    if len(data['task_meta']) != 125:
        raise ValueError('cohort size changed')
    return data, manifest, vehicle


def case_at(data, manifest, vehicle, config, index):
    row = config['regression']['indices'].index(index)
    source = config['sources'][row]
    if data['task_meta'][row]['task_id'] != source['task_id']:
        raise ValueError('subset/original index identity mismatch')
    case = build_case(data, manifest, row, vehicle)
    case['index'] = index  # Never use the subset ordinal for the CEM RNG.
    return case


def native_model(config):
    model = load_model_checkpoint(config['checkpoint']['path']).model
    model.eval()
    if model.stop_threshold != .5 or model.dt != .2 or model.max_horizon != 320:
        raise ValueError('native decoder contract drift')
    return model


def run(prepared, destination, *, smoke=False):
    config = precheck(prepared)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    budget = StageBudget(PROTOCOL['smoke_seconds'] if smoke else PROTOCOL['budget_seconds'])
    report = dict(status='running', formal=not smoke, jobs=jobs(config,smoke), rows=[],
        config_sha256=file_digest(Path(prepared)/'config.json'), training_performed=False,
        geometry_source='simulator_ground_truth')
    atomic_write_json(destination/'config.json',config)
    atomic_write_json(destination/'report.json',report)
    try:
        data,manifest,vehicle = load_cases(config)
        model = native_model(config)
        for job in report['jobs']:
            budget.check()
            case = case_at(data,manifest,vehicle,config,job['index'])
            stem = destination / f'{job["index"]:04d}-{job["arm"]}'
            print(f'#{job["index"]} {job["arm"]} BEGIN',flush=True)
            row = record_pair_arm(case,model,stem,budget,arm=job['arm'],
                max_steps=PROTOCOL['smoke_steps'] if smoke else PROTOCOL['max_steps'])
            report['rows'].append(row)
            if job['arm']=='candidate':
                audit_paired_prefix(destination/f'{job["index"]:04d}-parent',stem)
            print(f'#{job["index"]} {job["arm"]} success={row["success"]} '
                f'collision={row["collision"]} margin={row["margin_violation_steps"]} '
                f'steps={row["steps"]} rescue={row["controller_stats"].get("rescue_interventions",0)}',flush=True)
            atomic_write_json(destination/'report.json',report)
        precheck(prepared)
        budget.check()
        report.update(status='completed',elapsed_seconds=time.perf_counter()-budget.started,
            **summarize(report['rows'],config,formal=not smoke))
        atomic_write_json(destination/'report.json',report)
        atomic_write_json(destination/'artifacts.json',
            {str(p.resolve()):file_digest(p) for p in destination.iterdir() if p.is_file()})
        return report
    except BaseException as error:
        report.update(status='incomplete',error=str(error),elapsed_seconds=time.perf_counter()-budget.started)
        atomic_write_json(destination/'report.json',report)
        raise


def inspect(prepared,destination):
    config = precheck(prepared)
    destination = Path(destination).resolve()
    report = read_json(destination/'report.json')
    smoke = not report['formal']
    if (report['status']!='completed' or report['jobs']!=jobs(config,smoke)
            or len(report['rows'])!=len(jobs(config,smoke)) or read_json(destination/'config.json')!=config
            or report['config_sha256']!=file_digest(Path(prepared)/'config.json')
            or report['training_performed'] or report['geometry_source']!='simulator_ground_truth'):
        raise ValueError('completion/config/denominator mismatch')
    files = verify_artifacts(destination, required_files(config,smoke))
    data,manifest,vehicle = load_cases(config)
    model = native_model(config)
    budget = StageBudget(PROTOCOL['smoke_seconds'] if smoke else PROTOCOL['budget_seconds'])
    transitions, predictions = 0, 0
    for job,row in zip(jobs(config,smoke),report['rows']):
        budget.check()
        case = case_at(data,manifest,vehicle,config,job['index'])
        stem = destination/f'{job["index"]:04d}-{job["arm"]}'
        detail = read_json(stem.with_suffix('.json'))
        if (any(detail.get(k)!=v for k,v in row.items()) or row['task_id']!=case['task_id']
                or row['dataset_index']!=job['index'] or row['arm']!=job['arm']
                or bool(row['truncated_preflight'])!=smoke
                or detail['geometry_source']!='simulator_ground_truth' or detail['required_margin_m']!=.2
                or detail['candidate_qualified'] or detail['training_performed']
                or detail['execution_npz_sha256']!=file_digest(stem.with_suffix('.npz'))):
            raise ValueError('episode identity/evidence drift')
        transitions += audit_pair_arm(stem,detail,case)['selected_h10_transitions']
        predictions += audit_forward(stem,model,budget)
        if job['arm']=='candidate':
            audit_paired_prefix(destination/f'{job["index"]:04d}-parent',stem)
        if not smoke:
            print(f'AUDIT #{job["index"]} {job["arm"]} transitions={transitions} predictions={predictions}',flush=True)
    calculated = summarize(report['rows'],config,formal=not smoke)
    if any(report.get(k)!=v for k,v in calculated.items()):
        raise ValueError('acceptance summary drift')
    return dict(status='verified',formal=not smoke,tasks=2 if smoke else 125,
        episodes=len(report['rows']),artifacts=files,scalar_audited_h10_transitions=transitions,
        full_network_forward_replays=predictions,**calculated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['prepare','precheck','smoke','run','inspect'])
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase in {'prepare','precheck'}:
        config = prepare(args.prepared) if args.phase=='prepare' else precheck(args.prepared)
        result = dict(status='prepared' if args.phase=='prepare' else 'passed',
                      tasks=len(config['sources']),episodes=len(jobs(config)),source_files=len(config['code']))
    else:
        if args.output is None:
            parser.error('--output required')
        if args.phase=='inspect':
            result = inspect(args.prepared,args.output)
        else:
            report = run(args.prepared,args.output,smoke=args.phase=='smoke')
            result = {key:report[key] for key in ('status','formal','elapsed_seconds','metrics','decision')}
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
