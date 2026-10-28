"""LT16：先核验有效数据，再做匹配四臂闭环探索；正式run仅由用户启动。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration import TRAINING_SEEDS, fit_exploration_arm, fixed_frame_metrics, paired_counts, rank_directions
from experiments.behavior_exploration_data import evaluate_cases
from experiments.effective_recovery_data import (ARMS, collect_queries, collect_replay, data_readiness,
                                                 matched_pools, novel_samples, select_sources)
from experiments.exploration_store import ExplorationBudget, ExplorationStore, exploration_lock, verify_unit
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import FIT, PROBE, INPUTS, build_case, frozen_config, matrix
from scripts.prepare_recovery_balance_ab import file_digest, metadata_only
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

PREVIOUS = ROOT / 'runs/diagnostics/learning-exploration-20261021-v1'
PREVIOUS_HASHES = {'config.json': 'a41fc716c469e64098ea16ea36bb4e709145397b95e2ebdb0e4f0172d465bfe8',
                   'report.json': '884777e8b4b41ef6e8434a95b9aab8b9df4d3a061948ad3b089c7704d08085d3',
                   'identity.json': 'd39176510bc7c603acd61845dfc7f4dd12eef3e9908501cc2c4db625c515a00b'}


class IdentityMismatch(ValueError):
    pass


def previous_identity():
    """核验旧存档列出的源，不用新增文件后的当前源集合追认旧版本。"""
    for name, digest in PREVIOUS_HASHES.items():
        if file_digest(PREVIOUS / name) != digest:
            raise IdentityMismatch(f'LT15存档漂移：{name}')
    config = json.loads((PREVIOUS / 'config.json').read_text(encoding='utf-8'))
    for path, digest in config['code'].items():
        if file_digest(path) != digest:
            raise IdentityMismatch(f'LT15冻结源被改写：{path}')
    for item in list(config['inputs'].values()) + list(config['expected_R0_arrays'].values()):
        if file_digest(item['path']) != item['sha256']:
            raise IdentityMismatch(f'LT15冻结输入漂移：{item["path"]}')
    units = ['collect'] + [f'{name}-{seed}' for name in ('dense_joint', 'aggregate-labels') for seed in TRAINING_SEEDS]
    unit_hashes = {}
    for unit in units:
        path = PREVIOUS / 'units' / unit / 'completed.json'
        if not path.is_file():
            raise IdentityMismatch(f'LT15前置单元缺失：{unit}')
        unit_hashes[unit] = file_digest(path)
    return config, unit_hashes


def make_config(smoke=False):
    previous, units = previous_identity()
    config = frozen_config(refine_teacher=True, rolling_s4=True)
    metadata = metadata_only(INPUTS['train'])
    reserved = collect_metadata_ids(metadata_only(INPUTS['val'])) | collect_metadata_ids(metadata_only(INPUTS['selection']))
    groups = select_sources(metadata, previous['confirmation_indices'], reserved)
    config.update(kind='effective_replay_v1', smoke=smoke, previous_manifest_hashes=units,
                  previous_identity=PREVIOUS_HASHES, groups=groups,
                  old_confirmation=previous['confirmation_indices'], arms=list(ARMS),
                  training_seeds=list(TRAINING_SEEDS), query_policies=['R0', 'C240-20261020', 'C240-20261021'],
                  snapshots=[2] if smoke else [12, 60, 240], selection_steps=[2] if smoke else [60, 240],
                  steps=2 if smoke else 240, optimizer_steps=2 if smoke else 240,
                  control_seeds=[0] if smoke else [0, 1],
                  budget_seconds=180 if smoke else 5400, max_queries_per_source_policy=3,
                  max_confirmation_directions=2, automatic_deployment=False,
                  sample_identity='float32-field-shape-bytes-v1', source_tasks={str(i): metadata[i]['task_id']
                    for i in sorted(set(FIT + PROBE + previous['confirmation_indices'] +
                                        groups['fresh_confirmation'] + groups['recovery_sources'] + groups['replay_sources']))})
    config['code'][str(Path(__file__).resolve())] = file_digest(__file__)
    # 三个既有助手脚本不在code_identity的模块目录中，显式冻结。
    for name in ('explore_closed_loop_learning.py', 'prepare_recovery_balance_ab.py', 'build_selection_dataset.py'):
        path = ROOT / 'scripts' / name
        config['code'][str(path)] = file_digest(path)
    return config


def prepare(output, smoke=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('新探索目录必须不存在，不覆盖旧产物')
    config = make_config(smoke)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / 'config.json', config)
    atomic_write_json(output / 'identity.json', {'config_sha256': file_digest(output / 'config.json')})
    print(f'PREPARED {output}; NO FORMAL ROLLOUT STARTED', flush=True)


def verify_identity(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    identity = json.loads((output / 'identity.json').read_text(encoding='utf-8'))
    if file_digest(output / 'config.json') != identity['config_sha256'] or make_config(config['smoke']) != config:
        raise IdentityMismatch('配置/旧基线/输入/源/环境漂移，禁止续跑旧目录')
    return config


def precheck(output):
    config = verify_identity(output)
    # 只加载本地、已哈希且由本项目生成的可信torch产物。
    for unit in config['previous_manifest_hashes']:
        if verify_unit(PREVIOUS / 'units' / unit) is None:
            raise IdentityMismatch(f'旧完成产物缺失：{unit}')
    for directory in (Path(output) / 'units').glob('*'):
        if directory.is_dir():
            verify_unit(directory)
    print('IDENTITY AND ARTIFACTS VERIFIED; NO ROLLOUT STARTED', flush=True)


def train_arm(directory, arm, seed, samples, common, parent, cases, config, budget):
    snapshots = {}
    def checkpoint(step, model, optimizer, history):
        print(f'CHECKPOINT {arm} seed={seed} step={step}', flush=True)
        point = directory / f'step{step:04d}'; point.mkdir()
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'threshold': model.stop_threshold, 'step': step, 'purpose': 'diagnostic_not_deployment'}, point / 'state.pt')
        fit_indices = FIT[:1] if config['smoke'] else FIT + config['groups']['recovery_sources']
        probe_indices = PROBE[:1] if config['smoke'] else PROBE
        row = {'arm': arm, 'training_seed': seed, 'step': step, 'checkpoint': str(point / 'state.pt'),
               'fit': evaluate_cases(fit_indices, cases, model, budget, point, 'fit'),
               'probe': evaluate_cases(probe_indices, cases, model, budget, point, 'probe'),
               'fixed_metrics': fixed_frame_metrics(model, common, budget.check), 'loss_history': history}
        snapshots[step] = row; atomic_write_json(point / 'report.json', row)
    fit_exploration_arm(parent, samples, seed=seed, steps=config['steps'], snapshots=config['snapshots'],
                        check_budget=budget.check, checkpoint=checkpoint)
    return snapshots


def confirm(store, selected, directions, parent, cases, config, budget):
    results = []
    for group, indices in [('old_regression', config['old_confirmation']),
                           ('fresh_confirmation', config['groups']['fresh_confirmation'])]:
        indices = indices[:1] if config['smoke'] else indices
        baseline = {}
        for control in config['control_seeds']:
            key = f'confirm-{group}-r0-{control}'
            baseline[control], _ = store.execute(key, lambda p: evaluate_cases(indices, cases, parent, budget, p, 'R0', control))
        for direction in selected:
            name = direction['direction']; pairs = []
            for seed in TRAINING_SEEDS:
                model, _ = load_candidate(directions[name][seed]['checkpoint'], parent)
                for control in config['control_seeds']:
                    key = f'confirm-{group}-{name}-{seed}-{control}'
                    rows, _ = store.execute(key, lambda p: evaluate_cases(indices, cases, model, budget, p, 'candidate', control))
                    pairs.append({'training_seed': seed, 'control_seed': control, 'rows': rows,
                                  'paired': paired_counts(baseline[control], rows)})
            results.append({'group': group, 'direction': name, 'pairs': pairs, 'baseline': baseline,
                            'passed': not config['smoke'] and len(pairs) == 4 and
                                      all(p['paired']['confirmation_passed'] for p in pairs)})
    return results


def factorial_contrasts(directions, steps):
    result = []
    for step in steps:
        for factor, left, right in [('replay_with_old_recovery', 'old_old', 'broad_old'),
                                     ('replay_with_novel_recovery', 'old_novel', 'broad_novel'),
                                     ('recovery_with_old_replay', 'old_old', 'old_novel'),
                                     ('recovery_with_broad_replay', 'broad_old', 'broad_novel')]:
            pairs = []
            for seed in TRAINING_SEEDS:
                a = directions.get(f'{left}-{step}', {}).get(seed)
                b = directions.get(f'{right}-{step}', {}).get(seed)
                if a and b:
                    pairs.append({'training_seed': seed, 'fit': paired_counts(a['fit'], b['fit']),
                                  'probe': paired_counts(a['probe'], b['probe'])})
            result.append({'factor': factor, 'step': step, 'pairs': pairs})
    return result


def run(output, smoke=False):
    output = Path(output).resolve(); config = verify_identity(output)
    if config['smoke'] != smoke:
        raise ValueError('正式/冒烟目录不可混跑')
    precheck(output)
    torch.set_num_threads(1)
    progress = output / 'progress.json'
    used = json.loads(progress.read_text(encoding='utf-8'))['active_elapsed_seconds'] if progress.exists() else 0.
    budget = ExplorationBudget(config['budget_seconds'], used=used, progress_path=progress)
    with exploration_lock(output):
        store = ExplorationStore(output, lambda: verify_identity(output), budget)
        report = {'status': 'running', 'smoke': smoke, 'branch_failures': [], 'main_protocol_passed': False,
                  'automatic_deployment': False, 'config_sha256': file_digest(output / 'config.json')}
        def save():
            report['active_elapsed_seconds'] = budget.elapsed(); atomic_write_json(output / 'report.json', report)
        def execute(key, operation):
            try:
                payload, _ = store.execute(key, operation)
                return payload
            except IdentityMismatch:
                raise
            except ValueError as exc:
                report['branch_failures'].append({'unit': key, 'error': str(exc)})
                save(); return None
        save()
        try:
            collected, _ = verify_unit(PREVIOUS / 'units/collect')
            data = DatasetGenerator.load(INPUTS['train']); metadata = data['task_meta']
            manifest = matrix.load_dataset_manifest(INPUTS['train']); vehicle = VehicleConfig(**manifest['vehicle_model'])
            parent = load_model_checkpoint(INPUTS['R0']).model
            models = {'R0': parent}
            for seed in TRAINING_SEEDS:
                payload, _ = verify_unit(PREVIOUS / 'units' / f'dense_joint-{seed}')
                models[f'C240-{seed}'], _ = load_candidate(payload['final']['checkpoint'], parent)
            groups = config['groups']
            replay_indices = groups['replay_sources'] if not smoke else [0, 1720]
            query_indices = groups['recovery_sources'] if not smoke else [1559, 2330]
            needed = FIT + PROBE + replay_indices + query_indices + config['old_confirmation'] + groups['fresh_confirmation']
            cases = {i: build_case(data, manifest, i, vehicle) for i in sorted(set(needed))}
            replay = []; proposed = []; query_rows = []; replay_rows = []
            for index in replay_indices:
                payload = execute(f'replay-{index}', lambda p: collect_replay(cases[index], parent, budget, p))
                if payload:
                    replay.extend(payload['samples']); replay_rows.append(payload['row'])
            for index in query_indices:
                for policy in config['query_policies'] if not smoke else ['C240-20261020']:
                    # 冒烟固定复核既知S4非零起点，不据此改变正式均匀采样规则。
                    steps = [320] if smoke and index == 1559 else None
                    payload = execute(f'query-{index}-{policy.lower()}', lambda p: collect_queries(
                        cases[index], data, models[policy], parent, policy, budget, p, steps=steps))
                    if payload:
                        proposed.extend(payload['samples']); query_rows.append(dict(source_index=index, **{k: v for k, v in payload.items() if k != 'samples'}))
            novel, novelty = novel_samples(collected['dense'], proposed)
            _, duplicate_audit = novel_samples(collected['dense'], collected['dense'])
            historical_aggregation = []
            for seed in TRAINING_SEEDS:
                old_extra, _ = verify_unit(PREVIOUS / 'units' / f'aggregate-labels-{seed}')
                _, audit = novel_samples(collected['dense'], old_extra['extra'])
                historical_aggregation.append({'training_seed': seed, **audit})
                if audit['accepted_new_inputs'] != 0 or audit['novel_pairs_vs_old'] != 0:
                    raise IdentityMismatch('LT15聚合零新增反例未复现，不继续训练')
            replay, replay_audit = novel_samples(collected['dense'], replay)
            readiness = data_readiness(replay, novel, metadata)
            report.update(novelty=novelty, replay_novelty=replay_audit, duplicate_regression=duplicate_audit,
                          data_readiness=readiness, replay_rows=replay_rows, query_rows=query_rows,
                          historical_aggregation_audit=historical_aggregation,
                          formal_data_gate_passed=readiness['passed'])
            save()
            if not smoke and (not readiness['passed'] or report['branch_failures']):
                report.update(status='completed_data_gate_failed', decision='no_training_review_coverage_and_teacher_failures',
                              training_performed=False, all_predeclared_collections_attempted=True)
                verify_identity(output); save(); print(f'FINISHED {report["decision"]}', flush=True); return report
            if smoke and (not replay or not novel):
                raise ValueError('真实回放/非零新增查询为空，冒烟不得借旧数据伪造通过')
            pools = matched_pools(collected['dense'], replay, novel)
            common = pools['broad_novel'] if not smoke else [pools['broad_novel'][0], novel[0]]
            directions = {}; snapshots = {}
            for arm in ARMS:
                for seed in TRAINING_SEEDS:
                    payload = execute(f'fit-{arm}-{seed}', lambda p: train_arm(
                        p, arm, seed, pools[arm], common, parent, cases, config, budget))
                    if payload:
                        snapshots[f'{arm}-{seed}'] = payload
                        for step in config['selection_steps']:
                            directions.setdefault(f'{arm}-{step}', {})[seed] = payload[step]
            baseline = collected['baseline_probe'][:1] if smoke else collected['baseline_probe']
            selected = rank_directions(directions, baseline)
            report.update(training_performed=True, selected_directions=selected,
                          factorial_contrasts=factorial_contrasts(directions, config['selection_steps']), snapshots=snapshots)
            save()
            confirmation = confirm(store, selected, directions, parent, cases, config, budget)
            qualified = [d['direction'] for d in selected if d['discovery_eligible'] and
                         all(r['passed'] for r in confirmation if r['direction'] == d['direction'])]
            report.update(status='completed_with_branch_failures' if report['branch_failures'] else 'completed',
                          confirmation=confirmation, qualified_directions=qualified,
                          decision='smoke_only_no_benefit_claim' if smoke else
                                   'stage_candidate_for_review' if qualified and not report['branch_failures'] else 'no_qualified_direction',
                          all_predeclared_branches_completed=not report['branch_failures'] and len(snapshots) == 8)
            verify_identity(output); save(); print(f'FINISHED {report["decision"]}', flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else
                          'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=str(exc), decision='incomplete_do_not_infer_benefit')
            save(); raise
        finally:
            budget.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'precheck', 'smoke', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    elif args.action == 'precheck':
        precheck(args.output)
    elif args.action == 'smoke':
        prepare(args.output, smoke=True); run(args.output, smoke=True)
    else:
        run(args.output)
