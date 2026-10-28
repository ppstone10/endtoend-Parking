"""默认180秒保旧短检；完整诊断须显式准备并由用户运行，不增加训练量。"""
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
from experiments.behavior_exploration_data import cache_samples, evaluate_cases, run_case, save_record
from experiments.effective_recovery_data import matched_pools, novel_samples, scene_number
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import ExplorationBudget, exploration_lock, verify_unit
from experiments.recovery_execution import record_arrays
import experiments.retention_sampling as sampling
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.explore_closed_loop_learning import load_candidate
from scripts.explore_effective_replay import PREVIOUS, previous_identity
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix, result_row
from scripts.prepare_recovery_balance_ab import file_digest, metadata_only, runtime_identity
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

LT16 = ROOT / 'runs/diagnostics/effective-replay-20261021-v2'
LT16_HASHES = {'config.json': '368718182c5f8fbfef43ef68ea278ae87583d069245dd78606b3ed56e65cb088',
               'report.json': 'd2d001ff1b9a8c6da30df2444f15e5bef061a72e701efb3254f340e915219d03'}
PARTIAL = ROOT / 'runs/diagnostics/retention-sampling-preflight-20261021-v2'
PARTIAL_HASHES = {'config.json': 'e2f99ef21cea2bc9e7234005ca0b260c957736b2a0fef6d903087872244bd946',
                  'report.json': '704f9736f971087343cc7aed4955088d341d5f82e54d59561a520b12d2ce6866'}
ARMS = {'flat_old': ('old', 'flat'), 'balanced_old': ('old', 'hierarchy'),
        'flat_augmented': ('augmented', 'flat'), 'balanced_augmented': ('augmented', 'hierarchy')}


class FrozenIdentityError(RuntimeError):
    pass


def verify_files(files):
    for path, digest in files.items():
        if not Path(path).is_file() or file_digest(path) != digest:
            raise FrozenIdentityError(f'冻结身份漂移：{path}')


def make_config(*, complete_diagnostic=False):
    verify_files({str(LT16 / name): digest for name, digest in LT16_HASHES.items()})
    old = json.loads((LT16 / 'config.json').read_text(encoding='utf-8'))
    verify_files(old['code'])
    if runtime_identity() != old['runtime']:
        raise FrozenIdentityError('运行环境漂移')
    _, previous_units = previous_identity()
    if previous_units != old['previous_manifest_hashes']:
        raise FrozenIdentityError('LT15前置单元漂移')
    inputs = {item['path']: item['sha256'] for item in old['inputs'].values()}; verify_files(inputs)
    metadata = metadata_only(INPUTS['train'])
    forbidden = collect_metadata_ids([metadata[int(i)] for i in old['source_tasks']])
    forbidden |= collect_metadata_ids(metadata_only(INPUTS['val'])) | collect_metadata_ids(metadata_only(INPUTS['selection']))
    groups = sampling.select_groups(metadata, forbidden, gap_counts=(4, 12))
    files = {**inputs, **{str(LT16 / name): digest for name, digest in LT16_HASHES.items()}}
    if complete_diagnostic:
        partial_files = {str(PARTIAL / name): digest for name, digest in PARTIAL_HASHES.items()}
        verify_files(partial_files); files.update(partial_files)
        partial = json.loads((PARTIAL / 'config.json').read_text(encoding='utf-8'))
        if groups != partial['groups']:
            raise FrozenIdentityError('原来源/6探针变化，不是同一诊断完整化')
        excluded = forbidden | collect_metadata_ids([metadata[i] for value in groups.values() for i in value])
        groups['regression_candidates'] = sampling.regression_candidates(metadata, excluded)
    keys = [f'replay-{i}' for i in old['groups']['replay_sources']]
    keys += [f'query-{i}-{policy.lower()}' for i in old['groups']['recovery_sources'] for policy in old['query_policies']]
    keys += [f'fit-broad_novel-{seed}' for seed in sampling.TRAINING_SEEDS]
    for key in keys:
        path = LT16 / 'units' / key / 'completed.json'; files[str(path)] = file_digest(path)
    path = PREVIOUS / 'units/collect/completed.json'; files[str(path)] = file_digest(path)
    code = {**old['code'], str(Path(sampling.__file__).resolve()): file_digest(sampling.__file__),
            str(Path(__file__).resolve()): file_digest(__file__)}
    return {'kind': 'retention_sampling_diagnostic_v1' if complete_diagnostic else 'retention_sampling_preflight_v3',
            'runtime': runtime_identity(), 'code': code,
            'frozen_files': files, 'groups': groups, 'arms': ARMS, 'steps': 24, 'start_step': 240,
            'collection_scope': '一次S6有限扩源：4扩12，保留首4失败；不再自动增加',
            'training_seeds': list(sampling.TRAINING_SEEDS), 'budget_seconds': 900 if complete_diagnostic else 180,
            'purpose': '诊断配对，不是已有效的长训；完整模式不增加每臂24更新',
            'control_seed': 0, 'automatic_deployment': False, 'main_protocol_passed': False,
            'parent_checkpoints': {str(seed): json.loads((LT16 / 'report.json').read_text(encoding='utf-8'))[
                'snapshots'][f'broad_novel-{seed}']['240']['checkpoint'] for seed in sampling.TRAINING_SEEDS}}


def load_original_pool(old_config, budget):
    old, _ = verify_unit(PREVIOUS / 'units/collect'); replay = []; queries = []; directories = {}
    for index in old_config['groups']['replay_sources']:
        budget.check(); payload, directory = verify_unit(LT16 / 'units' / f'replay-{index}')
        replay.extend(payload['samples']); directories[index] = directory
    for index in old_config['groups']['recovery_sources']:
        for policy in old_config['query_policies']:
            budget.check(); payload, _ = verify_unit(LT16 / 'units' / f'query-{index}-{policy.lower()}')
            queries.extend(payload['samples'])
    replay, _ = novel_samples(old['dense'], replay); queries, _ = novel_samples(old['dense'], queries)
    samples = matched_pools(old['dense'], replay, queries)['broad_novel']
    if len(samples) != 390:
        raise FrozenIdentityError('LT16联合池390帧未复现')
    return samples, directories


def collect_anchor(case, parent, budget, directory, expected=None):
    result, engine, frames = run_case(case, parent, budget); save_record(directory, 'R0', result)
    if expected is not None:
        states, commands = record_arrays(result)
        with np.load(expected) as archive:
            if not np.array_equal(states, archive['states']) or not np.array_equal(commands, archive['commands']):
                raise FrozenIdentityError(f'旧回放源#{case["index"]}未逐元素复现')
    _, full = cache_samples(frames, parent, 'anchor', case['index'], parent, budget) if result.success else ([], [])
    return full, result_row(case, result, engine)


def verify_prepared(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    identity = json.loads((output / 'identity.json').read_text(encoding='utf-8'))
    expected = json.loads(json.dumps(make_config(complete_diagnostic=True)))
    if file_digest(output / 'config.json') != identity['config_sha256'] or config != expected:
        raise FrozenIdentityError('诊断配置/代码/输入/分组漂移，禁止运行')
    for path in config.get('frozen_files', {}):
        if Path(path).name == 'completed.json' and verify_unit(Path(path).parent) is None:
            raise FrozenIdentityError(f'前置完成产物缺失：{path}')
    print('PREPARED IDENTITY VERIFIED; NO ROLLOUT STARTED', flush=True)
    return config


def prepare_diagnostic(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('诊断准备目录必须不存在')
    config = make_config(complete_diagnostic=True); output.mkdir(parents=True)
    atomic_write_json(output / 'config.json', config)
    atomic_write_json(output / 'identity.json', {'config_sha256': file_digest(output / 'config.json')})
    atomic_write_json(output / 'report.json', {'status': 'prepared', 'training_performed': False,
                                             'allowed_to_prepare_training_plan': False, 'main_protocol_passed': False})
    print('DIAGNOSTIC PREPARED; NO ROLLOUT OR TRAINING STARTED', flush=True)


def execution_budget(complete_diagnostic):
    return ExplorationBudget(900) if complete_diagnostic else PreflightBudget(180)


def run(output, *, complete_diagnostic=False, prepared=False):
    output = Path(output).resolve()
    config = None
    if complete_diagnostic != prepared:
        raise ValueError('完整诊断必须先prepare，执行须显式run-prepared；短检不支持续跑')
    if prepared:
        config = verify_prepared(output)
        if json.loads((output / 'report.json').read_text(encoding='utf-8'))['status'] != 'prepared':
            raise FileExistsError('已有执行结果，不覆盖/自动续跑；反馈后再决策')
    else:
        if output.exists():
            raise FileExistsError('短检目录必须不存在，不覆盖/续跑')
        output.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.perf_counter(); budget = execution_budget(complete_diagnostic)
    report = {'status': 'running', 'stage': 'identity', 'branch_failures': [], 'completed_arms': {},
              'training_performed': False, 'allowed_to_prepare_long_task': False,
              'allowed_to_prepare_training_plan': False, 'complete_diagnostic': complete_diagnostic,
              'main_protocol_passed': False, 'automatic_deployment': False}
    def save():
        report['elapsed_seconds'] = time.perf_counter() - started
        atomic_write_json(output / 'report.json', report)
    with exploration_lock(output):
        if prepared and json.loads((output / 'report.json').read_text(encoding='utf-8'))['status'] != 'prepared':
            raise FileExistsError('已在其他进程执行，不覆盖结果')
        save()
        try:
            if config is None:
                config = make_config(); atomic_write_json(output / 'config.json', config)
            report['config_sha256'] = file_digest(output / 'config.json'); save(); budget.check()
            old_config = json.loads((LT16 / 'config.json').read_text(encoding='utf-8'))
            for seed in sampling.TRAINING_SEEDS:
                if verify_unit(LT16 / 'units' / f'fit-broad_novel-{seed}') is None:
                    raise FrozenIdentityError('联合起点单元缺失')
            old_samples, old_directories = load_original_pool(old_config, budget)
            data = DatasetGenerator.load(INPUTS['train']); metadata = data['task_meta']
            manifest = matrix.load_dataset_manifest(INPUTS['train']); vehicle = VehicleConfig(**manifest['vehicle_model'])
            parent = load_model_checkpoint(INPUTS['R0']).model
            models = {}; optimizers = {}
            for seed in sampling.TRAINING_SEEDS:
                models[seed], optimizers[seed] = load_candidate(config['parent_checkpoints'][str(seed)], parent)
            groups = config['groups']; indices = groups['dense_replay_sources'] + groups['gap_candidates'] + groups['probes']
            indices += groups.get('regression_candidates', [])
            cases = {i: build_case(data, manifest, i, vehicle) for i in sorted(set(indices))}
            proposed = []; source_rows = []; report['stage'] = 'collect'; save()
            for index in groups['dense_replay_sources'] + groups['gap_candidates']:
                print(f'COLLECT #{index}', flush=True); point = output / f'collect-{index}'; point.mkdir()
                expected = old_directories[index] / 'R0.npz' if index in groups['dense_replay_sources'] else None
                samples, row = collect_anchor(cases[index], parent, budget, point, expected)
                proposed.extend(samples); source_rows.append(row)
                report.update(collection=source_rows); save()
            novel, novelty = novel_samples(old_samples, proposed)
            fit_indices = []
            for number in (4, 6):
                matching = [row['dataset_index'] for row in source_rows if row['dataset_index'] in groups['gap_candidates']
                            and row['success'] and scene_number(metadata[row['dataset_index']]) == number]
                if matching:
                    fit_indices.append(matching[0])
            report.update(novelty=novelty, fit_indices=fit_indices, data_gate_passed=len(fit_indices) == 2,
                          augmented_frames=len(old_samples) + len(novel))
            torch.save({'purpose': 'train_sources_only_no_probe_labels', 'old': old_samples, 'novel': novel}, output / 'data.pt')
            save()
            if len(fit_indices) != 2:
                report.update(status='completed_data_gate_failed', decision='no_training_missing_positive_anchor_cell')
                verify_files(config['code']); verify_files(config['frozen_files']); save()
                print(f'FINISHED {report["decision"]}', flush=True); return report
            report['stage'] = 'baseline'; save()
            baseline_dir = output / 'baseline'; baseline_dir.mkdir()
            print('BASELINE R0 original 6 probes', flush=True)
            r0 = evaluate_cases(groups['probes'], cases, parent, budget, baseline_dir, 'R0-probe')
            baselines = {}; baseline_fit = {}
            for seed in sampling.TRAINING_SEEDS:
                print(f'BASELINE parent seed={seed} original probes/fit', flush=True)
                baselines[seed] = evaluate_cases(groups['probes'], cases, models[seed], budget, baseline_dir, f'parent-{seed}')
                baseline_fit[seed] = evaluate_cases(fit_indices, cases, models[seed], budget, baseline_dir, f'parent-fit-{seed}')
            report.update(r0_probe=r0, parent_probe=baselines, parent_fit=baseline_fit); save()
            evaluation_probes = list(groups['probes'])
            if complete_diagnostic:
                extra = groups['regression_candidates']
                report['stage'] = 'regression_candidate_baseline'; save()
                print('BASELINE R0 regression candidates=16', flush=True)
                candidate_r0 = evaluate_cases(extra, cases, parent, budget, baseline_dir, 'candidate-R0')
                candidate_parents = {}
                for seed in sampling.TRAINING_SEEDS:
                    print(f'BASELINE parent seed={seed} regression candidates=16', flush=True)
                    candidate_parents[seed] = evaluate_cases(extra, cases, models[seed], budget, baseline_dir, f'candidate-parent-{seed}')
                combined_r0 = r0 + candidate_r0
                combined_parents = {seed: baselines[seed] + candidate_parents[seed] for seed in sampling.TRAINING_SEEDS}
                baseline_gate = sampling.choose_witnesses(combined_r0, combined_parents, evaluation_probes + extra)
                chosen = [i for i in baseline_gate['selected_indices'] if i not in evaluation_probes]
                evaluation_probes += chosen
                r0_map = {row['dataset_index']: row for row in combined_r0}
                parent_maps = {seed: {row['dataset_index']: row for row in rows} for seed, rows in combined_parents.items()}
                r0 = [r0_map[i] for i in evaluation_probes]
                baselines = {seed: [parent_maps[seed][i] for i in evaluation_probes] for seed in sampling.TRAINING_SEEDS}
                report.update(regression_candidate_R0=candidate_r0, regression_candidate_parents=candidate_parents)
            else:
                baseline_gate = sampling.choose_witnesses(r0, baselines, evaluation_probes)
            report.update(r0_probe=r0, parent_probe=baselines, parent_fit=baseline_fit,
                          baseline_gate=baseline_gate, evaluation_probes=evaluation_probes); save()
            if not baseline_gate['passed']:
                report.update(status='completed_baseline_gate_failed', decision='no_training_missing_measurable_retention_regression')
                verify_files(config['code']); verify_files(config['frozen_files']); save()
                print(f'FINISHED {report["decision"]}', flush=True); return report
            pools = {'old': old_samples, 'augmented': old_samples + novel}; results = {}
            report['stage'] = 'matched_fit_and_probe'; save()
            for name, (pool, mode) in ARMS.items():
                for seed in sampling.TRAINING_SEEDS:
                    budget.check(); point = output / f'{name}-{seed}'; point.mkdir()
                    print(f'FIT {name} seed={seed} updates=24', flush=True)
                    try:
                        schedule = sampling.build_schedule(pools[pool], metadata, seed=seed, steps=24,
                                                           mode=mode, recovery_reference=old_samples)
                        audit = sampling.sampling_audit(schedule, metadata)
                        model, optimizer, history = sampling.fit_continuation(models[seed], optimizers[seed], schedule,
                            seed=seed, check_budget=budget.check)
                        if any(int(s['step']) != 264 for s in optimizer.state.values()):
                            raise ValueError('Adam未到264步')
                        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                                    'step': 264, 'threshold': model.stop_threshold,
                                    'purpose': 'diagnostic_not_deployment'}, point / 'state.pt')
                        probes = evaluate_cases(evaluation_probes, cases, model, budget, point, 'probe')
                        fit = evaluate_cases(fit_indices, cases, model, budget, point, 'fit')
                        row = {'arm': name, 'seed': seed, 'step': 264, 'probe': probes, 'fit': fit,
                               'sampling': audit, 'loss_history': history, 'checkpoint': str(point / 'state.pt')}
                        atomic_write_json(point / 'report.json', row)
                        report['completed_arms'].setdefault(name, {})[seed] = row
                        results.setdefault(name, {})[seed] = probes
                        report['training_performed'] = True; save()
                    except ValueError as exc:
                        report['branch_failures'].append({'arm': name, 'seed': seed, 'error': str(exc)})
                        atomic_write_json(point / 'error.json', {'error': str(exc)}); save()
            verdict = sampling.qualification(r0, baselines, results)
            complete = not report['branch_failures'] and sum(len(v) for v in results.values()) == 8
            report.update(status='completed' if complete else 'completed_with_branch_failures', qualification=verdict,
                          all_predeclared_branches_completed=complete,
                          allowed_to_prepare_long_task=complete and bool(verdict['qualified']),
                          allowed_to_prepare_training_plan=complete and bool(verdict['qualified']),
                          decision='closed_loop_retention_signal_prepare_manual_plan' if complete and verdict['qualified']
                                   else 'no_qualified_retention_signal_do_not_expand')
            verify_files(config['code']); verify_files(config['frozen_files']); budget.check()
            report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                                         if p.is_file() and p != output / 'report.json' and p.name not in ('run.lock', 'console.log')}
            budget.check()
            save(); print(f'FINISHED {report["decision"]}', flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else
                          'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=str(exc), allowed_to_prepare_long_task=False, decision='incomplete_do_not_infer_benefit')
            report['allowed_to_prepare_training_plan'] = False
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--complete-diagnostic', action='store_true', help='仅用户运行的完整诊断，不增加24步更新量')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare-only', action='store_true')
    actions.add_argument('--precheck', action='store_true')
    actions.add_argument('--run-prepared', action='store_true')
    args = parser.parse_args()
    if (args.prepare_only or args.precheck or args.run_prepared) and not args.complete_diagnostic:
        parser.error('准备/预检/执行已备配置必须显式complete-diagnostic')
    if args.prepare_only:
        prepare_diagnostic(args.output)
    elif args.precheck:
        verify_prepared(args.output)
    else:
        run(args.output, complete_diagnostic=args.complete_diagnostic, prepared=args.run_prepared)
