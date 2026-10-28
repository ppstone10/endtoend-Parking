"""S3/S6实际C240失败状态的严格恢复教师短检；180秒且不训练。"""
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
from experiments.behavior_exploration_data import run_case, save_record
from experiments.effective_recovery_data import input_id, label_id, novel_samples
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.learner_failure_teacher import SOURCES, failure_contexts, readiness, validated_query
from experiments.recovery_execution import record_arrays, verify_network_suffix
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix, result_row
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest, runtime_identity, metadata_only
from scripts.build_selection_dataset import collect_metadata_ids
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


PREVIOUS = ROOT / 'runs/diagnostics/retention-teacher-roles-preflight-20261022-v1'
STRICT = ROOT / 'runs/diagnostics/retention-teacher-preflight-20261022-v1'
LT17 = ROOT / 'runs/diagnostics/retention-sampling-diagnostic-20261021-v2'
HASHES = {'config.json': '09425de378dfeb095b43d41686701edf9833592ee33eb70d6f6831228fc4ad8b',
          'report.json': 'c6ae0a3b1137e2a99f02f3ba11cdc17a6ba36e3807986e77013c25439de735a9'}


def verify_config(config, budget):
    if runtime_identity() != config['runtime']:
        raise ValueError('运行环境漂移')
    for field in ('code', 'frozen_files'):
        for path, digest in config[field].items():
            budget.check(); verify_files({path: digest})


def prepare_config(budget):
    verify_files({str(PREVIOUS / k): h for k, h in HASHES.items()})
    old = json.loads((PREVIOUS / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((PREVIOUS / 'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed' or report['allowed_to_prepare_training_plan']:
        raise ValueError('不是完整且未放行的来源分工基线')
    code = dict(old['code']); files = dict(old['frozen_files'])
    files.update({str(PREVIOUS / k): h for k, h in HASHES.items()})
    files.update({str(artifact_path(PREVIOUS, k)): h for k, h in report['artifacts_sha256'].items()})
    from experiments import learner_failure_teacher
    for path in (Path(learner_failure_teacher.__file__).resolve(), Path(__file__).resolve()):
        code[str(path)] = file_digest(path)
    files = {str(Path(path).resolve()): digest for path, digest in files.items()}
    upstream = ROOT / 'runs/diagnostics/effective-replay-20261021-v2/config.json'
    files[str(upstream)] = '368718182c5f8fbfef43ef68ea278ae87583d069245dd78606b3ed56e65cb088'
    verify_files({str(upstream): files[str(upstream)]})
    inputs = json.loads(upstream.read_text(encoding='utf-8'))['inputs']
    files.update({str(Path(item['path']).resolve()): item['sha256'] for item in inputs.values()})
    config = dict(kind='learner_failure_teacher_preflight_v1', code=code, frozen_files=files,
        runtime=old['runtime'], parent_checkpoints=old['parent_checkpoints'], sources=list(SOURCES),
        budget_seconds=180, max_steps=600, min_remaining_steps=200, replan_every=10,
        context_selection='uniform_nonzero_planning_safe_before_first_margin_and_step400',
        query_order='latest_then_middle_then_earliest_round_robin_across_fixed_sources',
        control_seed=0, training_performed=False, automatic_deployment=False, main_protocol_passed=False)
    verify_config(config, budget)
    # 上游冻结清单必须确实包含所用输入、模型和实际来源轨迹，不能只相信路径存在。
    required = list(INPUTS.values()) + [LT17 / 'data.pt']
    required += [Path(p) for p in config['parent_checkpoints'].values()]
    required += [STRICT / f'source-{s}-{i}/C240.npz' for s, i in SOURCES]
    if any(str(p) not in files for p in required):
        raise ValueError('上游未冻结本轮必要输入/模型/源轨迹')
    metadata = metadata_only(INPUTS['train'])
    reserved = collect_metadata_ids(metadata_only(INPUTS['val'])) | collect_metadata_ids(metadata_only(INPUTS['selection']))
    selected = collect_metadata_ids([metadata[i] for _, i in SOURCES])
    if selected & reserved or any(i in old['probe_indices'] for _, i in SOURCES):
        raise ValueError('教师查询来源与保留集/探针冲突')
    config['source_tasks'] = [dict(seed=s, dataset_index=i, task_id=metadata[i]['task_id'],
                                  scene_name=metadata[i]['scene_name']) for s, i in SOURCES]
    return config


def precheck():
    config = prepare_config(PreflightBudget(180))
    print('IDENTITY AND FIXED TRAINING SOURCES VERIFIED; NO ROLLOUT OR TRAINING', flush=True)
    return config


def collect(config, budget, output, report, save):
    parent = load_model_checkpoint(INPUTS['R0']).model
    data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
    vehicle = VehicleConfig(**manifest['vehicle_model'])
    cases = {i: build_case(data, manifest, i, vehicle) for _, i in SOURCES}
    models = {s: load_candidate(config['parent_checkpoints'][str(s)], parent)[0] for s, _ in SOURCES}
    sources = []; report['sources'] = []
    for seed, index in SOURCES:
        budget.check(); case = cases[index]; directory = output / f'source-{seed}-{index}'; directory.mkdir()
        report['stage'] = f'learner-{seed}-{index}'; save()
        print(f'LEARNER seed={seed} source={index}', flush=True)
        result, engine, frames = run_case(case, models[seed], budget)
        states, commands = record_arrays(result)
        with np.load(STRICT / f'source-{seed}-{index}/C240.npz') as expected:
            if not np.array_equal(states, expected['states']) or not np.array_equal(commands, expected['commands']):
                raise ValueError('当前学习器未复现冻结实际失败回合')
        if result.success: raise ValueError('固定失败来源变成成功，身份不符')
        contexts = failure_contexts(engine.contexts, engine.margin_violations)
        row = dict(seed=seed, **result_row(case, result, engine),
                   selected_context_steps=[c.step for c in contexts], exact_archived_trajectory=True, queries=[])
        report['sources'].append(row)
        save_record(directory, 'learner', result, margin_violations=np.asarray(engine.margin_violations))
        sources.append(dict(case=case, model=models[seed], result=result, states=states,
                            frames=frames, contexts=contexts, directory=directory, row=row))
        save()
    config['selected_context_steps'] = {f'{r["seed"]}-{r["dataset_index"]}': r['selected_context_steps']
                                        for r in report['sources']}
    atomic_write_json(output / 'query-config.json', config)
    report['query_config_sha256'] = file_digest(output / 'query-config.json'); save()
    samples = []
    for rank in range(3):
        for source in sources:
            if rank >= len(source['contexts']): continue
            budget.check(); context = source['contexts'][-1-rank]
            case = source['case']; row = source['row']; directory = source['directory']
            report['stage'] = f'query-{row["seed"]}-{case["index"]}-{context.step}'; save()
            print(f'QUERY seed={row["seed"]} source={case["index"]} step={context.step}', flush=True)
            replay, _, _ = run_case(case, source['model'], budget, context=context)
            query = dict(step=context.step, remaining_steps=600-context.step, decision='running',
                         suffix_verification=verify_network_suffix(source['result'], replay, context.step))
            row['queries'].append(query); save()
            try:
                teacher, audit_engine, teacher_frames = run_case(case, None, budget, context=context, replanning_expert=True)
            except ValueError as exc:
                query.update(decision='teacher_execution_error', error=str(exc))
                atomic_write_json(directory / f'query-{context.step}.json', query); save()
                continue
            teacher_states, _ = record_arrays(teacher)
            save_record(directory, f'teacher-{context.step}', teacher, context,
                        margin_violations=np.asarray(audit_engine.margin_violations))
            query.update(teacher=result_row(case, teacher, audit_engine),
                         planning_failure=teacher.meta.get('planning_failure'),
                         teacher_continuation_frames=max(0, len(teacher_frames)-1),
                         decision='teacher_rejected')
            sample = None
            if teacher_frames and teacher_frames[0]['mask'].any():
                sample, window = validated_query(source['frames'][context.step//10], teacher_frames, case,
                    context.step, f'C240-{row["seed"]}', parent, budget, source['states'], teacher_states)
                query.update(query_window=window, input_sha256=input_id(sample), label_sha256=label_id(sample))
                if query['teacher']['reached_with_margin'] and window['clean']:
                    samples.append(sample); query['decision'] = 'strict_query_accepted'
                elif query['teacher']['reached_with_margin']:
                    query['decision'] = 'unsafe_consumed_reference_rejected'
            torch.save(dict(first_teacher_frame=teacher_frames[0] if teacher_frames else None,
                            proposed_query=sample, accepted=query['decision'] == 'strict_query_accepted'),
                       directory / f'query-{context.step}.pt')
            atomic_write_json(directory / f'query-{context.step}.json', query); save()
    bank = torch.load(LT17 / 'data.pt', weights_only=False, map_location='cpu')
    new, novelty = novel_samples(bank['old'] + bank['novel'], samples)
    report.update(novelty=novelty, readiness=readiness(report['sources'], {input_id(s) for s in new}),
                  accepted_strict_queries=len(samples), accepted_new_queries=len(new))
    torch.save(dict(purpose='teacher_feasibility_only_not_training', samples=new), output / 'queries.pt')
    return report['readiness']['passed']


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(180); torch.set_num_threads(1)
    report = dict(status='running', stage='identity', sources=[], training_performed=False,
                  learning_benefit_tested=False, allowed_to_prepare_training_plan=False,
                  main_protocol_passed=False, automatic_deployment=False)

    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_write_json(output / 'report.json', report)

    with exploration_lock(output):
        save()
        try:
            config = prepare_config(budget); atomic_write_json(output / 'config.json', config)
            report['config_sha256'] = file_digest(output / 'config.json'); save()
            passed = collect(config, budget, output, report, save)
            verify_config(config, budget)
            report['artifacts_sha256'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                if p.is_file() and p != output / 'report.json' and p.name not in ('run.lock', 'console.log')}
            budget.check()
            report.update(status='completed', stage='decision',
                decision='recovery_queries_ready_for_paired_short_test' if passed else 'teacher_readiness_failed_no_training')
            save(); print(f'FINISHED {report["decision"]}', flush=True)
            return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else
                          'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=str(exc), decision='incomplete_do_not_infer_teacher_feasibility')
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path); parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output目录')
    else: run(args.output)
