"""S3实际失败状态的固定参考/既有航向补偿对照；120秒，无训练。"""
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
from experiments.learner_failure_teacher import validated_query
from experiments.recovery_execution import record_arrays, verify_network_suffix
from experiments.reference_compensation import compensate_reference_yaw, reference_internal_safe
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix, result_row
from scripts.preflight_learner_failure_teacher import verify_config, LT17
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


PREVIOUS = ROOT / 'runs/diagnostics/learner-failure-teacher-preflight-20261022-v1'
HASHES = {'config.json': 'e002166f81d0e63d04dbaf8184673dc595968bc9cebb35c37f1f525d753546a7',
          'report.json': '9472126c081d17d84f2853aef06df11d2fe9fb0b48187aeef0728bbe9896535c'}


def safe_connected_reference(trajectory, checker, state):
    return (reference_internal_safe(trajectory, checker) and
            bool(checker.swept_segment_free(state.to_array()[:3], trajectory.points[0])))


def prepare_config(budget):
    verify_files({str(PREVIOUS / k): h for k, h in HASHES.items()})
    old = json.loads((PREVIOUS / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((PREVIOUS / 'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed' or report['readiness']['passed']:
        raise ValueError('不是已完成且未放行的恢复教师基线')
    config = dict(old); config['code'] = dict(old['code']); config['frozen_files'] = dict(old['frozen_files'])
    config['code'][str(Path(__file__).resolve())] = file_digest(__file__)
    config['frozen_files'].update({str(PREVIOUS / k): h for k, h in HASHES.items()})
    config['frozen_files'].update({str(artifact_path(PREVIOUS, k)): h for k, h in report['artifacts_sha256'].items()})
    sources = [dict(seed=r['seed'], dataset_index=r['dataset_index'], step=max(r['selected_context_steps']))
               for r in report['sources'] if r['dataset_index'] == 1196]
    if sources != [dict(seed=20261020, dataset_index=1196, step=90),
                   dict(seed=20261021, dataset_index=1196, step=80)]:
        raise ValueError('固定最晚S3上下文漂移')
    config.update(kind='failure_teacher_reference_refinement_v1', budget_seconds=120, sources=sources,
        reference_policy='fresh_plan_from_real_context_then_fixed',
        compensation=dict(max_updates=2, gain=.5, smooth_points=5, max_yaw_offset=.1),
        training_performed=False)
    verify_config(config, budget); return config


def precheck():
    config = prepare_config(PreflightBudget(120))
    print('REFINEMENT IDENTITY VERIFIED; NO PLANNING ROLLOUT OR TRAINING', flush=True)
    return config


def collect(config, budget, output, report, save):
    parent = load_model_checkpoint(INPUTS['R0']).model
    data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
    vehicle = VehicleConfig(**manifest['vehicle_model']); case = build_case(data, manifest, 1196, vehicle)
    samples = []; report['sources'] = []
    for source in config['sources']:
        seed, step = source['seed'], source['step']; directory = output / f'source-{seed}-1196'; directory.mkdir()
        report['stage'] = f'source-{seed}'; save()
        learner, _ = load_candidate(config['parent_checkpoints'][str(seed)], parent)
        result, engine, frames = run_case(case, learner, budget)
        states, commands = record_arrays(result)
        with np.load(PREVIOUS / f'source-{seed}-1196/learner.npz') as archive:
            if not np.array_equal(states, archive['states']) or not np.array_equal(commands, archive['commands']):
                raise ValueError('C240实际回合漂移')
        context = next(c for c in engine.contexts if c.step == step)
        replay, _, _ = run_case(case, learner, budget, context=context)
        row = dict(seed=seed, dataset_index=1196, step=step, remaining_steps=600-step,
                   suffix_verification=verify_network_suffix(result, replay, step), attempts=[], accepted=False)
        report['sources'].append(row); save_record(directory, 'learner', result); save()
        print(f'PLAN seed={seed} step={step}', flush=True)
        budget.check()
        try:
            original = case['planner'].plan(context.state, case['goal'])
        except (ValueError, RuntimeError) as exc:
            row.update(decision='initial_planner_rejected', error=str(exc)); save(); continue
        teacher = original
        for update in range(3):
            budget.check()
            safe = safe_connected_reference(teacher, case['planner']._collision_checker, context.state)
            attempt = dict(update=update, reference_connected_safe=safe, decision='unsafe_reference_rejected')
            row['attempts'].append(attempt); save()
            if not safe: break
            print(f'FIXED seed={seed} update={update}', flush=True)
            executed, expert_engine, teacher_frames = run_case(case, None, budget, context=context, trajectory=teacher)
            actual, _ = record_arrays(executed)
            save_record(directory, f'fixed-{update}', executed, context, reference=teacher.points,
                dt=np.asarray(teacher.dt), progress_trace=np.asarray(expert_engine.mpc.execution_trace),
                margin_violations=np.asarray(expert_engine.margin_violations))
            attempt.update(execution=result_row(case, executed, expert_engine), decision='execution_rejected')
            sample = None
            if teacher_frames and teacher_frames[0]['mask'].any():
                sample, window = validated_query(frames[step//10], teacher_frames, case, step,
                    f'C240-{seed}', parent, budget, states, actual)
                attempt.update(query_window=window, input_sha256=input_id(sample), label_sha256=label_id(sample))
                if attempt['execution']['reached_with_margin'] and window['clean']:
                    samples.append(sample); attempt['decision'] = 'strict_query_accepted'; row['accepted'] = True
            torch.save(dict(first_teacher_frame=teacher_frames[0] if teacher_frames else None,
                            proposed_query=sample, accepted=row['accepted']), directory / f'query-{update}.pt')
            save()
            if row['accepted'] or update == 2 or not len(actual): break
            teacher = compensate_reference_yaw(original, teacher, actual, expert_engine.mpc.execution_trace)
        row['decision'] = 'strict_teacher_available' if row['accepted'] else 'no_verified_fixed_teacher'
        save()
    bank = torch.load(LT17 / 'data.pt', weights_only=False, map_location='cpu')
    new, novelty = novel_samples(bank['old'] + bank['novel'], samples)
    passed = len(report['sources']) == 2 and all(r['accepted'] for r in report['sources']) and len(new) == 2
    report.update(readiness_passed=passed, novelty=novelty, accepted_new_queries=len(new))
    torch.save(dict(purpose='reference_teacher_feasibility_only', samples=new), output / 'queries.pt')
    return passed


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(120); torch.set_num_threads(1)
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
            report.update(status='completed', stage='decision', decision='fixed_teacher_ready_not_learning_benefit'
                if passed else 'fixed_teacher_not_ready_no_training')
            save(); print(f'FINISHED {report["decision"]}', flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc, TimeoutError) else
                          'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc),
                          decision='incomplete_do_not_infer_refinement_benefit')
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path); parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output目录')
    else: run(args.output)
