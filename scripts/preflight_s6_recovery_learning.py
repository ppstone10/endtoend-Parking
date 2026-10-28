"""S6严格纠偏首帧与C240自目标配对；180秒，固定24更新，无自动扩张。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration_data import evaluate_cases
from experiments.effective_recovery_data import input_id, label_id, novel_samples
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.retention_sampling import TRAINING_SEEDS, fit_continuation, sampling_audit
from experiments.s6_recovery_learning import ARMS, batch_identity, build_recovery_schedules, recovery_qualification
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix
from scripts.preflight_retention_consumption import prepare_context as prepare_consumption_context
from scripts.preflight_learner_failure_teacher import verify_config, STRICT, LT17
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


PREVIOUS = ROOT / 'runs/diagnostics/failure-teacher-refinement-preflight-20261022-v1'
QUERIES = ROOT / 'runs/diagnostics/learner-failure-teacher-preflight-20261022-v1'
CONSUMPTION = ROOT / 'runs/diagnostics/retention-consumption-preflight-20261021-v1'
HASHES = {'config.json': '038adeb6d54cda6945496af375166baa9ad0e4c4d896414b53b174b33d5e61f3',
          'report.json': 'dec7c4256164696845ed6b5e19003f0e0e075c9cd43b1ef168e26507d9d81342'}
CONSUMPTION_HASHES = {'config.json': 'b69652b11de886cdc7997f588db93663d677482788f6991c1af1c777799f704c',
                      'report.json': '54f88af74e79c20fe9fd6e5dbd4b1759a861a99b821964e3c0e029804dac7c52'}
FIT = (1196, 2149, 1400, 2256)


def prepare_context(budget):
    torch.set_num_threads(1)
    verify_files({str(PREVIOUS/k): h for k, h in HASHES.items()})
    old = json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'))
    previous_report = json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    if previous_report['status'] != 'completed' or previous_report['training_performed']:
        raise ValueError('不是完整且未训练的教师校正证据')
    config = dict(old); config['code'] = dict(old['code']); config['frozen_files'] = dict(old['frozen_files'])
    config['frozen_files'].update({str(PREVIOUS/k): h for k, h in HASHES.items()})
    config['frozen_files'].update({str(artifact_path(PREVIOUS,k)): h for k,h in previous_report['artifacts_sha256'].items()})
    config['frozen_files'].update({str(CONSUMPTION/k): h for k,h in CONSUMPTION_HASHES.items()})
    from experiments import s6_recovery_learning
    for path in (Path(s6_recovery_learning.__file__).resolve(), Path(__file__).resolve()):
        config['code'][str(path)] = file_digest(path)
    verify_config(config, budget)
    query_report = json.loads((QUERIES/'report.json').read_text(encoding='utf-8'))
    if query_report['status'] != 'completed' or query_report['accepted_new_queries'] != 3:
        raise ValueError('不是固定三个有效新增查询')
    queries = torch.load(QUERIES/'queries.pt', map_location='cpu', weights_only=False)['samples']
    bank = torch.load(LT17/'data.pt', map_location='cpu', weights_only=False)
    novel, novelty = novel_samples(bank['old']+bank['novel'], queries)
    if len(novel) != 3:
        raise ValueError('旧库外新增性未复现')
    rows = next(r for r in query_report['sources'] if r['seed'] == 20261020 and r['dataset_index'] == 2149)
    if {q['step'] for q in rows['queries']} != {10, 200, 400}:
        raise ValueError('查询来源不完整')
    evidence = {q['step']: q for q in rows['queries']}
    for q in queries:
        row = evidence[q['query_step']]
        if (row['decision'] != 'strict_query_accepted' or not row['teacher']['reached_with_margin'] or
                not row['query_window']['clean'] or row['input_sha256'] != input_id(q) or
                row['label_sha256'] != label_id(q)):
            raise ValueError('查询缓存与真实教师执行证据不符')
    context = prepare_consumption_context(budget)
    if context['config']['parent_checkpoints'] != config['parent_checkpoints']:
        raise ValueError('C240起点漂移')
    config.update(kind='s6_recovery_learning_preflight_v1', budget_seconds=180, arms=list(ARMS),
        start_step=240, steps=24, training_seeds=list(TRAINING_SEEDS),
        probe_indices=context['config']['probe_indices'], fit_indices=list(FIT),
        query_steps=[10, 200, 400], query_policy='C240-20261020', novelty=novelty,
        comparison='same three recovery inputs/masks; C240 free self-targets versus verified expert targets')
    parent = load_model_checkpoint(INPUTS['R0']).model
    plans = {a: {} for a in ARMS}; sampling = {a: {} for a in ARMS}; pairing = {}; identities = {a: {} for a in ARMS}
    for seed in TRAINING_SEEDS:
        model, _ = load_candidate(config['parent_checkpoints'][str(seed)], parent)
        by_arm, pairing[seed] = build_recovery_schedules(context['schedules'][seed], queries, model, budget.check)
        for arm in ARMS:
            plans[arm][seed] = by_arm[arm]; sampling[arm][seed] = sampling_audit(by_arm[arm], context['metadata'])
            identities[arm][seed] = [batch_identity(batch) for batch in by_arm[arm]]
    config.update(sampling=sampling, pairing_audits=pairing, planned_batch_identities=identities)
    consumption = json.loads((CONSUMPTION/'report.json').read_text(encoding='utf-8'))
    strict = json.loads((STRICT/'report.json').read_text(encoding='utf-8'))
    context.update(config=config, plans=plans, parent=parent, consumption=consumption,
                   parent_fit={seed:[r for r in strict['source_rows'] if r['seed']==seed] for seed in TRAINING_SEEDS})
    budget.check(); return context


def precheck():
    context = prepare_context(PreflightBudget(180))
    print('S6 QUERIES AND PAIRED TARGET PLANS VERIFIED; NO TRAINING OR ROLLOUT', flush=True)
    return context


def fit_recovery_point(context, arm, seed, budget, directory):
    config = context['config']; schedule = context['plans'][arm][seed]
    model, optimizer_state = load_candidate(config['parent_checkpoints'][str(seed)], context['parent'])
    presented = []; planned = config['planned_batch_identities'][arm][seed]

    def check_batch():
        budget.check(); position = len(presented)
        if position >= 24: raise ValueError('超过预定24更新')
        actual = batch_identity(schedule[position])
        if actual != planned[position]: raise ValueError('实际梯度批次输入或标签身份漂移')
        presented.append(actual)

    model, optimizer, history = fit_continuation(model, optimizer_state, schedule, seed=seed, check_budget=check_batch)
    if len(presented) != 24 or {int(s['step']) for s in optimizer.state.values()} != {264}:
        raise ValueError('实际批次/Adam264计数不符')
    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), step=264,
                   threshold=model.stop_threshold, purpose='diagnostic_not_deployment'), directory/'state.pt')
    point = dict(arm=arm, seed=seed, step=264, checkpoint=str(directory/'state.pt'),
                 presented_batch_identities=presented, loss_history=history)
    atomic_write_json(directory/'report.json', point)
    point['probe'] = evaluate_cases(config['probe_indices'], context['cases'], model, budget, directory, 'probe')
    point['fit'] = evaluate_cases(config['fit_indices'], context['cases'], model, budget, directory, 'fit')
    atomic_write_json(directory/'report.json', point); return point


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(180)
    report = dict(status='running', stage='identity_and_query_gate', completed_arms={}, training_started=False,
        training_performed=False, allowed_to_prepare_training_plan=False, main_protocol_passed=False, automatic_deployment=False)

    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_write_json(output/'report.json', report)

    with exploration_lock(output):
        save()
        try:
            context = prepare_context(budget); config = context['config']
            atomic_write_json(output/'config.json', config)
            report['config_sha256'] = file_digest(output/'config.json'); save()
            data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
            vehicle = VehicleConfig(**manifest['vehicle_model'])
            context['cases'] = {i:build_case(data,manifest,i,vehicle) for i in sorted(set(config['probe_indices']+list(FIT)))}
            report.update(reused_parent_probe=context['previous']['parent_probe'], reused_parent_fit=context['parent_fit'],
                reused_consumption_probe={s:p['probe'] for s,p in context['consumption']['completed_seeds'].items()},
                pairing_audits=config['pairing_audits']); save()
            for arm in ARMS:
                for seed in TRAINING_SEEDS:
                    budget.check(); directory = output/f'{arm}-{seed}'; directory.mkdir()
                    report.update(stage=f'{arm}-{seed}', training_started=True); save()
                    print(f'FIT {arm} seed={seed} updates=24', flush=True)
                    point = fit_recovery_point(context, arm, seed, budget, directory)
                    report['completed_arms'].setdefault(arm,{})[seed] = point
                    report['training_performed'] = True; save()
            previous = context['previous']; int_keys = lambda d:{int(k):v for k,v in d.items()}
            candidates = {arm:{s:p['probe'] for s,p in values.items()} for arm,values in report['completed_arms'].items()}
            verdict = recovery_qualification(previous['r0_probe'], int_keys(previous['parent_probe']),
                {int(s):p['probe'] for s,p in previous['completed_arms']['flat_old'].items()},
                int_keys(report['reused_consumption_probe']), candidates['self_targets'], candidates['strict_targets'],
                context['parent_fit'], {s:p['fit'] for s,p in report['completed_arms']['strict_targets'].items()})
            verify_config(config, budget)
            report['artifacts_sha256'] = {str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p != output/'report.json' and p.name not in ('run.lock','console.log')}
            budget.check()
            report.update(status='completed', stage='decision', qualification=verdict,
                allowed_to_prepare_training_plan=verdict['passed'], decision='s6_recovery_signal_prepare_manual_plan'
                if verdict['passed'] else 'no_qualified_s6_recovery_signal_do_not_expand')
            save(); print(f'FINISHED {report["decision"]}', flush=True); return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', error=str(exc),
                allowed_to_prepare_training_plan=False, decision='incomplete_do_not_infer_learning_benefit')
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path); parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output目录')
    else: run(args.output)
