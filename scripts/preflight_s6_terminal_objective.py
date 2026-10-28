"""停止边界/终点位姿监督配对；180秒总预算，四模型44闭环，无自动扩张。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration import paired_counts
from experiments.behavior_exploration_data import evaluate_cases
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path, exploration_lock
from experiments.retention_sampling import TRAINING_SEEDS
from experiments.s6_recovery_learning import batch_identity, recovery_qualification
from experiments.s6_terminal_objective import ARMS, objective_loss, objective_slots, terminal_query, objective_qualification
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS, build_case, matrix
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.preflight_s6_terminal_consumption import prepare_context as prepare_terminal_context, ARM as OLD_ARM
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.reporting import atomic_write_json


PREVIOUS = ROOT/'runs/diagnostics/s6-terminal-consumption-preflight-20261022-v1'
HASHES = {'config.json':'423ea9feef9d9b92395e27006e235032a62c43a39b4ba93c1b10edc1feccefa3',
          'report.json':'66899e30b89a6eb5ace1c8326cc72165d2541360f17c884e89fe1182ebf81e99'}


def prepare_context(budget):
    verify_files({str(PREVIOUS/k):v for k,v in HASHES.items()})
    old = json.loads((PREVIOUS/'config.json').read_text(encoding='utf-8'))
    report = json.loads((PREVIOUS/'report.json').read_text(encoding='utf-8'))
    if report['status'] != 'completed' or report['allowed_to_prepare_training_plan']:
        raise ValueError('不是完整且未放行的终段mask对照')
    context = prepare_terminal_context(budget)
    if json.loads(json.dumps(context['config'])) != old:
        raise ValueError('原129文件配置/标签/排程未完整复现')
    base = context['plans'][OLD_ARM]
    for seed in TRAINING_SEEDS:
        identities = [batch_identity(batch) for batch in base[seed]]
        if (identities != old['planned_batch_identities'][OLD_ARM][str(seed)] or
                identities != report['completed_seeds'][str(seed)]['presented_batch_identities']):
            raise ValueError('原终段mask对照实际梯度身份未复现')
        objective_slots(base[seed])
    config = deepcopy(old)
    config['frozen_files'].update({str(PREVIOUS/k):v for k,v in HASHES.items()})
    config['frozen_files'].update({str(artifact_path(PREVIOUS,k)):v for k,v in report['artifacts_sha256'].items()})
    from experiments import s6_terminal_objective
    for path in (Path(s6_terminal_objective.__file__).resolve(), Path(__file__).resolve()):
        config['code'][str(path)] = file_digest(path)
    config.update(kind='s6_terminal_objective_preflight_v1', budget_seconds=180, arms=list(ARMS),
        comparison='same complete19 labels/masks/order/24 updates; single-query balanced boundary then terminal pose',
        objective=dict(slots=[[16,2]], length=19, stop_weight=.2, balance_stop=True,
                       stop_target_mode='terminal', terminal_pose_weight=1, yaw_weight=10,
                       yaw_error_mode='wrapped', autonomous_stop_threshold_unchanged=True),
        sampling={arm:old['sampling'][OLD_ARM] for arm in ARMS},
        planned_batch_identities={arm:{s:[batch_identity(b) for b in base[s]] for s in TRAINING_SEEDS} for arm in ARMS},
        reused_terminal_mask_control=str(PREVIOUS))
    verify_config(config,budget)
    context.update(config=config, plans={arm:base for arm in ARMS}, terminal_report=report)
    return context


def precheck():
    context = prepare_context(PreflightBudget(180))
    print('FROZEN FULL19 PLANS AND SINGLE OBJECTIVE SLOT VERIFIED; NO TRAINING OR ROLLOUT',flush=True)
    return context


def fit_point(context, arm, seed, budget, directory):
    config = context['config']; schedule = context['plans'][arm][seed]
    if seed not in TRAINING_SEEDS or objective_slots(schedule) != config['objective']['slots']:
        raise ValueError('顺序/干预槽漂移')
    model, optimizer_state = load_candidate(config['parent_checkpoints'][str(seed)],context['parent'])
    if not optimizer_state['state'] or any(int(s['step']) != 240 for s in optimizer_state['state'].values()):
        raise ValueError('必须继承真实Adam240起点')
    torch.manual_seed(seed); model = deepcopy(model).cpu().train()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    if any(g['lr'] != 1e-5 for g in optimizer.param_groups):
        raise ValueError('学习率漂移')
    planned = config['planned_batch_identities'][arm][seed]
    presented = []; applied = []; history = []
    for position,batch in enumerate(schedule):
        budget.check()
        actual = batch_identity(batch)
        if actual != planned[position] or [s['role'] for s in batch] != ['anchor','anchor','recovery','recovery']:
            raise ValueError('梯度前发现输入/标签/角色身份漂移')
        slots = [[position,i] for i,s in enumerate(batch) if terminal_query(s)]
        if slots != [slot for slot in config['objective']['slots'] if slot[0] == position]:
            raise ValueError('实际损失干预槽漂移')
        stack = lambda key:torch.stack([s[key] for s in batch])
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_with_stop(stack('bev'),stack('goal'),stack('state'))
        loss = objective_loss(out.points,out.stop_logits,batch,arm)
        if not torch.isfinite(loss):
            raise ValueError('短检loss非有限')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step(); presented.append(actual); applied.extend(slots); history.append(float(loss.detach()))
    if len(presented) != 24 or applied != [[16,2]] or {int(s['step']) for s in optimizer.state.values()} != {264}:
        raise ValueError('实际更新/Adam计数/损失干预次数不符')
    model.eval()
    torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),step=264,
                   threshold=model.stop_threshold,purpose='diagnostic_not_deployment'),directory/'state.pt')
    point = dict(arm=arm,seed=seed,step=264,checkpoint=str(directory/'state.pt'),
                 presented_batch_identities=presented,objective_slots=applied,loss_history=history)
    atomic_write_json(directory/'report.json',point)
    point['probe'] = evaluate_cases(config['probe_indices'],context['cases'],model,budget,directory,'probe')
    point['fit'] = evaluate_cases(config['fit_indices'],context['cases'],model,budget,directory,'fit')
    atomic_write_json(directory/'report.json',point)
    return point


def decide(context, completed):
    previous = context['previous']; learning = context['window_report']; terminal = context['terminal_report']
    cast = lambda d:{int(k):v for k,v in d.items()}
    probes = lambda d:{int(s):p['probe'] for s,p in d.items()}
    verdicts = {}
    for arm in ARMS:
        candidates = probes(completed[arm])
        original = recovery_qualification(previous['r0_probe'],cast(learning['reused_parent_probe']),
            probes(previous['completed_arms']['flat_old']),cast(learning['reused_consumption_probe']),
            probes(learning['completed_arms']['self_targets']),candidates,cast(learning['reused_parent_fit']),
            {int(s):p['fit'] for s,p in completed[arm].items()})
        verdicts[arm] = objective_qualification(original,probes(terminal['completed_seeds']),candidates)
    pairs = [dict(seed=s,**paired_counts(probes(completed[ARMS[0]])[s],probes(completed[ARMS[1]])[s]))
             for s in TRAINING_SEEDS]
    fits = {a:{int(s):next(r for r in p['fit'] if r['dataset_index']==2149)['reached_with_margin']
               for s,p in completed[a].items()} for a in ARMS}
    extra_fit = any(not fits[ARMS[0]][s] and fits[ARMS[1]][s] for s in TRAINING_SEEDS)
    no_fit_loss = all(not fits[ARMS[0]][s] or fits[ARMS[1]][s] for s in TRAINING_SEEDS)
    terminal_extra = (all(p['success_delta']>=0 and p['collision_delta']<=0 and not p['lost_success_indices'] for p in pairs)
                      and no_fit_loss and (sum(p['success_delta'] for p in pairs)>=1 or extra_fit))
    return dict(qualified=[a for a in ARMS if verdicts[a]['passed']],directions=verdicts,
                joint_vs_boundary=pairs,terminal_extra_signal=terminal_extra,strict_source_by_arm=fits,
                main_protocol_passed=False,automatic_deployment=False)


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter(); budget = PreflightBudget(180)
    report = dict(status='running',stage='identity',completed_arms={},training_started=False,training_performed=False,
        allowed_to_prepare_training_plan=False,main_protocol_passed=False,automatic_deployment=False)
    def save():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context = prepare_context(budget); config = context['config']
            atomic_write_json(output/'config.json',config)
            report['config_sha256'] = file_digest(output/'config.json'); save()
            data = DatasetGenerator.load(INPUTS['train']); manifest = matrix.load_dataset_manifest(INPUTS['train'])
            vehicle = VehicleConfig(**manifest['vehicle_model'])
            context['cases'] = {i:build_case(data,manifest,i,vehicle)
                                for i in sorted(set(config['probe_indices']+config['fit_indices']))}
            for arm in ARMS:
                for seed in TRAINING_SEEDS:
                    budget.check(); directory = output/f'{arm}-{seed}'; directory.mkdir()
                    report.update(stage=f'{arm}-{seed}',training_started=True); save()
                    print(f'FIT {arm} seed={seed} updates=24',flush=True)
                    point = fit_point(context,arm,seed,budget,directory)
                    report['completed_arms'].setdefault(arm,{})[seed] = point
                    report['training_performed'] = True; save()
            verdict = decide(context,report['completed_arms'])
            verify_config(config,budget)
            report['artifacts_sha256'] = {str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p != output/'report.json' and p.name not in ('run.lock','console.log')}
            budget.check()
            report.update(status='completed',stage='decision',qualification=verdict,
                allowed_to_prepare_training_plan=bool(verdict['qualified']),
                decision='objective_signal_prepare_manual_plan' if verdict['qualified'] else 'no_qualified_objective_signal_do_not_expand')
            save(); print(f'FINISHED {report["decision"]}',flush=True)
            return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),
                training_performed=report['training_performed'] or any(output.glob('*/state.pt')),
                allowed_to_prepare_training_plan=False,decision='incomplete_do_not_infer_objective_benefit')
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path); parser.add_argument('--precheck',action='store_true')
    args = parser.parse_args()
    if args.precheck: precheck()
    elif args.output is None: parser.error('需要不存在的新output目录')
    else: run(args.output)
