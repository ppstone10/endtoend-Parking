"""离线原MPC/候选保留双臂；默认不改变任何生产入口。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller import MPCController
from experiments import validation_matrix as matrix
from experiments.mpc_candidate_solver import CandidatePreservingMPC
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest, runtime_identity
from training.reporting import atomic_write_json


def execution_identity():
    return {**code_identity(), str(Path(__file__).resolve()): file_digest(__file__)}


@contextmanager
def solver_scope(arm):
    if arm not in ('original', 'candidate'):
        raise ValueError('未知求解器试验臂')
    previous = matrix.MPCController
    matrix.MPCController = MPCController if arm == 'original' else CandidatePreservingMPC
    try:
        yield
    finally:
        matrix.MPCController = previous


def prepare(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('必须使用新目录')
    protocol_path = ROOT/'runs/validation/independent-acceptance/p2-regression/protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    inputs = {'protocol': protocol_path, 'data': Path(protocol['data']),
              'manifest': Path(protocol['data']).parent/'manifest.json',
              'model': Path(protocol['model']), 'baseline': Path(protocol['baseline_report'])}
    if (len(protocol['indices']) != 69 or len(set(protocol['indices'])) != 69
            or protocol['outcome_counts'] != {'success':36, 'collision':27, 'oscillation':6}
            or file_digest(inputs['model']) != 'd7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8'
            or file_digest(inputs['data']) != 'ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963'):
        raise ValueError('不是冻结的R0/已知69条回归协议')
    config = {'kind':'offline_mpc_candidate_comparison', 'inputs':{k:{'path':str(v.resolve()), 'sha256':file_digest(v)} for k,v in inputs.items()},
              'code':execution_identity(), 'runtime':runtime_identity(), 'indices':protocol['indices'],
              'seed':0, 'max_steps':600, 'replan_every':10, 'smoke_indices':[164,292], 'smoke_max_steps':60,
              'arms':['original','candidate'], 'automatic_training':False, 'automatic_deployment':False,
              'gates':{'min_successes':39, 'max_collisions':24, 'min_retained_successes':33, 'max_scene_success_losses':1}}
    destination.mkdir(parents=True)
    path = atomic_write_json(destination/'config.json', config)
    print(json.dumps({'config':str(path), 'sha256':file_digest(path)}, ensure_ascii=False))


def load_config(path, expected_sha):
    path = Path(path).resolve()
    if file_digest(path) != expected_sha:
        raise ValueError('配置摘要变化')
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['kind'] != 'offline_mpc_candidate_comparison' or config['runtime'] != runtime_identity() or config['code'] != execution_identity():
        raise ValueError('实验/代码/环境身份变化')
    for entry in config['inputs'].values():
        if file_digest(entry['path']) != entry['sha256']:
            raise ValueError('输入摘要变化')
    return config


def episode_map(report, indices):
    if report['status'] != 'completed' or report['reconstruct_failures']:
        raise ValueError('报告未完成或有任务复原失败')
    rows = {r['meta']['dataset_index']:r for r in report['episodes']}
    if len(rows) != len(report['episodes']) or set(rows) != set(indices):
        raise ValueError('逐任务分母/索引不一致')
    if report['experiment']['name'] != 'E5' or report['experiment']['safety_mode'] != 'none':
        raise ValueError('不是无干预E5')
    return rows


def paired_summary(original, candidate, indices, gates):
    before, after = episode_map(original, indices), episode_map(candidate, indices)
    counts = lambda rows: {'success':sum(r['success'] for r in rows.values()),
                           'collision':sum(r['collision'] for r in rows.values()),
                           'oscillation':sum(r['failure']=='oscillation' for r in rows.values())}
    losses = {}
    retained = 0
    for index in indices:
        a, b = before[index], after[index]
        for key in ('task_id','scene_name','tol_pos','tol_yaw'):
            if a['meta'][key] != b['meta'][key]:
                raise ValueError('配对任务身份/容差不一致')
        if a['success']:
            retained += int(b['success'])
            if not b['success']:
                scene = a['meta']['scene_name']
                losses[scene] = losses.get(scene,0)+1
    result = {'original':counts(before), 'candidate':counts(after), 'retained_original_successes':retained,
              'scene_original_success_losses':losses, 'automatic_training':False, 'automatic_deployment':False}
    result['gate_passed'] = (result['candidate']['success'] >= gates['min_successes']
                             and result['candidate']['collision'] <= gates['max_collisions']
                             and retained >= gates['min_retained_successes']
                             and all(n <= gates['max_scene_success_losses'] for n in losses.values()))
    result['decision'] = 'candidate_for_independent_selection_only' if result['gate_passed'] else 'stop_no_expansion_or_deployment'
    return result


def verify_baseline(report, baseline, indices):
    rows = episode_map(report,indices)
    archived = {r['meta']['dataset_index']:r for r in baseline['episodes']}
    for index, actual in rows.items():
        old = archived[index]
        for key in ('task_id','scene_name','tol_pos','tol_yaw'):
            if actual['meta'][key] != old['meta'][key]:
                raise ValueError('原臂未复现历史任务身份')
        if any(actual[k] != old[k] for k in ('success','collision','failure','steps')) or any(
                not np.isclose(actual[k],old[k],atol=1e-9,rtol=0) for k in ('final_pos_err','final_yaw_err')):
            raise ValueError(f'原臂未精确复现历史结果 #{index}')


def run(config_path, expected_sha, *, check_only=False, smoke=False):
    config = load_config(config_path,expected_sha)
    torch.set_num_threads(1)
    if check_only:
        print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True)
        return
    output = Path(config_path).resolve().parent/('preflight' if smoke else 'results')
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/'identity.json',{'config_sha256':expected_sha,'config':config,'smoke':smoke})
    indices = config['smoke_indices'] if smoke else config['indices']
    reports = {}
    started = time.perf_counter()
    try:
        for arm in config['arms']:
            print(f'ARM {arm} BEGIN samples={len(indices)} smoke={smoke}',flush=True)
            with solver_scope(arm):
                report = matrix.run_validation_experiment('E5',data_path=config['inputs']['data']['path'],
                    checkpoint_path=config['inputs']['model']['path'],indices=indices,samples=0,
                    max_steps=config['smoke_max_steps'] if smoke else config['max_steps'],
                    replan_every=config['replan_every'],control_seed=config['seed'],
                    progress=lambda n,total,b:print(f'{arm} [{n}/{total}] #{b.result.meta["dataset_index"]} {b.result.failure or "success"}',flush=True))
            episode_map(report,indices)
            report['offline_solver_trial'] = {'arm':arm, 'solver_class':('MPCController' if arm=='original' else 'CandidatePreservingMPC'),
                                               'production_default_unchanged':True, 'extra_cost_sequences_per_solve':0 if arm=='original' else 3}
            atomic_write_json(output/f'{arm}.json',report)
            reports[arm] = report
            if not smoke and arm=='original':
                baseline = json.loads(Path(config['inputs']['baseline']['path']).read_text(encoding='utf-8'))
                verify_baseline(report,baseline,indices)
                print('ORIGINAL 69 HISTORICAL RESULTS MATCHED',flush=True)
        summary = paired_summary(reports['original'],reports['candidate'],indices,config['gates'])
        if smoke:
            summary['gate_passed'] = False
            summary['decision'] = 'bounded_preflight_not_formal_evidence'
        load_config(config_path,expected_sha)
        atomic_write_json(output/'report.json',{'status':'completed','smoke':smoke,'elapsed_sec':time.perf_counter()-started,
                                               'summary':summary,'config_sha256':expected_sha})
        print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as exc:
        atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_arms':list(reports)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--config-sha256')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.config or args.config_sha256 or args.check_only or args.smoke:
            parser.error('prepare不能混用运行参数')
        prepare(args.prepare)
    else:
        if not args.config or not args.config_sha256:
            parser.error('必须指定配置及冻结摘要')
        run(args.config,args.config_sha256,check_only=args.check_only,smoke=args.smoke)


if __name__=='__main__':
    main()
