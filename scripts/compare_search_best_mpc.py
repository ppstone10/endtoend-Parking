"""当前搜索最佳样本的独立冻结双臂入口，不修改上一版试验。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from controller import MPCController
from experiments import validation_matrix as matrix
from experiments.mpc_search_best_solver import SearchBestMPC
from scripts import compare_mpc_solvers as common
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest,runtime_identity
from training.reporting import atomic_write_json

PRIOR_HASH='45246333b17039c3a289420721773c13629ec39ea8f17209dccc9bf721686ec8'


def execution_identity():
    return {**code_identity(),str(Path(common.__file__).resolve()):file_digest(common.__file__),
            str(Path(__file__).resolve()):file_digest(__file__)}


def prepare(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('必须使用新目录')
    prior=ROOT/'runs/validation/mpc-candidate-ab-20261019-v1/config.json'
    if file_digest(prior)!=PRIOR_HASH:raise ValueError('上一试验配置身份变化')
    config=json.loads(prior.read_text(encoding='utf-8'))
    # 新增文件会改变全集，但原86个文件必须逐项保持不变，历史身份不重写。
    if any(file_digest(p)!=h for p,h in config['code'].items()):raise ValueError('上一试验执行源被修改')
    for entry in config['inputs'].values():
        if file_digest(entry['path'])!=entry['sha256']:raise ValueError('上一试验输入被修改')
    config.update(kind='offline_mpc_search_best_comparison',solver_variant='current_search_best',
                  code=execution_identity(),runtime=runtime_identity())
    config['inputs']['prior_config']={'path':str(prior),'sha256':PRIOR_HASH}
    destination.mkdir(parents=True)
    path=atomic_write_json(destination/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path)},ensure_ascii=False))


def load_config(path,expected_sha):
    path=Path(path).resolve()
    if file_digest(path)!=expected_sha:raise ValueError('配置摘要变化')
    config=json.loads(path.read_text(encoding='utf-8'))
    if (config['kind']!='offline_mpc_search_best_comparison' or config['solver_variant']!='current_search_best'
            or config['code']!=execution_identity() or config['runtime']!=runtime_identity()):
        raise ValueError('方案/代码/环境变化')
    for entry in config['inputs'].values():
        if file_digest(entry['path'])!=entry['sha256']:raise ValueError('输入摘要变化')
    return config


@contextmanager
def solver_scope(arm,output):
    if arm not in ('original','candidate'):raise ValueError('未知试验臂')
    old_mpc,old_engine=matrix.MPCController,matrix.ClosedLoopEngine

    class AuditEngine(old_engine):
        def run(self,start,goal):
            records=[]
            solve=self.mpc._solve
            def observe(warm,ref,state):
                selected=solve(warm,ref,state)
                records.append({'step':len(records),'state_before':state.to_array().tolist(),
                                'window':ref.tolist(),'first_command':selected[:2].tolist(),
                                **self.mpc.last_search_audit})
                return selected
            self.mpc._solve=observe
            try:result=super().run(start,goal)
            finally:self.mpc._solve=solve
            if len(records)!=result.steps:raise ValueError('控制步与审计账本数量不一致')
            atomic_write_json(Path(output)/'audits'/f'task-{self.meta["dataset_index"]:04d}.json',
                              {'dataset_index':self.meta['dataset_index'],'task_id':self.meta['task_id'],
                               'solver_class':'SearchBestMPC','steps':result.steps,'records':records})
            return result

    matrix.MPCController=MPCController if arm=='original' else SearchBestMPC
    if arm=='candidate':matrix.ClosedLoopEngine=AuditEngine
    try:yield
    finally:matrix.MPCController,matrix.ClosedLoopEngine=old_mpc,old_engine


def audit_outputs(report,output,indices):
    episodes=common.episode_map(report,indices)
    identities={}
    for index,episode in episodes.items():
        path=Path(output)/'audits'/f'task-{index:04d}.json'
        audit=json.loads(path.read_text(encoding='utf-8'))
        if (audit['dataset_index']!=index or audit['task_id']!=episode['meta']['task_id']
                or audit['solver_class']!='SearchBestMPC' or audit['steps']!=episode['steps']
                or len(audit['records'])!=episode['steps']):
            raise ValueError('试验账本身份/步数不一致')
        for step,row in enumerate(audit['records']):
            if (row['step']!=step or row['selected'] not in ('legacy_mean','search_sample')
                    or row['selected_cost']>row['legacy_mean_cost']
                    or row['selected_cost']!=min(row['legacy_mean_cost'],row['best_sample_cost'])):
                raise ValueError('试验账本选择或代价不一致')
        identities[str(index)]={'path':str(path.resolve()),'sha256':file_digest(path)}
    return identities


def run(path,expected_sha,*,check_only=False,smoke=False):
    config=load_config(path,expected_sha);torch.set_num_threads(1)
    if check_only:
        print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True);return
    output=Path(path).resolve().parent/('preflight' if smoke else 'results')
    output.mkdir(parents=True,exist_ok=False)
    (output/'audits').mkdir()
    atomic_write_json(output/'identity.json',{'config_sha256':expected_sha,'config':config,'smoke':smoke})
    indices=config['smoke_indices'] if smoke else config['indices']
    reports={};started=time.perf_counter()
    try:
        for arm in config['arms']:
            print(f'ARM {arm} BEGIN samples={len(indices)} smoke={smoke}',flush=True)
            with solver_scope(arm,output):
                report=matrix.run_validation_experiment('E5',data_path=config['inputs']['data']['path'],
                    checkpoint_path=config['inputs']['model']['path'],indices=indices,samples=0,
                    max_steps=config['smoke_max_steps'] if smoke else config['max_steps'],
                    replan_every=config['replan_every'],control_seed=config['seed'],
                    progress=lambda n,total,b:print(f'{arm} [{n}/{total}] #{b.result.meta["dataset_index"]} {b.result.failure or "success"}',flush=True))
            common.episode_map(report,indices)
            report['offline_solver_trial']={'arm':arm,'solver_variant':'current_search_best',
                'solver_class':'MPCController' if arm=='original' else 'SearchBestMPC',
                'extra_cost_sequences_per_solve':0 if arm=='original' else 1,'production_default_unchanged':True}
            if arm=='candidate':report['solver_audit_files']=audit_outputs(report,output,indices)
            atomic_write_json(output/f'{arm}.json',report);reports[arm]=report
            if arm=='original' and not smoke:
                baseline=json.loads(Path(config['inputs']['baseline']['path']).read_text(encoding='utf-8'))
                common.verify_baseline(report,baseline,indices)
                print('ORIGINAL 69 HISTORICAL RESULTS MATCHED',flush=True)
        summary=common.paired_summary(reports['original'],reports['candidate'],indices,config['gates'])
        if smoke:summary.update(gate_passed=False,decision='bounded_preflight_not_formal_evidence')
        load_config(path,expected_sha)
        atomic_write_json(output/'report.json',{'status':'completed','smoke':smoke,'config_sha256':expected_sha,
                                               'elapsed_sec':time.perf_counter()-started,'summary':summary})
        print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as exc:
        atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_arms':list(reports)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--config',type=Path);parser.add_argument('--config-sha256')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--check-only',action='store_true');mode.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        if args.config or args.config_sha256 or args.check_only or args.smoke:parser.error('prepare参数不可混用')
        prepare(args.prepare)
    else:
        if not args.config or not args.config_sha256:parser.error('必须指定冻结配置及摘要')
        run(args.config,args.config_sha256,check_only=args.check_only,smoke=args.smoke)


if __name__=='__main__':main()
