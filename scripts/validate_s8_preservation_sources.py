"""用户运行：固定9个保持来源上LT18 policy/新保持学生18完整回合，无训练。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch

from dataset import DatasetGenerator
from experiments.behavior_exploration import paired_counts
from experiments.behavior_exploration_data import evaluate_cases
from experiments.executable_behavior import PreflightBudget
from experiments.exploration_store import artifact_path,exploration_lock
from scripts.explore_closed_loop_learning import load_candidate
from scripts.preflight_behavior_repair import INPUTS,build_case,matrix
from scripts.preflight_learner_failure_teacher import verify_config
from scripts.preflight_retention_sampling import verify_files
from scripts.prepare_recovery_balance_ab import file_digest
from sim import VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json


LEARNING=ROOT/'runs/diagnostics/s8-preservation-learning-preflight-20261022-v1'
HASHES={'config.json':'9929618dc30eb43f4e072877b3f756e1f99e42a47218f45e118ffdb489407ea9',
    'report.json':'c88ba24b4ce2be948306a6166c628e00aa686442ed07613316e5272cb56a6c80'}
AUDIT=ROOT/'runs/diagnostics/s8-preservation-audit-20261022-v1'
LT18=ROOT/'runs/diagnostics/s6-policy-transfer-preflight-20261022-v1'


def prepare_context(budget):
    torch.set_num_threads(1);verify_files({str(LEARNING/k):v for k,v in HASHES.items()})
    previous=json.loads((LEARNING/'report.json').read_text(encoding='utf-8'))
    old=json.loads((LEARNING/'config.json').read_text(encoding='utf-8'));verify_config(old,budget)
    if previous['status']!='completed' or set(previous['completed_seeds'])!={'20261020','20261021'}:
        raise ValueError('不是完整双学生短检')
    for name,digest in previous['artifacts_sha256'].items():verify_files({str(artifact_path(LEARNING,name)):digest})
    source_report=json.loads((AUDIT/'report.json').read_text(encoding='utf-8'))
    parents={seed:[row for row in source_report['source_rows'] if row['seed']==seed and row['query_gate']['passed']]
        for seed in (20261020,20261021)}
    sources={seed:sorted(row['dataset_index'] for row in rows) for seed,rows in parents.items()}
    if sources!={20261020:[2338,2351,2375],20261021:[2330,2338,2351,2370,2378,2398]}:
        raise ValueError('固定9来源集合漂移')
    models={arm:{seed:str((LT18/f'policy_targets-{seed}/state.pt') if arm=='LT18_policy' else
        LEARNING/f'{seed}/state.pt') for seed in sources} for arm in ('LT18_policy','preservation_student')}
    cfg=dict(kind='s8_preservation_source_validation_v1',runtime=old['runtime'],code=dict(old['code']),
        frozen_files=dict(old['frozen_files']),budget_seconds=120,sources=sources,checkpoints=models,
        total_episodes=18,training_updates=0,probe_inputs_used_for_gradient=0,
        main_protocol_passed=False,automatic_deployment=False,allowed_to_prepare_training_plan=False)
    cfg['frozen_files'].update({str(LEARNING/k):v for k,v in HASHES.items()})
    cfg['frozen_files'].update({str(artifact_path(LEARNING,k)):v for k,v in previous['artifacts_sha256'].items()})
    for seed in sources:
        child=LEARNING/str(seed)/'report.json'
        if json.loads(child.read_text(encoding='utf-8'))!=previous['completed_seeds'][str(seed)]:
            raise ValueError('学生子报告与根报告不符')
        cfg['frozen_files'][str(child)]=file_digest(child)
    cfg['code'][str(Path(__file__).resolve())]=file_digest(__file__)
    verify_config(cfg,budget)
    return dict(config=cfg,cached_C240=parents)


def precheck(prepared=None):
    context=prepare_context(PreflightBudget(45))
    if prepared is not None:
        prepared=Path(prepared).resolve()
        if prepared.exists():raise FileExistsError(prepared)
        prepared.parent.mkdir(parents=True,exist_ok=True);atomic_write_json(prepared,context['config'])
    print('18 SOURCE EPISODES/FOUR CHECKPOINTS VERIFIED; ZERO TRAINING/ROLLOUTS',flush=True)
    return context


def run(prepared,output):
    prepared=Path(prepared).resolve();expected=json.loads(prepared.read_text(encoding='utf-8'))
    budget=PreflightBudget(120);start=time.perf_counter();verify_config(expected,budget)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',stage='identity',completed_arms={},training_updates=0,
        main_protocol_passed=False,automatic_deployment=False,allowed_to_prepare_training_plan=False)
    def save():
        report['elapsed_seconds']=time.perf_counter()-start;atomic_write_json(output/'report.json',report)
    with exploration_lock(output):
        save()
        try:
            context=prepare_context(budget);cfg=context['config']
            if json.loads(json.dumps(cfg))!=expected:raise ValueError('冻结配置漂移')
            atomic_write_json(output/'config.json',cfg);report['config_sha256']=file_digest(output/'config.json')
            report['cached_C240']=context['cached_C240'];save()
            data=DatasetGenerator.load(INPUTS['train']);manifest=matrix.load_dataset_manifest(INPUTS['train'])
            vehicle=VehicleConfig(**manifest['vehicle_model']);parent=load_model_checkpoint(INPUTS['R0']).model
            cases={i:build_case(data,manifest,i,vehicle) for i in sorted({i for values in cfg['sources'].values() for i in values})}
            for arm,checkpoints in cfg['checkpoints'].items():
                report['completed_arms'][arm]={}
                for seed,path in checkpoints.items():
                    budget.check();directory=output/f'{arm}-{seed}';directory.mkdir()
                    report['stage']=f'{arm}-{seed}';save();print('EVALUATE',arm,seed,len(cfg['sources'][seed]),flush=True)
                    model,_=load_candidate(path,parent)
                    rows=evaluate_cases(cfg['sources'][seed],cases,model,budget,directory,'source')
                    report['completed_arms'][arm][seed]=rows;save()
            comparisons={}
            for seed in cfg['sources']:
                baseline=report['completed_arms']['LT18_policy'][seed]
                student=report['completed_arms']['preservation_student'][seed]
                comparisons[seed]=dict(to_LT18_policy=paired_counts(baseline,student),
                    to_cached_C240=paired_counts(context['cached_C240'][seed],student),
                    strict_LT18=sum(row['reached_with_margin'] for row in baseline),
                    strict_student=sum(row['reached_with_margin'] for row in student))
            verify_config(cfg,budget);budget.check()
            report.update(status='completed',stage='decision',comparisons=comparisons,
                complete_episodes=sum(len(rows) for by_seed in report['completed_arms'].values() for rows in by_seed.values()),
                decision='source_retention_diagnostic_only_no_promotion')
            if report['complete_episodes']!=18:raise ValueError('完整分母不是18')
            report['artifacts_sha256']={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*')
                if p.is_file() and p!=output/'report.json' and p.name not in ('run.lock','console.log')}
            save();print('FINISHED',report['decision'],flush=True);return report
        except BaseException as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else
                'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),decision='incomplete_do_not_infer')
            save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precheck',action='store_true');p.add_argument('--prepare',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.precheck:
        if a.config or a.output:p.error('precheck只允许可选prepare')
        precheck(a.prepare)
    elif not a.config or not a.output or a.prepare:p.error('需要config和不存在的新output')
    else:run(a.config,a.output)
