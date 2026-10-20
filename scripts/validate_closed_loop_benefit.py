"""一次固定R0/R2e2独立300条配对E5；不训练、不改控制器、不自动晋升。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from controller import MPCController
from experiments import validation_matrix as matrix
from experiments.closed_loop_benefit import summarize_pair, validate_report
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest, metadata_only, runtime_identity
from scripts.train_fixed_threshold import exclusive_run
from training.checkpoint import load_model_checkpoint
from training.reporting import atomic_write_json

FIXED={
    'baseline':('runs/validation/arch-ab/dep-epoch0007.pt','d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8'),
    'candidate':('runs/training/recovery-balance-ab-20261018-v2/R2/checkpoints/epoch0002.pt','fe258ad0186a0ce40717befbc96452bd2f959ea2e97293b323aaa2a33f26b620'),
    'data':('data/task_dataset/independent_selection_v1/selection.npz','797e9e5d0d02576c45df65c74f142cc05e83770eb6bbfcc28be560dbc722d51d'),
    'manifest':('data/task_dataset/independent_selection_v1/manifest.json','a4c804ec327220d3698ade8ec891c43295a408a51d9afa470470f640ab0a1a5b'),
    'source_report':('runs/training/recovery-balance-ab-20261018-v2/R2/report.json','d30667ca6d2421a7d084963f1041e992523cb0eb78d30cd6bd6b450c5a25b754'),
    'old_failed_screen':('runs/validation/recovery-balance-ab-20261019-regression/summary.md','7ee81312958d75803f795e79854bc411f3011cc3f31a7f65e2f1b08bbaaf6b53')}


def execution_identity():
    return {**code_identity(),**{str(p.resolve()):file_digest(p) for p in
            (Path(__file__),ROOT/'scripts/build_selection_dataset.py',ROOT/'scripts/train_fixed_threshold.py')}}


def prepare(destination):
    destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError('拒绝覆盖任务目录')
    inputs={k:{'path':str(ROOT/p),'sha256':sha} for k,(p,sha) in FIXED.items()}
    for value in inputs.values():
        if file_digest(value['path'])!=value['sha256']: raise ValueError('固定来源身份变化')
    source=json.loads(Path(inputs['source_report']['path']).read_text(encoding='utf-8'))
    if source['status']!='completed' or source['arm']!='R2' or source['snapshots'][1]!=inputs['candidate']:
        raise ValueError('R2来源训练报告不符')
    train=Path(source['config']['train'])
    inputs['training_data']={'path':str(train),'sha256':file_digest(train)}
    metadata=metadata_only(inputs['data']['path']); train_meta=metadata_only(train)
    if (len(metadata)!=300 or any(m['scene_name'].startswith('S9') for m in metadata)
            or collect_metadata_ids(metadata)&collect_metadata_ids(train_meta)):
        raise ValueError('选型分母/S9/训练来源交集不符')
    if json.loads(Path(inputs['manifest']['path']).read_text(encoding='utf-8'))['purpose']!='checkpoint_selection_only':
        raise ValueError('非选型集')
    baseline=load_model_checkpoint(inputs['baseline']['path']); candidate=load_model_checkpoint(inputs['candidate']['path'])
    if (baseline.model_config!=candidate.model_config or baseline.model_name!=candidate.model_name
            or candidate.epoch!=1 or baseline.model.stop_threshold!=.26000000000000006):
        raise ValueError('候选模型配置/轮次/阈值不一致')
    smoke_indices=[next(i for i,m in enumerate(metadata) if m['scene_name'].startswith(scene)) for scene in ('S3','S4','S8')]
    config={'kind':'fixed_closed_loop_benefit_trial','inputs':inputs,'code':execution_identity(),'runtime':runtime_identity(),
            'indices':list(range(300)),'smoke_indices':smoke_indices,'smoke_max_steps':20,'max_steps':600,
            'replan_every':10,'seed':0,'meaningful_count':9,'arms':['baseline','candidate'],
            'model_config':baseline.model_config,'model_name':baseline.model_name,
            'task_identities':[{'task_id':m['task_id'],'scene_name':m['scene_name']} for m in metadata],
            'old_screen_failed':True,'single_fixed_reconsideration_only':True,
            'training_selection_overlap':0,'automatic_training':False,'automatic_deployment':False}
    destination.mkdir(parents=True,exist_ok=False)
    path=atomic_write_json(destination/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path),'smoke_indices':smoke_indices},ensure_ascii=False),flush=True)
    return config


def load_config(path,expected_sha):
    path=Path(path).resolve()
    if file_digest(path)!=expected_sha: raise ValueError('配置SHA漂移')
    config=json.loads(path.read_text(encoding='utf-8'))
    if (config['kind']!='fixed_closed_loop_benefit_trial' or config['code']!=execution_identity()
            or config['runtime']!=runtime_identity() or config['indices']!=list(range(300))
            or config['arms']!=['baseline','candidate'] or config['max_steps']!=600
            or config['replan_every']!=10 or config['seed']!=0 or config['meaningful_count']!=9):
        raise ValueError('任务/执行源/环境/协议漂移')
    for value in config['inputs'].values():
        if file_digest(value['path'])!=value['sha256']: raise ValueError('输入SHA漂移')
    if matrix.MPCController is not MPCController: raise ValueError('原MPC被替换')
    return config


def validate_frozen_report(report,config,arm,smoke):
    indices=config['smoke_indices'] if smoke else config['indices']
    mapped=validate_report(report,indices)
    protocol=report['protocol']
    expected={'data':config['inputs']['data']['path'],'checkpoint':config['inputs'][arm]['path'],
              'max_steps':config['smoke_max_steps'] if smoke else config['max_steps'],
              'replan_every':config['replan_every'],'control_seed':config['seed'],'mpc_dt':.1,
              'model_config':config['model_config'],'model_name':config['model_name']}
    if any(protocol.get(k)!=v for k,v in expected.items()): raise ValueError('臂报告与冻结协议不符')
    for i,row in mapped.items():
        if any(row['meta'][k]!=v for k,v in config['task_identities'][i].items()):
            raise ValueError('任务身份与冻结清单不符')


def run(config_path,expected_sha,*,check_only=False,smoke=False,resume=False):
    config=load_config(config_path,expected_sha)
    if check_only:
        print('INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED',flush=True);return
    if smoke and resume: raise ValueError('冒烟不能resume')
    output=Path(config_path).resolve().parent/('preflight' if smoke else 'results')
    if resume:
        if not output.exists(): raise FileNotFoundError('无可恢复输出')
    else: output.mkdir(parents=True,exist_ok=False)
    if (output/'report.json').exists(): raise FileExistsError('已完成，不重复运行')
    identity={'config_sha256':expected_sha,'smoke':smoke}
    reports={};receipts={};started=time.perf_counter();torch.set_num_threads(1)
    with exclusive_run(output):
        if resume:
            if json.loads((output/'identity.json').read_text(encoding='utf-8'))!=identity: raise ValueError('恢复身份不符')
            state=json.loads((output/'status.json').read_text(encoding='utf-8'))
            receipts=state['completed_arms']
            if not set(receipts)<=set(config['arms']): raise ValueError('恢复臂清单非法')
            for arm in config['arms']:
                path=output/(arm+'.json')
                if path.exists()!=(arm in receipts): raise ValueError('存在未登记/缺失报告，停止并反馈')
                if arm in receipts:
                    if file_digest(path)!=receipts[arm]: raise ValueError('完整臂SHA漂移')
                    reports[arm]=json.loads(path.read_text(encoding='utf-8'));validate_frozen_report(reports[arm],config,arm,smoke)
        else: atomic_write_json(output/'identity.json',identity)
        try:
            atomic_write_json(output/'status.json',{'status':'running','completed_arms':receipts})
            for arm in config['arms']:
                if arm in reports:
                    print(f'REUSE VERIFIED COMPLETE ARM {arm}',flush=True);continue
                indices=config['smoke_indices'] if smoke else config['indices']
                print(f'ARM {arm} BEGIN samples={len(indices)} smoke={smoke}',flush=True)
                report=matrix.run_validation_experiment('E5',data_path=config['inputs']['data']['path'],
                        checkpoint_path=config['inputs'][arm]['path'],indices=indices,samples=0,
                        max_steps=config['smoke_max_steps'] if smoke else config['max_steps'],
                        replan_every=config['replan_every'],control_seed=config['seed'],
                        progress=lambda n,total,b:print(f'{arm} [{n}/{total}] #{b.result.meta["dataset_index"]} {b.result.failure or "success"}',flush=True))
                validate_frozen_report(report,config,arm,smoke)
                path=atomic_write_json(output/(arm+'.json'),report)
                receipts[arm]=file_digest(path);reports[arm]=report
                atomic_write_json(output/'status.json',{'status':'running','completed_arms':receipts})
            load_config(config_path,expected_sha)
            summary=summarize_pair(reports['baseline'],reports['candidate'],indices,
                                   meaningful_count=config['meaningful_count'],smoke=smoke)
            atomic_write_json(output/'report.json',{'status':'completed','smoke':smoke,'config_sha256':expected_sha,
                              'report_sha256':receipts,'elapsed_seconds_this_invocation':time.perf_counter()-started,'summary':summary})
            atomic_write_json(output/'status.json',{'status':'completed','completed_arms':receipts})
            print('COMPLETE '+json.dumps(summary,ensure_ascii=False),flush=True)
        except BaseException as exc:
            atomic_write_json(output/'status.json',{'status':'interrupted_or_failed','error':str(exc),'completed_arms':receipts})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--config',type=Path);parser.add_argument('--config-sha256')
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        if args.config or args.config_sha256 or args.check_only or args.smoke or args.resume:parser.error('prepare不可混用运行选项')
        prepare(args.prepare)
    else:
        if not args.config or not args.config_sha256 or args.check_only and (args.smoke or args.resume):parser.error('须指定config/SHA，check-only独立使用')
        run(args.config,args.config_sha256,check_only=args.check_only,smoke=args.smoke,resume=args.resume)
