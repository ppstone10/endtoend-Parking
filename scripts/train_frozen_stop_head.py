"""准备/人工执行冻结轨迹生成的停止头双臂，不自动评测闭环或部署。"""
from __future__ import annotations
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset import DatasetGenerator
from experiments.frozen_stop_head import HEAD_KEYS,backbone_digest,capture_stop_features,fit_stop_head,make_head_checkpoint
from experiments.terminal_alignment_audit import analyze_terminal_alignment
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import assert_original_equivalence,file_digest,metadata_only,runtime_identity
from training.checkpoint import load_model_checkpoint
from training.data import prepare_batches
from training.reporting import atomic_write_json

PRIOR=ROOT/'runs/diagnostics/terminal-alignment-train-20261020-v1/config.json'
PRIOR_SHA='57092907eab527c28ace9e42c9e93119ccd3871251fc8640cd2a46d10ffcb641'
REPORT_SHA='c6a99f711603dabf691ff3155adea59ee85797ea8f6ec996c0a4ea22c23cf09b'
SETTINGS={'epochs':20,'learning_rate':2e-4,'batch_size':8,'seed':20261020}
TRAIN_SHA='b9531bbfc3577b9a8e7db3e622cb2578fc0d3cb926fc792dfa0eadb5cf2cb8cd'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def execution_identity():
    return {**read_json(PRIOR)['code'],**code_identity(),str(Path(__file__).resolve()):file_digest(__file__)}


def save_torch(path,payload):
    path=Path(path)
    if path.exists():raise FileExistsError(str(path))
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.partial')
    torch.save(payload,temporary);temporary.replace(path)
    return str(path)


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('要求全新目录')
    if file_digest(PRIOR)!=PRIOR_SHA or file_digest(PRIOR.parent/'results/report.json')!=REPORT_SHA:raise ValueError('LT11身份变化')
    prior=read_json(PRIOR)
    for path,digest in prior['code'].items():
        if file_digest(path)!=digest:raise ValueError('旧99执行源变化')
    inputs=deepcopy(prior['inputs'])
    additions={'training_balanced':ROOT/'data/task_dataset/v40_dagger_collision/train-balanced.npz',
               'validation_ids':ROOT/'data/task_dataset/tracked_pivot_v8_3000/val.npz',
               'selection_ids':ROOT/'data/task_dataset/independent_selection_v1/selection.npz',
               'lt11_config':PRIOR,'lt11_report':PRIOR.parent/'results/report.json',
               'lt11_predictions':PRIOR.parent/'results/predictions.npz'}
    for name,path in additions.items():inputs[name]={'path':str(path),'sha256':file_digest(path)}
    for item in inputs.values():
        if file_digest(item['path'])!=item['sha256']:raise ValueError('输入身份变化')
    if inputs['training_balanced']['sha256']!=TRAIN_SHA:raise ValueError('不是冻结2918训练池')
    source=DatasetGenerator.load(inputs['training_balanced']['path']);original=DatasetGenerator.load(inputs['train']['path'])
    if len(source['task_meta'])!=2918 or len(original['task_meta'])!=2400:raise ValueError('数据数量不符')
    if (np.any(np.asarray(source['masks']).sum(1)>320) or not np.isfinite(source['dt']).all()
            or not np.allclose(source['dt'],.2)):raise ValueError('训练标签超出horizon或dt不符')
    assert_original_equivalence(source,original)
    if [i for i,m in enumerate(source['task_meta']) if not m.get('recovery')]!=list(range(2400)):raise ValueError('原任务不是固定前2400')
    if any(m['scene_name'].startswith('S9') for m in source['task_meta']):raise ValueError('禁止S9')
    ids=collect_metadata_ids(source['task_meta']);overlaps={name:len(ids&collect_metadata_ids(metadata_only(inputs[name]['path']))) for name in ('validation_ids','selection_ids')}
    if any(overlaps.values()):raise ValueError('训练来源与评测任务重叠')
    loaded=load_model_checkpoint(inputs['model']['path'])
    if loaded.model_name!='net-v1' or loaded.model.max_horizon!=320 or loaded.model.dt!=.2 or loaded.model.stop_threshold!=prior['threshold']:raise ValueError('父模型契约不符')
    config={'kind':'frozen_stop_head_guided_free_ab','inputs':inputs,'code':execution_identity(),'runtime':runtime_identity(),
            'settings':SETTINGS,'threshold':loaded.model.stop_threshold,'expected_samples':2918,'original_samples':2400,
            'smoke_indices':[1298,1455,1559,1441,1334,1680,2400,2917],'task_identity_overlaps':overlaps,
            'output':str(output/'results'),'automatic_closed_loop_evaluation':False,'automatic_deployment':False,'automatic_expansion':False}
    output.mkdir(parents=True);path=atomic_write_json(output/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path),'inputs':len(inputs),'code':len(config['code']),'overlaps':overlaps}))


def load_config(path,sha):
    if file_digest(path)!=sha:raise ValueError('配置摘要变化')
    c=read_json(path)
    if c['kind']!='frozen_stop_head_guided_free_ab' or c['code']!=execution_identity() or c['runtime']!=runtime_identity():raise ValueError('代码/运行时变化')
    for item in c['inputs'].values():
        if file_digest(item['path'])!=item['sha256']:raise ValueError('输入身份变化')
    if c['settings']!=SETTINGS or c['expected_samples']!=2918 or c['original_samples']!=2400:raise ValueError('训练预算/范围变化')
    return c


def evaluate_head(head,cache,metadata,destination,*,threshold,count,source_indices):
    """逐时刻同形状Linear求值，避免向量化GEMM改变停止交叉舍入。"""
    logits=[]
    with torch.no_grad():
        for begin in range(0,count,8):
            features=cache['features'][begin:min(begin+8,count)]
            logits.append(torch.stack([head(features[:,j].contiguous()).squeeze(-1) for j in range(features.shape[1])],dim=1))
    logits=torch.cat(logits)
    rows,summary=analyze_terminal_alignment(cache['points'][:count].numpy(),logits.numpy(),cache['targets'][:count].numpy(),
          cache['masks'][:count].numpy(),cache['goals'][:count].numpy(),metadata[:count],threshold=threshold,indices=source_indices[:count])
    path=atomic_write_json(destination/'samples.json',rows)
    np.savez_compressed(destination/'stop-logits.npz',logits=logits.numpy())
    return {'summary':summary,'samples_sha256':file_digest(path),'logits_sha256':file_digest(destination/'stop-logits.npz')}


def verify_export(parent,payload,loaded,batch,baseline_points):
    if payload['model_config']!=parent['model_config']:raise ValueError('模型配置或阈值变化')
    for name,tensor in parent['model_state'].items():
        if name not in HEAD_KEYS and not torch.equal(tensor,payload['model_state'][name]):raise ValueError('冻结张量变化')
    with torch.no_grad():points=loaded.forward_with_stop(*batch[:3]).points
    if not torch.equal(points,baseline_points):raise ValueError('完整320点变化')
    return {'all_non_head_tensors_equal':True,'probe_points_bitwise_equal':True,'probe_samples':len(points),'head_parameters':sum(p.numel() for p in loaded.decoder.stop_head.parameters())}


def run(path,sha,*,smoke=False,check_only=False):
    c=load_config(path,sha)
    if check_only:print('INPUT/CODE/RUNTIME VERIFIED; NO TRAINING STARTED');return
    destination=Path(path).resolve().parent/'preflight' if smoke else Path(c['output']);destination.mkdir(parents=True,exist_ok=False)
    atomic_write_json(destination/'identity.json',{'config':c,'config_sha256':sha,'smoke':smoke});started=time.perf_counter();torch.set_num_threads(1);completed=[]
    try:
        source=DatasetGenerator.load(c['inputs']['training_balanced']['path'])
        if len(source['task_meta'])!=2918:raise ValueError('训练池数量变化')
        indices=c['smoke_indices'] if smoke else list(range(2918))
        data={**source,**{key:np.asarray(source[key])[indices] for key in ('bevs','goals','states','trajs','masks')}}
        metadata=[source['task_meta'][i] for i in indices];batches=prepare_batches(data,horizon=320,batch_size=8)
        parent=torch.load(c['inputs']['model']['path'],map_location='cpu',weights_only=False);model=load_model_checkpoint(c['inputs']['model']['path']).model
        if model.stop_threshold!=c['threshold']:raise ValueError('父阈值变化')
        fingerprint=backbone_digest(model);caches={};cache_sha={}
        for mode in ('guided','free'):
            print('CACHE '+mode+' BEGIN',flush=True)
            def progress(batch_number):
                if batch_number%25==0:print(f'CACHE {mode} batch={batch_number}/{len(batches)}',flush=True)
            caches[mode]=capture_stop_features(model,batches,mode,progress=progress)
            cache_path=save_torch(destination/('cache-'+mode+'.pt'),caches[mode]);cache_sha[mode]=file_digest(cache_path)
        if backbone_digest(model)!=fingerprint:raise ValueError('缓存过程改变主干')
        free=caches['free'];count=len(indices) if smoke else 2400
        if not smoke:
            archived=np.load(c['inputs']['lt11_predictions']['path'])
            for key in ('points','logits','targets','masks','goals'):
                np.testing.assert_array_equal(free[key][:2400].numpy(),archived[key])
        parent_root=destination/'parent';parent_root.mkdir();baseline=evaluate_head(model.decoder.stop_head,free,metadata,parent_root,threshold=c['threshold'],count=count,source_indices=indices)
        if not smoke and baseline['summary']!=read_json(c['inputs']['lt11_report']['path'])['summary']:raise ValueError('原2400基线未复现LT11')
        report={'status':'running','smoke':smoke,'config_sha256':sha,'training_samples':len(indices),'training_indices':indices,'metric_samples':count,
                'backbone_sha256':fingerprint,'cache_sha256':cache_sha,'parent':baseline,'arms':{},'fixed_threshold':c['threshold'],
                'automatic_closed_loop_evaluation':False,'automatic_expansion':False,'automatic_deployment':False}
        for mode in ('guided','free'):
            arm_root=destination/mode;arm_root.mkdir();epochs=1 if smoke else SETTINGS['epochs']
            def checkpoint_epoch(epoch,head,history):
                payload=make_head_checkpoint(parent,head,epoch=epoch,metadata={'arm':mode,'feature_policy':mode,'config_sha256':sha,
                    'cache_sha256':cache_sha[mode],'backbone_sha256':fingerprint,'loss_history':history,'training_samples':len(indices),'smoke':smoke,
                    'parent_sha256':c['inputs']['model']['sha256'],'settings':{**SETTINGS,'epochs':epochs}})
                save_torch(arm_root/f'epoch-{epoch:04d}.pt',payload)
                print(f'TRAIN {mode} epoch={epoch}/{epochs} loss={history[-1]:.8f} smoke={smoke}',flush=True)
            _,history=fit_stop_head(model.decoder.stop_head,caches[mode],**{**SETTINGS,'epochs':epochs},on_epoch=checkpoint_epoch)
            checkpoint=arm_root/f'epoch-{epochs:04d}.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            reloaded=load_model_checkpoint(checkpoint).model
            integrity=verify_export(parent,payload,reloaded,batches[0],free['points'][:len(batches[0][0])])
            metrics=evaluate_head(reloaded.decoder.stop_head,free,metadata,arm_root,threshold=c['threshold'],count=count,source_indices=indices)
            if metrics['summary']['categories']['no_joint_point']!=baseline['summary']['categories']['no_joint_point']:raise ValueError('冻结轨迹无联合点数不应变化')
            report['arms'][mode]={'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':file_digest(checkpoint),'history':history,'integrity':integrity,**metrics}
            completed.append(mode);atomic_write_json(destination/'report.json',report)
        if backbone_digest(model)!=fingerprint:raise ValueError('源模型主干变化')
        load_config(path,sha);report.update({'status':'completed','elapsed_sec':time.perf_counter()-started,'decision':'smoke_integrity_only' if smoke else 'training_complete_requires_independent_review'})
        atomic_write_json(destination/'report.json',report);print('COMPLETE '+report['decision'],flush=True)
    except BaseException as exc:
        atomic_write_json(destination/'status.json',{'status':'failed_or_interrupted','error':str(exc),'completed_arms':completed});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--config',type=Path);p.add_argument('--config-sha256')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--smoke',action='store_true');modes.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.prepare:
        if a.config or a.config_sha256 or a.smoke or a.check_only:p.error('prepare不可混用运行参数')
        prepare(a.prepare)
    else:
        if not a.config or not a.config_sha256:p.error('要求冻结配置及SHA256')
        run(a.config,a.config_sha256,smoke=a.smoke,check_only=a.check_only)


if __name__=='__main__':main()
