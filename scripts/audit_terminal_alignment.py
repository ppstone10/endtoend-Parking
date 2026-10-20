"""冻结并人工运行原train停点与目标联合对齐审计，不训练。"""
from __future__ import annotations
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
from experiments.terminal_alignment_audit import analyze_terminal_alignment
from metrics.prediction_analysis import collect_open_loop_predictions
from scripts.diagnose_reference_execution import code_identity
from scripts.prepare_recovery_balance_ab import file_digest,runtime_identity
from training.checkpoint import load_model_checkpoint
from training.data import prepare_batches
from training.reporting import atomic_write_json

PRIOR=ROOT/'runs/diagnostics/cem-polishing-ab-20261020-v1/config.json'
PRIOR_SHA='b66fcb7adda57629cb9454324d13c946f6aa8ef243e38dafaf7b5d57fcec96d1'
REPORT_SHA='5cb3c9eefa3866278fd12469cb44cf8efd729cac05e391d02025c7a00d068fb9'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def execution_identity():
    return {**read_json(PRIOR)['code'],**code_identity(),str(Path(__file__).resolve()):file_digest(__file__)}


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('必须使用新目录')
    if file_digest(PRIOR)!=PRIOR_SHA or file_digest(PRIOR.parent/'results/report.json')!=REPORT_SHA:raise ValueError('LT10证据身份变化')
    old=read_json(PRIOR)
    for path,digest in old['code'].items():
        if file_digest(path)!=digest:raise ValueError('旧执行源变化')
    inputs={key:old['inputs'][key] for key in ('train','train_manifest','model')}
    inputs.update({'lt10_config':{'path':str(PRIOR),'sha256':PRIOR_SHA},'lt10_report':{'path':str(PRIOR.parent/'results/report.json'),'sha256':REPORT_SHA}})
    for item in inputs.values():
        if file_digest(item['path'])!=item['sha256']:raise ValueError('输入身份变化')
    config={'kind':'train_terminal_alignment_audit','inputs':inputs,'code':execution_identity(),'runtime':runtime_identity(),
            'expected_samples':2400,'smoke_indices':[1298,1455,1559,1441,1334,1680],'batch_size':8,
            'output':str(output/'results'),
            'automatic_training':False,'automatic_deployment':False,'automatic_expansion':False}
    config['threshold']=load_model_checkpoint(inputs['model']['path']).model_config['stop_threshold']
    output.mkdir(parents=True);path=atomic_write_json(output/'config.json',config)
    print(json.dumps({'config':str(path),'sha256':file_digest(path),'inputs':len(inputs),'code':len(config['code'])}))


def load_config(path,sha):
    if file_digest(path)!=sha:raise ValueError('配置身份变化')
    c=read_json(path)
    if c['kind']!='train_terminal_alignment_audit' or c['code']!=execution_identity() or c['runtime']!=runtime_identity():raise ValueError('代码或环境变化')
    for item in c['inputs'].values():
        if file_digest(item['path'])!=item['sha256']:raise ValueError('输入身份变化')
    if c['expected_samples']!=2400 or c['batch_size']!=8:raise ValueError('范围或批量变化')
    return c


def run(path,sha,*,smoke=False,check_only=False):
    c=load_config(path,sha)
    if check_only:print('INPUT/CODE/RUNTIME VERIFIED; NO INFERENCE STARTED');return
    destination=Path(path).resolve().parent/'preflight' if smoke else Path(c['output'])
    destination.mkdir(parents=True,exist_ok=False);atomic_write_json(destination/'identity.json',{'config':c,'config_sha256':sha,'smoke':smoke})
    started=time.perf_counter();torch.set_num_threads(1)
    try:
        data=DatasetGenerator.load(c['inputs']['train']['path'])
        if len(data['task_meta'])!=2400:raise ValueError('不是固定原train')
        indices=c['smoke_indices'] if smoke else list(range(2400))
        selected={**data,**{key:np.asarray(data[key])[indices] for key in ('bevs','goals','states','trajs','masks')}}
        selected['task_meta']=[data['task_meta'][i] for i in indices]
        model=load_model_checkpoint(c['inputs']['model']['path']).model
        if model.stop_threshold!=c['threshold']:raise ValueError('阈值变化')
        batches=prepare_batches(selected,horizon=model.max_horizon,batch_size=c['batch_size'])
        goals=np.concatenate([b[1].numpy() for b in batches])
        def progress_batches():
            for j,batch in enumerate(batches):
                if j%25==0:print(f'INFERENCE batch={j+1}/{len(batches)} samples={len(indices)} smoke={smoke}',flush=True)
                yield batch
        prediction=collect_open_loop_predictions(model,progress_batches())
        rows,summary=analyze_terminal_alignment(prediction.points,prediction.stop_logits,prediction.targets,prediction.masks,goals,selected['task_meta'],threshold=model.stop_threshold,indices=indices)
        np.savez_compressed(destination/'predictions.npz',points=prediction.points,logits=prediction.stop_logits,targets=prediction.targets,masks=prediction.masks,goals=goals,indices=np.array(indices))
        atomic_write_json(destination/'samples.json',rows);load_config(path,sha)
        atomic_write_json(destination/'report.json',{'status':'completed','smoke':smoke,'config_sha256':sha,'elapsed_sec':time.perf_counter()-started,
            'samples_sha256':file_digest(destination/'samples.json'),'predictions_sha256':file_digest(destination/'predictions.npz'),
            'threshold':model.stop_threshold,'summary':summary})
        print('COMPLETED '+json.dumps(summary),flush=True)
    except BaseException as exc:
        atomic_write_json(destination/'status.json',{'status':'failed_or_interrupted','error':str(exc)});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',type=Path);p.add_argument('--config',type=Path);p.add_argument('--config-sha256')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--smoke',action='store_true');modes.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.prepare:
        if a.config or a.config_sha256 or a.smoke or a.check_only:p.error('prepare不可与运行参数混用')
        prepare(a.prepare)
    else:
        if not a.config or not a.config_sha256:p.error('需要配置及SHA256')
        run(a.config,a.config_sha256,smoke=a.smoke,check_only=a.check_only)


if __name__=='__main__':main()
