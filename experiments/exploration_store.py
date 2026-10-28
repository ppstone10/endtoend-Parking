"""单次探索的有界预算、互斥与可恢复产物单元。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

import torch

from scripts.prepare_recovery_balance_ab import file_digest
from training.reporting import atomic_write_json


class ExplorationBudget:
    def __init__(self,seconds,*,used=0.,progress_path=None):
        if not 0<seconds<=5400 or used<0:raise ValueError('非法探索预算')
        self.limit=seconds;self.used=used;self.started=time.perf_counter()
        self.last_save=self.started;self.progress_path=progress_path

    def elapsed(self):return self.used+time.perf_counter()-self.started

    def save(self):
        if self.progress_path is not None:
            atomic_write_json(self.progress_path,{'active_elapsed_seconds':self.elapsed(),'budget_seconds':self.limit})
        self.last_save=time.perf_counter()

    def check(self):
        if time.perf_counter()-self.last_save>=30:self.save()
        if self.elapsed()>=self.limit:raise TimeoutError('探索总预算耗尽，不自动扩展或续跑')


@contextmanager
def exploration_lock(root):
    path=Path(root)/'run.lock'
    with path.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid()},stream)
    try:yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('pid')==os.getpid():path.unlink()


def artifact_path(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or path==root:raise ValueError('产物路径越界')
    return path


def verify_unit(directory):
    manifest=directory/'completed.json'
    if not manifest.exists():return None
    record=json.loads(manifest.read_text(encoding='utf-8'))
    if not record.get('files') or record.get('payload_path') not in record['files']:raise ValueError('完成清单缺产物')
    for relative,digest in record['files'].items():
        path=artifact_path(directory,relative)
        if not path.is_file() or file_digest(path)!=digest:raise ValueError(f'已完成单元产物漂移：{path}')
    payload_path=artifact_path(directory,record['payload_path'])
    if record['payload_path'] not in record['files']:raise ValueError('未校验的payload路径')
    return torch.load(payload_path,map_location='cpu',weights_only=False),payload_path.parent


class ExplorationStore:
    def __init__(self,root,verify_identity,budget):
        self.root=Path(root);self.verify_identity=verify_identity;self.budget=budget
        self.units=self.root/'units';self.units.mkdir(exist_ok=True)

    def execute(self,key,operation):
        if not key or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in key):
            raise ValueError('非法单元名')
        self.verify_identity();self.budget.check()
        directory=self.units/key;directory.mkdir(exist_ok=True)
        restored=verify_unit(directory)
        if restored is not None:
            print(f'SKIP VERIFIED {key}',flush=True);return restored
        attempt=directory/f'attempt-{uuid.uuid4().hex[:12]}';attempt.mkdir(exist_ok=False)
        started=time.perf_counter();print(f'BEGIN {key} {attempt.name}',flush=True)
        try:
            payload=operation(attempt)
            torch.save(payload,attempt/'payload.pt')
            self.budget.check();self.verify_identity()
            files={str(p.relative_to(directory)).replace('\\','/'):file_digest(p)
                   for p in sorted(attempt.rglob('*')) if p.is_file()}
            relative=str((attempt/'payload.pt').relative_to(directory)).replace('\\','/')
            # payload.pt的存在使用相对attempt路径验证，避免混淆同单元旧失败尝试。
            atomic_write_json(directory/'completed.json',{'status':'completed','payload_path':relative,
                                                         'files':files,'elapsed_seconds':time.perf_counter()-started})
            self.budget.save();print(f'END {key} seconds={time.perf_counter()-started:.2f}',flush=True)
            return payload,attempt
        except BaseException as exc:
            atomic_write_json(attempt/'error.json',{'type':type(exc).__name__,'error':str(exc)})
            self.budget.save();raise
