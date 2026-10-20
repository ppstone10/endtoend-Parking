"""人工运行的固定阈值实验入口；不校准、不部署、不自动启动闭环评测。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dataset import DatasetGenerator
from scripts.prepare_recovery_balance_ab import file_digest, execution_code_identity, runtime_identity
from training.checkpoint import load_model_checkpoint
from training.data import prepare_batches, recovery_sample_groups, validate_model_dataset, epoch_batches
from training.reporting import atomic_write_json
from training.trainer import Trainer, TrainerConfig


def load_frozen_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    manifest_path = Path(config['manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'ready_for_user_training' or config.get('kind') not in {'recovery_balance_ab', 'yaw_loss_ab'} or manifest.get('kind', 'recovery_balance_ab') != config.get('kind'):
        raise ValueError('不是准备完成的双臂实验')
    expected = manifest['arms'][config['arm']]
    if Path(expected['config']) != path or expected['sha256'] != file_digest(path):
        raise ValueError('配置身份改变，禁止运行/续跑')
    if manifest['code'] != execution_code_identity():
        raise ValueError('执行代码已改变，禁止使用旧计划续跑')
    if manifest['runtime'] != runtime_identity():
        raise ValueError('Python/依赖运行环境已改变，禁止混用实验')
    for identity in manifest['inputs'].values():
        if file_digest(identity['path']) != identity['sha256']:
            raise ValueError(f'输入文件身份改变：{identity["path"]}')
    return config, manifest, file_digest(manifest_path)


@contextmanager
def exclusive_run(output):
    path = Path(output) / 'run.lock'
    # x模式避免两个进程同时从同一last恢复并相互覆盖产物。
    stream = path.open('x', encoding='utf-8')
    try:
        stream.write(str(os.getpid()))
        stream.close()
        yield
    finally:
        stream.close()
        path.unlink()


def check_run_state(output, *, resume, identity):
    output = Path(output)
    receipt = output / 'run-identity.json'
    if resume:
        if not receipt.is_file() or json.loads(receipt.read_text(encoding='utf-8')) != identity:
            raise ValueError('恢复身份不匹配')
        if not (output / 'checkpoints/last.pt').is_file():
            raise ValueError('没有已完成epoch检查点；请反馈异常，不要伪造resume')
    elif receipt.exists() or (output / 'checkpoints').exists():
        raise FileExistsError('已有正式运行产物；仅能显式resume或使用新任务目录')
    if (output / 'report.json').exists():
        raise FileExistsError('该训练已完成，不可重复执行或续跑')


class ProgressTrainer(Trainer):
    """只增加进度与非有限数检查，不改变基类训练计算。"""

    logger: logging.Logger

    def _batch_loss(self, batch, *, training, epoch, batch_index):
        result = super()._batch_loss(batch, training=training, epoch=epoch, batch_index=batch_index)
        if not all(torch.isfinite(value).all() for value in result):
            raise FloatingPointError('出现非有限损失，停止并保留上轮last')
        if training and batch_index % 50 == 0:
            self.logger.info('epoch %d/4 train batch %d loss %.6f', epoch + 1, batch_index, float(result[0].detach()))
        return result

    def _run_epoch(self, batches, *, training, epoch):
        started = time.perf_counter()
        self.logger.info('epoch %d/4 %s BEGIN', epoch + 1, 'train' if training else 'val')
        result = super()._run_epoch(batches, training=training, epoch=epoch)
        self.logger.info('epoch %d/4 phase END elapsed %.1fs', epoch + 1, time.perf_counter() - started)
        return result

    def _rollout_metrics(self, batches):
        self.logger.info('free rollout BEGIN samples=%d', sum(len(batch[0]) for batch in batches))
        result = super()._rollout_metrics(batches)
        self.logger.info('free rollout END ADE=%.6f FDE=%.6f', result['ade_m'], result['fde_m'])
        return result


def make_training(config, *, progress=False):
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['trainer']['seed'])
    loaded = load_model_checkpoint(config['parent'])
    if loaded.model_config != config['model_config']:
        raise ValueError('父模型配置不一致')
    train = DatasetGenerator.load(Path(config['train']))
    val = DatasetGenerator.load(Path(config['validation']))
    for data in (train, val):
        validate_model_dataset(loaded.model, data)
    if train['bevs'].shape[1:] != val['bevs'].shape[1:]:
        raise ValueError('train/val BEV形状不同')
    batches = prepare_batches(train, horizon=loaded.model.max_horizon, batch_size=config['batch_size'],
                              sample_groups=recovery_sample_groups(train['task_meta']))
    validation = prepare_batches(val, horizon=loaded.model.max_horizon, batch_size=config['batch_size'])
    trainer_class = ProgressTrainer if progress else Trainer
    trainer = trainer_class(loaded.model, TrainerConfig(**config['trainer']),
                            model_name=loaded.model_name, model_config=loaded.model_config,
                            initialization={'policy': 'model_weights_only', 'checkpoint': config['parent'],
                                            'checkpoint_sha256': file_digest(config['parent']), 'source_epoch': loaded.epoch})
    return trainer, batches, validation


def run_preflight(config_path, *, smoke=False):
    started = time.perf_counter()
    config, manifest, identity = load_frozen_config(config_path)
    trainer, batches, validation = make_training(config)
    summary = {'status': 'preflight_passed', 'formal_training_started': False, 'arm': config['arm'],
               'manifest_sha256': identity, 'train_samples': sum(len(b[0]) for b in batches),
               'val_samples': sum(len(b[0]) for b in validation),
               'threshold': trainer.model.stop_threshold, 'yaw_error_mode': trainer.config.yaw_error_mode,
               'teacher_forcing': [trainer.config.teacher_forcing_ratio(i) for i in range(4)],
               'training_batches_per_epoch': math.ceil(manifest['sample_counts']['original'] / (config['batch_size'] - 2)),
               'free_rollout_batches_per_epoch': len(batches) + len(validation), 'smoke': smoke}
    if smoke:
        active = epoch_batches(batches, shuffle=True, seed=trainer.config.seed, epoch=0, balance_recovery=True)
        before = {key: value.detach().clone() for key, value in trainer.model.state_dict().items()}
        tick = time.perf_counter()
        trainer.model.train()
        loss, _ = trainer._batch_loss(active[0], training=True, epoch=0, batch_index=0)
        if not torch.isfinite(loss):
            raise FloatingPointError('冒烟损失非有限数')
        loss.backward()  # 不调用optimizer.step，不保存检查点。
        if not all(torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters() if parameter.grad is not None):
            raise FloatingPointError('冒烟梯度非有限数')
        if any(not torch.equal(before[key], value) for key, value in trainer.model.state_dict().items()):
            raise AssertionError('预检不得更新模型权重')
        summary.update(smoke_loss=float(loss.detach()), smoke_batch_sec=time.perf_counter() - tick, model_weights_unchanged=True)
    summary['elapsed_sec'] = time.perf_counter() - started
    atomic_write_json(Path(config['output']) / ('preflight-smoke.json' if smoke else 'preflight.json'), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_training(config_path, *, resume=False):
    config, manifest, manifest_digest = load_frozen_config(config_path)
    output = Path(config['output'])
    identity = {'manifest_sha256': manifest_digest, 'config_sha256': file_digest(config_path), 'arm': config['arm']}
    with exclusive_run(output):
        check_run_state(output, resume=resume, identity=identity)
        atomic_write_json(output / 'run-identity.json', identity)
        logger = logging.getLogger(str(output))
        logger.setLevel(logging.INFO)
        handlers = [logging.FileHandler(output / 'run.log', mode='a', encoding='utf-8'), logging.StreamHandler()]
        for handler in handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(handler)
        started = time.perf_counter()
        try:
            atomic_write_json(output / 'status.json', {'status': 'running', **identity})
            logger.info('START arm=%s resume=%s fixed_threshold=%s', config['arm'], resume, config['model_config']['stop_threshold'])
            trainer, batches, validation = make_training(config, progress=True)
            trainer.logger = logger
            trainer.initialization = {**(trainer.initialization or {}), 'experiment_identity': identity}
            last = output / 'checkpoints/last.pt'
            if resume:
                next_epoch, previous_history = trainer.load_checkpoint(last)
                if (trainer.initialization or {}).get('experiment_identity') != identity:
                    raise ValueError('last检查点不属于当前实验臂/配置')
                if not 1 <= next_epoch <= 4 or len(previous_history.train_loss) != next_epoch:
                    raise ValueError('last检查点轮数与history不一致')
                missing_snapshot = output / f'checkpoints/epoch{next_epoch:04d}.pt'
                if not missing_snapshot.exists():
                    trainer._save_checkpoint(missing_snapshot.name, next_epoch - 1, previous_history)

            def checkpoint(epoch, history):
                trainer._save_checkpoint(f'epoch{epoch + 1:04d}.pt', epoch, history)
                atomic_write_json(output / 'history.json', history.to_dict())
                atomic_write_json(output / 'status.json', {'status': 'running', 'completed_epochs': epoch + 1, **identity})
                logger.info('EPOCH %d/4 COMPLETE tf=%.2f val_loss=%.6f', epoch + 1, trainer.config.teacher_forcing_ratio(epoch), history.val_loss[-1])

            history = trainer.fit(batches, validation, resume_from=last if resume else None, on_epoch_end=checkpoint)
            snapshots = [output / f'checkpoints/epoch{i:04d}.pt' for i in range(1, 5)]
            if len(history.train_loss) != 4 or not all(path.is_file() for path in snapshots):
                raise RuntimeError('未完成固定4轮或缺少快照，不写completed')
            if trainer.model.stop_threshold != config['model_config']['stop_threshold']:
                raise AssertionError('停止阈值不得改变')
            atomic_write_json(output / 'history.json', history.to_dict())
            report = {'status': 'completed', **identity, 'config': config, 'epochs': 4,
                      'elapsed_sec_this_invocation': time.perf_counter() - started,
                      'calibration': 'disabled_fixed_parent_threshold', 'deployment_modified': False,
                      'automatic_evaluation': False, 'history': history.to_dict(),
                      'snapshots': [{'path': str(path), 'sha256': file_digest(path)} for path in snapshots]}
            atomic_write_json(output / 'report.json', report)
            atomic_write_json(output / 'status.json', {'status': 'completed', 'completed_epochs': 4, **identity})
            logger.info('TRAINING COMPLETED; WAIT FOR AGENT REVIEW; NO DEPLOYMENT OR EVALUATION STARTED')
            return report
        except BaseException as exc:
            atomic_write_json(output / 'status.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                                      'error': str(exc), **identity})
            logger.exception('训练中断/失败；不要自动重试，检查last后人工resume')
            raise
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.check_only or args.smoke:
        run_preflight(args.config, smoke=args.smoke)
    else:
        run_training(args.config, resume=args.resume)


if __name__ == '__main__':
    main()
