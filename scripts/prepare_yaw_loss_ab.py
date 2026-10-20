"""复用已核验候选数据准备 raw/wrapped 双臂；不启动训练。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_recovery_balance_ab import execution_code_identity, file_digest, runtime_identity
from training.reporting import atomic_write_json
from training.trainer import TrainerConfig

SOURCE = ROOT / 'runs/training/recovery-balance-ab-20261018-v2'
SOURCE_MANIFEST_SHA = '0da9473018ab22ce23e29086c5ef50be661dc5c6dac63f46d2cf2e0454eb700b'


def paired_configs(source, output):
    """两个实验只能在目录、臂名和航向误差模式上不同。"""
    configs = {}
    for arm, mode in (('W0', 'raw'), ('W1', 'wrapped')):
        config = deepcopy(source)
        config.update(kind='yaw_loss_ab', arm=arm, manifest=str(output / 'manifest.json'), output=str(output / arm))
        config['trainer'].update(yaw_error_mode=mode, checkpoint_dir=str(output / arm / 'checkpoints'))
        TrainerConfig(**config['trainer'])
        configs[arm] = config
    return configs


def prepare_experiment(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('新实验必须使用不存在的目录；禁止覆盖或自动重跑')
    source_manifest_path = SOURCE / 'manifest.json'
    if file_digest(source_manifest_path) != SOURCE_MANIFEST_SHA:
        raise ValueError('旧实验 manifest 身份改变')
    old = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    source_config_path = SOURCE / 'R2/config.json'
    if old['arms']['R2']['sha256'] != file_digest(source_config_path):
        raise ValueError('候选数据训练配置身份改变')
    source = json.loads(source_config_path.read_text(encoding='utf-8'))
    inputs = deepcopy(old['inputs'])
    inputs.update(source_manifest={'path': str(source_manifest_path), 'sha256': SOURCE_MANIFEST_SHA},
                  source_config={'path': str(source_config_path), 'sha256': file_digest(source_config_path)})
    for entry in inputs.values():
        if file_digest(entry['path']) != entry['sha256']:
            raise ValueError(f'冻结输入身份改变：{entry["path"]}')
    if source['train'] != inputs['candidate']['path'] or source['parent'] != inputs['parent']['path'] or source['validation'] != inputs['validation']['path']:
        raise ValueError('训练输入未指向冻结候选/原父模型/原验证集')
    configs = paired_configs(source, output)
    output.mkdir(parents=True, exist_ok=False)
    arms = {}
    for arm, config in configs.items():
        path = atomic_write_json(output / arm / 'config.json', config)
        arms[arm] = {'config': str(path), 'sha256': file_digest(path)}
    manifest = {'schema_version': 1, 'kind': 'yaw_loss_ab', 'status': 'ready_for_user_training',
                'inputs': inputs, 'arms': arms, 'code': execution_code_identity(), 'runtime': runtime_identity(),
                'sample_counts': old['sample_counts'], 'task_identity_overlaps': old['task_identity_overlaps'],
                'primary_variable': 'trainer.yaw_error_mode: raw versus wrapped',
                'formal_training_started': False, 'deployment_modified': False,
                'resume_policy': 'same manifest/config/arm only; new optimizer initialized from frozen parent weights'}
    atomic_write_json(output / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'manifest_sha256': file_digest(output / 'manifest.json')}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    prepare_experiment(parser.parse_args().output)


if __name__ == '__main__':
    main()
