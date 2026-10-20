"""准备场景覆盖双臂实验；只写数据/配置，不启动训练。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import DatasetGenerator
from scripts.build_selection_dataset import collect_metadata_ids
from training.checkpoint import load_model_checkpoint
from training.config import load_training_run_config
from training.reporting import atomic_write_json


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata_only(path):
    with np.load(path, allow_pickle=False) as archive:
        return [json.loads(str(row)) for row in archive['task_meta']]


def execution_code_identity():
    paths = [*ROOT.joinpath('model').glob('*.py'), *ROOT.joinpath('training').glob('*.py')]
    paths += [Path(__file__), ROOT / 'scripts/train_fixed_threshold.py']
    yaw_preparation = ROOT / 'scripts/prepare_yaw_loss_ab.py'
    if yaw_preparation.exists():
        paths.append(yaw_preparation)
    return {str(path.resolve()): file_digest(path) for path in sorted(paths)}


def runtime_identity():
    return {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'torch': str(torch.__version__)}


def select_recovery_indices(metadata, *, seed):
    original, reverse, forward = [], [], []
    seen = set()
    for index, meta in enumerate(metadata):
        recovery = meta.get('recovery')
        if not recovery:
            original.append(index)
            continue
        source = recovery.get('source_task_id')
        if not source or 'rollout_step' not in recovery:
            raise ValueError('恢复样本缺少来源身份/步数')
        key = (source, int(recovery['rollout_step']))
        if key in seen:
            raise ValueError('重复恢复来源状态')
        seen.add(key)
        maneuver = meta['difficulty']['maneuver']
        if maneuver not in ('forward', 'reverse'):
            raise ValueError('恢复样本机动必须明确')
        (forward if maneuver == 'forward' else reverse).append(index)
    if not reverse or len(forward) < len(reverse):
        raise ValueError('本实验要求前进池不小于非空倒车池')
    protected = [i for i in forward if metadata[i]['recovery']['trigger'] == 'collision_backtrack']
    budget = len(reverse) - len(protected)
    if budget < 0:
        raise ValueError('前进碰撞回溯保护量超过预算')
    protected_set = set(protected)
    pools = defaultdict(list)
    for index in forward:
        if index not in protected_set:
            meta = metadata[index]
            pools[(meta['scene_name'], meta['recovery']['trigger'])].append(index)
    total = sum(map(len, pools.values()))
    if budget and not total:
        raise ValueError('没有可分配的前进恢复样本')
    quotas = {key: int(np.floor(budget * len(pool) / total)) for key, pool in pools.items()}
    remainder = budget - sum(quotas.values())
    ranked = sorted(pools, key=lambda key: (-(budget * len(pools[key]) / total - quotas[key]), key))
    for key in ranked[:remainder]:
        quotas[key] += 1
    rng = np.random.default_rng(seed)
    selected = original + reverse + protected
    for key in sorted(pools):
        selected.extend(rng.permutation(pools[key])[:quotas[key]].tolist())
    return sorted(selected)


def assert_original_equivalence(source, control):
    left = [i for i, meta in enumerate(source['task_meta']) if not meta.get('recovery')]
    right = [i for i, meta in enumerate(control['task_meta']) if not meta.get('recovery')]
    if len(left) != len(right) or any(source['task_meta'][i] != control['task_meta'][j] for i, j in zip(left, right)):
        raise ValueError('原始专家身份/元数据不一致')
    for field in ('bevs', 'goals', 'states'):
        if not np.array_equal(source[field][left], control[field][right]):
            raise ValueError(f'原始专家输入改变：{field}')
    for i, j in zip(left, right):
        count = int(source['masks'][i].sum())
        if count != int(control['masks'][j].sum()) or not np.array_equal(source['trajs'][i, :count], control['trajs'][j, :count]):
            raise ValueError('原始专家有效标签改变')


def write_subset(data, indices, path):
    temporary = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(
        temporary, schema_version=np.asarray(2, dtype=np.uint16),
        bev_meta=np.asarray(DatasetGenerator._encode_metadata(data['bev_meta'], 'bev_meta'), dtype=np.str_),
        task_meta=np.asarray([DatasetGenerator._encode_metadata(data['task_meta'][i], 'task_meta') for i in indices], dtype=np.str_),
        **{key: np.asarray(data[key])[indices] for key in ('bevs', 'goals', 'states', 'trajs', 'masks')},
        dt=np.asarray(data['dt']),
    )
    temporary.replace(path)


def prepare_experiment(output: Path):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('任务目录已存在，请使用新目录；不得覆盖或重新prepare已有训练')
    inputs = {
        'source': ROOT / 'data/task_dataset/v40_dagger_collision/train_with_recovery.npz',
        'control': ROOT / 'data/task_dataset/v40_dagger_collision/train-balanced.npz',
        'parent': ROOT / 'runs/validation/arch-ab/dep-epoch0007.pt',
        'validation': ROOT / 'data/task_dataset/tracked_pivot_v8_3000/val.npz',
        'selection': ROOT / 'data/task_dataset/independent_selection_v1/selection.npz',
        'base_training_config': ROOT / 'configs/training/net-v1-v40-lineage.yaml',
    }
    source, control = (DatasetGenerator.load(inputs[key]) for key in ('source', 'control'))
    if (len(source['task_meta']), len(control['task_meta'])) != (3298, 2918):
        raise ValueError('不是已诊断的固定v40数据')
    assert_original_equivalence(source, control)
    selected = select_recovery_indices(source['task_meta'], seed=20261018)
    chosen_meta = [source['task_meta'][i] for i in selected]
    counts = Counter('original' if not m.get('recovery') else m['difficulty']['maneuver'] for m in chosen_meta)
    backtracks = sum(m.get('recovery', {}).get('trigger') == 'collision_backtrack' for m in chosen_meta)
    if dict(counts) != {'original': 2400, 'forward': 259, 'reverse': 259} or backtracks != 98:
        raise ValueError('固定样本/机动/回溯配额不符')
    overlaps = {}
    training_ids = collect_metadata_ids(source['task_meta']) | collect_metadata_ids(control['task_meta'])
    for key in ('validation', 'selection'):
        overlaps[key] = len(training_ids & collect_metadata_ids(metadata_only(inputs[key])))
    if any(overlaps.values()):
        raise ValueError('训练来源与评测任务身份重叠')
    loaded = load_model_checkpoint(inputs['parent'])
    if loaded.epoch != 6 or not np.isclose(loaded.model.stop_threshold, .26) or file_digest(inputs['parent']).lower() != 'd7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8':
        raise ValueError('父模型不是已冻结v40 epoch7校准模型')
    base = load_training_run_config(inputs['base_training_config'])
    code = execution_code_identity()
    output.mkdir(parents=True)
    candidate = output / 'candidate-train.npz'
    write_subset(source, selected, candidate)
    files = {key: {'path': str(path), 'sha256': file_digest(path)} for key, path in inputs.items()}
    files['candidate'] = {'path': str(candidate), 'sha256': file_digest(candidate)}
    arms = {}
    for arm, train_path in (('R1', inputs['control']), ('R2', candidate)):
        arm_root = output / arm
        settings = replace(base.trainer, epochs=4, learning_rate=5e-5, device='cpu',
                           checkpoint_dir=str(arm_root / 'checkpoints'), patience=4,
                           early_stopping_start_epoch=3, teacher_forcing_start=.44,
                           teacher_forcing_end=.2, teacher_forcing_decay_epochs=3)
        config = {'schema_version': 1, 'kind': 'recovery_balance_ab', 'arm': arm,
                  'manifest': str(output / 'manifest.json'), 'output': str(arm_root),
                  'train': str(train_path), 'validation': str(inputs['validation']),
                  'parent': str(inputs['parent']), 'batch_size': 8, 'threads': 1,
                  'model_name': loaded.model_name, 'model_config': loaded.model_config,
                  'trainer': asdict(settings)}
        path = atomic_write_json(arm_root / 'config.json', config)
        arms[arm] = {'config': str(path), 'sha256': file_digest(path)}
    by_scene = {}
    for scene in sorted({m['scene_name'] for m in chosen_meta}):
        rows = [m for m in chosen_meta if m['scene_name'] == scene]
        by_scene[scene] = dict(Counter('original' if not m.get('recovery') else m['difficulty']['maneuver'] for m in rows))
    git = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    manifest = {'schema_version': 1, 'status': 'ready_for_user_training', 'kind': 'recovery_balance_ab',
                'inputs': files, 'code': code, 'arms': arms, 'git_head': git.stdout.strip(),
                'runtime': runtime_identity(),
                'selected_source_indices': selected, 'sample_counts': dict(counts),
                'backtracks': backtracks, 'task_identity_overlaps': overlaps, 'by_scene': by_scene,
                'original_valid_trajectories_verified_equal': 2400,
                'formal_training_started': False, 'deployment_modified': False}
    atomic_write_json(output / 'manifest.json', manifest)
    print(json.dumps({key: manifest[key] for key in ('status', 'sample_counts', 'backtracks', 'by_scene')}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare_experiment(args.output)


if __name__ == '__main__':
    main()
