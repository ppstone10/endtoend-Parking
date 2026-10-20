"""一次性只读停止目标诊断，不训练、不写模型。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stop_objective_audit import audit_head_gradients
from experiments.terminal_alignment_audit import analyze_terminal_alignment

PRIOR = ROOT / 'runs/training/frozen-stop-head-ab-20261020-v1'
CONFIG_SHA = '3dd49985a2d81ac95fe2b88c522db03122f1d01a0f0a2d70186a49f7db252002'
REPORT_SHA = '0e0a91c75fcdf69e412601263b2b2e4a7a82b401e233073aa215eb6751138589'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_verified_inputs():
    if digest(PRIOR / 'config.json') != CONFIG_SHA or digest(PRIOR / 'results/report.json') != REPORT_SHA:
        raise ValueError('LT12配置/报告身份漂移')
    config = json.loads((PRIOR / 'config.json').read_text())
    report = json.loads((PRIOR / 'results/report.json').read_text())
    paths = {'cache': PRIOR / 'results/cache-free.pt',
             'model': Path(config['inputs']['model']['path']),
             'samples': PRIOR / 'results/parent/samples.json'}
    expected = {'cache': report['cache_sha256']['free'], 'model': config['inputs']['model']['sha256'],
                'samples': report['parent']['samples_sha256']}
    if any(digest(path) != expected[key] for key, path in paths.items()):
        raise ValueError('缓存/父模型/样本身份漂移')
    cache = torch.load(paths['cache'], map_location='cpu', weights_only=False)
    state = torch.load(paths['model'], map_location='cpu', weights_only=False)['model_state']
    rows = json.loads(paths['samples'].read_text())
    n, horizon, hidden = cache['features'].shape
    expected_shapes = {'logits': (n, horizon), 'points': (n, horizon, 3), 'targets': (n, horizon, 3),
                       'masks': (n, horizon), 'lengths': (n,), 'goals': (n, 3)}
    if ((n, horizon, hidden) != (2918, 320, 128) or len(rows) != 2400
            or any(cache[key].shape != shape for key, shape in expected_shapes.items())
            or not all(torch.isfinite(value).all() for value in cache.values())):
        raise ValueError('缓存形状/有限值错误')
    mask = cache['masks']
    if (not ((mask == 0) | (mask == 1)).all() or (mask[:, 1:] > mask[:, :-1]).any()
            or not torch.equal(mask.sum(1), cache['lengths']) or (cache['lengths'] < 1).any()):
        raise ValueError('缓存长度/mask错误')
    weight, bias = state['decoder.stop_head.weight'], state['decoder.stop_head.bias']
    for begin in range(0, n, 64):
        computed = (cache['features'][begin:begin + 64] @ weight.T).squeeze(-1) + bias
        if not torch.allclose(computed, cache['logits'][begin:begin + 64], atol=2e-6, rtol=0):
            raise ValueError('父head/logit不一致')
    fresh, summary = analyze_terminal_alignment(
        *(cache[key][:2400].numpy() for key in ('points', 'logits', 'targets', 'masks', 'goals')),
        rows, threshold=config['threshold'])
    if fresh != rows or summary['categories'] != report['parent']['summary']['categories']:
        raise ValueError('父终点审计未复现')
    return cache, weight, bias, rows, config['threshold'], expected


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('拒绝覆盖审计目录')
    started = time.perf_counter()
    torch.set_num_threads(1)
    cache, weight, bias, rows, threshold, inputs = load_verified_inputs()
    groups = Counter()
    for row in rows:
        groups['gt_index_valid' if row['teacher_index_prediction']['joint_valid'] else 'gt_index_invalid'] += 1
        groups['parent_deployment_valid' if row['deployment_endpoint']['joint_valid'] else 'parent_deployment_invalid'] += 1
        if row['category'] == 'no_joint_point':
            groups['no_joint_point'] += 1
    scenes = {}
    for scene in sorted({row['scene_name'] for row in rows}):
        selected = [row for row in rows if row['scene_name'] == scene]
        scenes[scene] = {'samples': len(selected),
                         'gt_index_valid': sum(row['teacher_index_prediction']['joint_valid'] for row in selected),
                         'parent_deployment_valid': sum(row['deployment_endpoint']['joint_valid'] for row in selected),
                         'no_joint_point': sum(row['category'] == 'no_joint_point' for row in selected)}
    result = {'status': 'completed_requires_design_review', 'config_sha256': CONFIG_SHA,
              'prior_report_sha256': REPORT_SHA, 'input_sha256': inputs,
              'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
                              (Path(__file__).resolve(), ROOT / 'experiments/stop_objective_audit.py',
                               ROOT / 'experiments/terminal_alignment_audit.py')},
              'runtime': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                          'threads': 1, 'gradient_dtype': 'float64'},
              'threshold': threshold, 'groups_original_2400': dict(groups), 'by_scene': scenes,
              'objectives': {}, 'training_performed': False, 'model_exported': False,
              'automatic_training': False, 'closed_loop_evaluated': False}
    for count in (2400, 2918):
        result['objectives'][str(count)] = audit_head_gradients(
            cache['features'][:count], cache['lengths'][:count], weight, bias, threshold=threshold)
    # 分层只用于诊断；不删除、重标或改变任何训练采样。
    partitions = {
        'original_gt_valid': [i for i, row in enumerate(rows) if row['teacher_index_prediction']['joint_valid']],
        'original_gt_invalid': [i for i, row in enumerate(rows) if not row['teacher_index_prediction']['joint_valid']],
        'original_parent_deployment_valid': [i for i, row in enumerate(rows) if row['deployment_endpoint']['joint_valid']],
        'existing_recovery_518': list(range(2400, 2918)),
    }
    result['partition_gradients'] = {}
    for name, indices in partitions.items():
        result['partition_gradients'][name] = {
            'samples': len(indices),
            'objectives': audit_head_gradients(cache['features'][indices], cache['lengths'][indices],
                                               weight, bias, threshold=threshold),
        }
    result['elapsed_seconds'] = time.perf_counter() - started
    # 所有计算通过后才创建新目录；输入永远只读，不产生可部署模型。
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
