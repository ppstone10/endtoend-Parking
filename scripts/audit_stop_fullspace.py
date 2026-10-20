"""完整129维停止头可行性审计；长预算须先prepare并由用户运行。"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stop_fullspace_feasibility import fullspace_parent_constraints, protected_common_descent
from experiments.stop_objective_audit import audit_head_gradients
from scripts.audit_stop_objectives import CONFIG_SHA, REPORT_SHA, PRIOR, digest, load_verified_inputs

SOURCES = (Path(__file__).resolve(), ROOT / 'experiments/stop_fullspace_feasibility.py',
           ROOT / 'experiments/stop_objective_audit.py', ROOT / 'experiments/stop_cross_effects.py',
           ROOT / 'scripts/audit_stop_objectives.py', ROOT / 'experiments/terminal_alignment_audit.py')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def identity():
    if digest(PRIOR / 'config.json') != CONFIG_SHA or digest(PRIOR / 'results/report.json') != REPORT_SHA:
        raise ValueError('LT12身份漂移')
    config = json.loads((PRIOR / 'config.json').read_text())
    prior = json.loads((PRIOR / 'results/report.json').read_text())
    inputs = {'config': (PRIOR / 'config.json', CONFIG_SHA), 'report': (PRIOR / 'results/report.json', REPORT_SHA),
              'model': (Path(config['inputs']['model']['path']), config['inputs']['model']['sha256']),
              'cache': (PRIOR / 'results/cache-free.pt', prior['cache_sha256']['free']),
              'samples': (PRIOR / 'results/parent/samples.json', prior['parent']['samples_sha256'])}
    for path, expected in inputs.values():
        if digest(path) != expected:
            raise ValueError('输入摘要漂移')
    return {'inputs': {key: {'path': str(path), 'sha256': expected} for key, (path, expected) in inputs.items()},
            'code_sha256': {str(path): digest(path) for path in SOURCES},
            'runtime': {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
                        'torch': torch.__version__}}


def prepare(directory):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError('拒绝覆盖任务根目录')
    config = {'kind': 'stop_fullspace_feasibility', 'identity': identity(),
              'budgets': {'max_iterations': 4096, 'max_seconds': 120.},
              'output': str(directory / 'results')}
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / 'config.json', config)
    print('prepared', directory / 'config.json', digest(directory / 'config.json'), flush=True)


def check_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if (config.get('kind') != 'stop_fullspace_feasibility' or config.get('identity') != identity()
            or config.get('budgets') != {'max_iterations': 4096, 'max_seconds': 120.}
            or Path(config.get('output', '')).resolve() != Path(path).resolve().parent / 'results'):
        raise ValueError('冻结配置/执行源/运行时漂移')
    return config


def run(output, *, max_iterations=256, max_seconds=8., config_sha256=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('拒绝覆盖全空间审计目录')
    started = time.perf_counter()
    torch.set_num_threads(1)
    frozen_identity = identity()
    cache, weight, bias, rows, threshold, inputs = load_verified_inputs()
    _, full = audit_head_gradients(cache['features'], cache['lengths'], weight, bias,
                                   threshold=threshold, return_gradients=True)
    groups = {
        'full_2918': list(range(2918)),
        'original_gt_valid': [i for i, row in enumerate(rows) if row['teacher_index_prediction']['joint_valid']],
        'S4': [i for i, row in enumerate(rows) if row['scene_name'].startswith('S4_')],
        'S8': [i for i, row in enumerate(rows) if row['scene_name'].startswith('S8_')],
    }
    gradients = []; labels = []
    for name, indices in groups.items():
        if name == 'full_2918':
            values = full
        else:
            _, values = audit_head_gradients(cache['features'][indices], cache['lengths'][indices], weight, bias,
                                             threshold=threshold, return_gradients=True)
        for term in ('threshold_prefix', 'threshold_event'):
            gradient = values[term]
            gradients.append(gradient / gradient.norm().clamp_min(1e-30))
            labels.append(name + ':' + term)
    g = torch.stack(gradients)
    retained = [i for i, row in enumerate(rows) if row['deployment_endpoint']['joint_valid']]
    if len(retained) != 1144 or any(not rows[i]['stop_found'] for i in retained):
        raise ValueError('原合格组分母/事件漂移')
    x = cache['features'][retained]
    logits = (x.double() @ weight.double().T).squeeze(-1) + bias.double()
    stops = torch.tensor([rows[i]['stop_length'] - 1 for i in retained])
    points, goals = cache['points'][retained], cache['goals'][retained]
    delta = points[:, :, 2] - goals[:, None, 2]
    joint = ((torch.linalg.vector_norm(points[:, :, :2] - goals[:, None, :2], dim=2) <= .3)
             & (torch.atan2(torch.sin(delta), torch.cos(delta)).abs() <= math.pi / 18))
    strict, compatible = fullspace_parent_constraints(x, logits, stops, joint, threshold=threshold)
    empty = torch.empty((0, 129), dtype=torch.double)
    scenarios = {'aggregate_only': (g[:2], empty), 'guarded_metrics': (g, empty),
                 'strict_index_protection': (g, strict), 'goal_compatible_anchor_protection': (g, compatible)}
    result = {'status': 'running', 'identity': frozen_identity, 'config_sha256': config_sha256,
              'budgets': {'max_iterations': max_iterations, 'max_seconds': max_seconds},
              'threshold': threshold, 'group_samples': {k: len(v) for k, v in groups.items()},
              'metric_labels': labels, 'normalized_metric_gradients': g.tolist(),
              'scenarios': {}, 'training_performed': False, 'model_exported': False,
              'automatic_training': False, 'closed_loop_evaluated': False}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'status.json', {'status': 'running'})
    try:
        for name, (metrics, protection) in scenarios.items():
            print('start', name, len(protection), flush=True)
            solved = protected_common_descent(
                metrics, protection, max_iterations=max_iterations, max_seconds=max_seconds,
                progress=lambda step, bound, violation: print(name, 'iteration', step, 'upper_bound', bound,
                                                               'protection_violation', violation, flush=True))
            solved['metric_rows'] = len(metrics)
            solved['protection_rows'] = len(protection)
            solved['dual']['active_protection_rows'] = protection[solved['dual']['active_protection_indices']].tolist()
            result['scenarios'][name] = solved
            save_json(output / 'report.partial.json', result)
            print('done', name, solved['status'], flush=True)
        if identity() != frozen_identity:
            raise ValueError('执行中输入/源身份漂移')
        result['status'] = 'completed_requires_design_review'
        result['elapsed_seconds'] = time.perf_counter() - started
        save_json(output / 'report.json', result)
        save_json(output / 'status.json', {'status': 'completed', 'report_sha256': digest(output / 'report.json')})
    except Exception as error:
        save_json(output / 'status.json', {'status': 'failed', 'error': repr(error)})
        raise
    print('completed', output / 'report.json', result['elapsed_seconds'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument('--output', type=Path, help='固定短预算独立目录')
    entry.add_argument('--prepare', type=Path, help='冻结供用户运行的长预算任务')
    entry.add_argument('--config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only and args.config is None:
        parser.error('--check-only需要--config')
    if args.prepare:
        prepare(args.prepare)
    elif args.config:
        config = check_config(args.config)
        if args.check_only:
            print('check-only passed', flush=True)
        else:
            run(config['output'], **config['budgets'], config_sha256=digest(args.config))
    else:
        run(args.output)
