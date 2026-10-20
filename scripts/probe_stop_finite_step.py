"""固定LT13方向的一次有限更新；不训练、不导出部署模型。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import DatasetGenerator
from experiments.stop_cross_effects import first_stop_preservation_bound
from experiments.stop_finite_step import quarter_bound_head, cached_float32_logits, finite_loss_changes
from experiments.terminal_alignment_audit import analyze_terminal_alignment
from interfaces import BEVTensor, GoalPose, VehicleState
from scripts.audit_stop_objectives import PRIOR, digest, load_verified_inputs
from training.checkpoint import load_model_checkpoint
from training.data import prepare_batches

EXTENDED = ROOT / 'runs/diagnostics/stop-fullspace-extended-20261020-v1/results/report.json'
EXTENDED_SHA = 'f08e3595b39fe2359a92eba2e03066e41e8a668287e55c00e2039ea97d26ad35'
NATIVE_INDICES = [105, 791, 1229, 1298, 1455, 1559, 2050, 2348]


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('拒绝覆盖有限更新验证目录')
    started = time.perf_counter(); torch.set_num_threads(1)
    if digest(EXTENDED) != EXTENDED_SHA:
        raise ValueError('不是冻结LT13结果')
    extended = json.loads(EXTENDED.read_text(encoding='utf-8'))
    c = json.loads((PRIOR / 'config.json').read_text())
    for name, expected in c['code'].items():
        if digest(name) != expected:
            raise ValueError('缓存生成执行源漂移')
    cache, weight, bias, parent_rows, threshold, inputs = load_verified_inputs()
    direction = torch.tensor(extended['scenarios']['guarded_metrics']['witness']['direction'], dtype=torch.double)
    kept = [i for i, row in enumerate(parent_rows) if row['deployment_endpoint']['joint_valid']]
    x = cache['features'][kept].double()
    base = (x @ weight.double().T).squeeze(-1) + bias.double()
    velocity = x @ direction[:-1] + direction[-1]
    stops = torch.tensor([parent_rows[i]['stop_length'] - 1 for i in kept])
    bound = first_stop_preservation_bound(base, velocity, stops, threshold=threshold)
    parent = load_model_checkpoint(c['inputs']['model']['path']).model
    candidate = deepcopy(parent)
    candidate.decoder.stop_head, eta = quarter_bound_head(parent.decoder.stop_head, direction,
                                                         bound['positive_step_supremum'])
    changed = [name for name, tensor in candidate.state_dict().items()
               if not torch.equal(tensor, parent.state_dict()[name])]
    if set(changed) != {'decoder.stop_head.weight', 'decoder.stop_head.bias'}:
        raise ValueError('变化范围不是两个head键')
    parent_logits = cached_float32_logits(parent.decoder.stop_head, cache['features'])
    if not torch.equal(parent_logits, cache['logits']):
        raise ValueError('逐时刻父缓存logits未逐元素复现')
    candidate_logits = cached_float32_logits(candidate.decoder.stop_head, cache['features'])
    rows, summary = analyze_terminal_alignment(
        *(cache[key][:2400].numpy() for key in ('points',)), candidate_logits[:2400].numpy(),
        *(cache[key][:2400].numpy() for key in ('targets', 'masks', 'goals')), parent_rows, threshold=threshold)
    groups = {'full_2918': list(range(2918)),
              'original_gt_valid': [i for i, row in enumerate(parent_rows) if row['teacher_index_prediction']['joint_valid']],
              'S4': [i for i, row in enumerate(parent_rows) if row['scene_name'].startswith('S4_')],
              'S8': [i for i, row in enumerate(parent_rows) if row['scene_name'].startswith('S8_')]}
    losses = finite_loss_changes(parent_logits, candidate_logits, cache['lengths'], groups, threshold=threshold)
    source_item = c['inputs']['train']
    if digest(source_item['path']) != source_item['sha256']:
        raise ValueError('原始抽查数据漂移')
    source = DatasetGenerator.load(source_item['path'])
    subset = {key: np.asarray(source[key])[NATIVE_INDICES] for key in ('bevs', 'goals', 'states', 'trajs', 'masks')}
    subset['dt'] = source['dt']
    batches = prepare_batches(subset, horizon=320, batch_size=1)
    native = []
    for index, batch in zip(NATIVE_INDICES, batches):
        with torch.no_grad():
            before = parent.forward_with_stop(*batch[:3]); after = candidate.forward_with_stop(*batch[:3])
        if not torch.equal(before.points, after.points):
            raise ValueError('原生完整预测点改变')
        meta = source['bev_meta']
        bev = BEVTensor(batch[0][0].numpy(), meta['resolution'], tuple(meta['extent']), meta['channels'])
        goal = GoalPose(*batch[1][0].tolist()); state = VehicleState(0., 0., 0., *batch[2][0].tolist())
        parent_prediction = parent.predict(bev, goal, state)
        candidate_prediction = candidate.predict(bev, goal, state)
        native.append({'dataset_index': index, 'parent_length': len(parent_prediction.points),
                       'candidate_length': len(candidate_prediction.points), 'full_points_equal': True,
                       'predict_points_equal': bool(np.array_equal(parent_prediction.points, candidate_prediction.points)),
                       'parent_cache_length': parent_rows[index]['stop_length'],
                       'candidate_cache_length': rows[index]['stop_length']})
    retained = sum(rows[i]['deployment_endpoint']['joint_valid'] for i in kept)
    gained = sum(row['deployment_endpoint']['joint_valid'] and not parent_rows[i]['deployment_endpoint']['joint_valid']
                 for i, row in enumerate(rows))
    result = {'status': 'completed_requires_review', 'lt13_report_sha256': EXTENDED_SHA,
              'inputs': inputs, 'source_data': source_item,
              'code_sha256': {str(path): digest(path) for path in
                              (Path(__file__).resolve(), ROOT / 'experiments/stop_finite_step.py',
                               ROOT / 'experiments/stop_cross_effects.py', ROOT / 'experiments/stop_objective_audit.py',
                               ROOT / 'experiments/terminal_alignment_audit.py')},
              'step_fraction': .25, 'eta': eta, 'analytic_bound': bound, 'threshold': threshold,
              'changed_keys': changed, 'parent_logits_bitwise_equal': True,
              'parent_valid': len(kept), 'retained_parent_valid': retained, 'lost_parent_valid': len(kept)-retained,
              'gained_valid': gained, 'changed_cached_stop_lengths': sum(r['stop_length'] != p['stop_length'] for r, p in zip(rows, parent_rows)),
              'summary': summary, 'finite_losses': losses, 'native_spot_checks': native,
              'full_2400_native_B1_evaluated': False, 'training_performed': False,
              'model_exported': False, 'automatic_expansion': False, 'closed_loop_evaluated': False,
              'elapsed_seconds': time.perf_counter()-started}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'head-probe.npz', parent_weight=weight.numpy(), parent_bias=bias.numpy(),
                        candidate_weight=candidate.decoder.stop_head.weight.detach().numpy(),
                        candidate_bias=candidate.decoder.stop_head.bias.detach().numpy(), direction=direction.numpy(),
                        parent_logits=parent_logits.numpy(), candidate_logits=candidate_logits.numpy())
    (output / 'samples.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    result['evidence_sha256'] = {name: digest(output / name) for name in ('head-probe.npz', 'samples.json')}
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('eta', 'elapsed_seconds', 'retained_parent_valid', 'lost_parent_valid',
                      'gained_valid', 'changed_cached_stop_lengths', 'finite_losses', 'native_spot_checks')}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
