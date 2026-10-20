"""固定LT13方向的终点合格区间审计；不选择或施加模型步长。"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stop_cross_effects import first_stop_preservation_bound
from experiments.stop_endpoint_intervals import affine_cache_logits, first_stop_intervals, earliest_selected_interval, event_record
from scripts.audit_stop_objectives import digest, load_verified_inputs
from scripts.probe_stop_finite_step import EXTENDED, EXTENDED_SHA


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('拒绝覆盖终点区间审计目录')
    started = time.perf_counter(); torch.set_num_threads(1)
    if digest(EXTENDED) != EXTENDED_SHA:
        raise ValueError('LT13结果漂移')
    extended = json.loads(EXTENDED.read_text(encoding='utf-8'))
    cache, weight, bias, rows, threshold, inputs = load_verified_inputs()
    direction = torch.tensor(extended['scenarios']['guarded_metrics']['witness']['direction'], dtype=torch.double)
    if abs(float(direction.norm())-1.) > 1e-9:
        raise ValueError('方向不是单位见证')
    base, velocity = affine_cache_logits(cache['features'],weight,bias,direction,count=2400)
    cutoff = math.log(threshold/(1-threshold))
    crossing = base >= cutoff
    stops = torch.where(crossing.any(1), crossing.long().argmax(1), torch.full((2400,), 319))
    saved_stops = torch.tensor([row['stop_length']-1 for row in rows])
    if not torch.equal(stops, saved_stops):
        raise ValueError('float64起点未复现原2400首停索引')
    points = cache['points'][:2400].numpy(); goals = cache['goals'][:2400].numpy()
    delta = points[:, :, 2] - goals[:, None, 2]
    joint = torch.from_numpy((np.linalg.norm(points[:, :, :2]-goals[:, None, :2], axis=2) <= .3)
                             & (np.abs(np.arctan2(np.sin(delta), np.cos(delta))) <= math.pi/18))
    valid = joint[torch.arange(2400), stops]
    if int(valid.sum()) != 1144 or valid.tolist() != [row['deployment_endpoint']['joint_valid'] for row in rows]:
        raise ValueError('原终点分组漂移')
    intervals = first_stop_intervals(base, velocity, threshold=threshold)
    bad = earliest_selected_interval(intervals, ~joint)
    good = earliest_selected_interval(intervals, joint)
    index_bound = first_stop_preservation_bound(base[valid], velocity[valid], stops[valid], threshold=threshold)

    def describe(events, sample):
        value = event_record(intervals, events, sample)
        if value is None:
            return None
        value.update({'dataset_index': sample, 'task_id': rows[sample]['task_id'], 'scene_name': rows[sample]['scene_name']})
        end = value['end']
        eta = (value['start'] + (end-value['start'])/2) if end is not None else value['start']+max(1.,abs(value['start'])*.1)
        # 只检验此解析区间的一个内部代表；不更新任何模型或筛步长。
        representable = math.isfinite(eta) and (eta > value['start'] or value['start_closed'])
        if representable:
            selected = torch.nonzero(base[sample]+eta*velocity[sample] >= cutoff).flatten()
            actual = int(selected[0]) if len(selected) else 319
            value['interval_witness'] = {'eta_for_algebra_only': eta, 'selected_point': actual,
                                         'expected_point_matches': actual == value['point_index'],
                                         'endpoint_joint_valid': bool(joint[sample,actual])}
            if actual != value['point_index']:
                raise ValueError('解析事件内部点未复现首停')
        else:
            value['interval_witness'] = {'status': 'no_representable_interior'}
        return value

    loss_starts = torch.where(valid, bad['start'], torch.full_like(bad['start'], torch.inf))
    gain_starts = torch.where(~valid, good['start'], torch.full_like(good['start'], torch.inf))
    loss_index = int(loss_starts.argmin()); gain_index = int(gain_starts.argmin())
    first_loss = describe(bad, loss_index) if torch.isfinite(loss_starts[loss_index]) else None
    first_gain = describe(good, gain_index) if torch.isfinite(gain_starts[gain_index]) else None
    loss_bound = float(loss_starts.min())
    if index_bound['positive_step_supremum'] is not None and loss_bound + 1e-12 < index_bound['positive_step_supremum']:
        raise ValueError('终点合格界竟早于相同索引界')
    potential_gain = ~valid & torch.isfinite(gain_starts) & (gain_starts < loss_bound)
    samples = [{'dataset_index': i, 'task_id': rows[i]['task_id'], 'parent_valid': bool(valid[i]),
                'first_bad': event_record(intervals,bad,i), 'first_good': event_record(intervals,good,i),
                'possible_gain_before_first_parent_loss': bool(potential_gain[i])} for i in range(2400)]
    result = {
        'status': 'completed_diagnostic_only', 'lt13_report_sha256': EXTENDED_SHA, 'inputs': inputs,
        'code_sha256': {str(path): digest(path) for path in
                        (Path(__file__).resolve(), ROOT/'experiments/stop_endpoint_intervals.py',
                         ROOT/'experiments/stop_cross_effects.py', ROOT/'scripts/audit_stop_objectives.py',
                         ROOT/'scripts/probe_stop_finite_step.py')},
        'threshold': threshold, 'dtype': 'float64_affine_cache_only', 'samples': 2400,
        'parent_valid': int(valid.sum()), 'no_joint_point': int((~joint.any(1)).sum()),
        'first_index_change_bound': index_bound, 'first_parent_invalid_event': first_loss,
        'first_possible_gain_event': first_gain,
        'initial_parent_valid_interval_upper': loss_bound if math.isfinite(loss_bound) else None,
        'bound_ratio_over_same_index': loss_bound/index_bound['positive_step_supremum']
                                     if math.isfinite(loss_bound) and index_bound['positive_step_supremum'] else None,
        'possible_gain_samples_before_first_loss': int(potential_gain.sum()),
        'possible_gains_by_scene': {scene: int(potential_gain[torch.tensor([i for i,r in enumerate(rows) if r['scene_name']==scene])].sum())
                                    for scene in sorted({r['scene_name'] for r in rows})},
        'bottleneck_1229_first_bad': describe(bad,1229),
        'no_model_step_applied': True, 'parameter_step_selected': False, 'training_performed': False,
        'model_exported': False, 'float32_evaluated': False, 'closed_loop_evaluated': False,
        'elapsed_seconds': time.perf_counter()-started,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'intervals.npz', base=base.numpy(),velocity=velocity.numpy(),joint_valid=joint.numpy(),
                        parent_stop=stops.numpy(),parent_valid=valid.numpy(),**{k:v.numpy() for k,v in intervals.items()})
    (output/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result['evidence_sha256']={name:digest(output/name) for name in ('intervals.npz','samples.json')}
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('elapsed_seconds','first_parent_invalid_event','first_possible_gain_event',
                      'initial_parent_valid_interval_upper','bound_ratio_over_same_index',
                      'possible_gain_samples_before_first_loss','possible_gains_by_scene','bottleneck_1229_first_bad')},ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
