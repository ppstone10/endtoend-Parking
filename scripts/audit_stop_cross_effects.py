"""完整池停止方向跨组审计；不更新head、不扫描步长。"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stop_cross_effects import cross_directional_derivatives, first_stop_preservation_bound
from experiments.stop_objective_audit import OBJECTIVES, audit_head_gradients
from scripts.audit_stop_objectives import CONFIG_SHA, REPORT_SHA, digest, load_verified_inputs


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('拒绝覆盖跨组审计目录')
    started = time.perf_counter()
    torch.set_num_threads(1)
    cache, weight, bias, rows, threshold, inputs = load_verified_inputs()
    metrics, gradients = audit_head_gradients(cache['features'], cache['lengths'], weight, bias,
                                             threshold=threshold, return_gradients=True)
    full = {mode: gradients[mode] for mode in OBJECTIVES}
    groups = {
        'original_2400': list(range(2400)),
        'original_gt_valid': [i for i, row in enumerate(rows) if row['teacher_index_prediction']['joint_valid']],
        'original_gt_invalid': [i for i, row in enumerate(rows) if not row['teacher_index_prediction']['joint_valid']],
        'original_parent_deployment_valid': [i for i, row in enumerate(rows) if row['deployment_endpoint']['joint_valid']],
        'original_no_joint_point': [i for i, row in enumerate(rows) if row['category'] == 'no_joint_point'],
        'existing_recovery_518': list(range(2400, 2918)),
    }
    for scene in sorted({row['scene_name'] for row in rows}):
        groups['scene:' + scene] = [i for i, row in enumerate(rows) if row['scene_name'] == scene]
    result = {
        'status': 'completed_requires_design_review', 'config_sha256': CONFIG_SHA,
        'prior_report_sha256': REPORT_SHA, 'input_sha256': inputs, 'threshold': threshold,
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
                        (Path(__file__).resolve(), ROOT / 'experiments/stop_cross_effects.py',
                         ROOT / 'experiments/stop_objective_audit.py', ROOT / 'scripts/audit_stop_objectives.py',
                         ROOT / 'experiments/terminal_alignment_audit.py')},
        'runtime': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                    'threads': 1, 'gradient_dtype': 'float64'},
        'full_pool_samples': 2918, 'full_pool_metrics': metrics, 'cross_effects': {},
        'parent_first_stop_protection': {}, 'training_performed': False,
        'model_exported': False, 'automatic_training': False, 'closed_loop_evaluated': False,
    }
    for name, indices in groups.items():
        _, group_gradients = audit_head_gradients(cache['features'][indices], cache['lengths'][indices],
                                                  weight, bias, threshold=threshold, return_gradients=True)
        result['cross_effects'][name] = {
            'samples': len(indices), 'derivatives_under_full_pool_descent':
                cross_directional_derivatives(group_gradients, full),
            'group_objective_derivatives_under_full_pool_descent': {
                mode: float(-torch.dot(group_gradients[mode], full[mode])) for mode in OBJECTIVES},
        }
    retained = groups['original_parent_deployment_valid']
    if len(retained) != 1144 or any(not rows[i]['stop_found'] for i in retained):
        raise ValueError('原合格组分母/停止事件漂移')
    parent_lengths = torch.tensor([rows[i]['stop_length'] for i in retained], dtype=torch.long)
    _, stop_gradients = audit_head_gradients(cache['features'][retained], parent_lengths, weight, bias,
                                            threshold=threshold, return_gradients=True)
    result['parent_decision_reference'] = {
        'samples': len(retained), 'reference': 'saved_parent_stop_index_not_ground_truth_event',
        'derivatives_under_full_pool_descent': cross_directional_derivatives(stop_gradients, full),
    }
    x = cache['features'][retained].double()
    logits = (x @ weight.double().T).squeeze(-1) + bias.double()
    for mode, gradient in full.items():
        norm = float(gradient.norm())
        direction = -gradient / norm if norm else torch.zeros_like(gradient)
        velocity = x @ direction[:-1] + direction[-1]
        bound = first_stop_preservation_bound(logits, velocity, parent_lengths - 1, threshold=threshold)
        offset = bound['limiting_sample_offset']
        if offset is not None:
            source = retained[offset]
            bound['limiting_dataset_index'] = source
            bound['limiting_task_id'] = rows[source]['task_id']
        bound['direction_normalization'] = 'unit_L2_in_129_parameter_space'
        bound['full_gradient_norm'] = norm
        bound['zero_direction'] = norm == 0.
        result['parent_first_stop_protection'][mode] = bound
    result['elapsed_seconds'] = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'parent_first_stop_protection': result['parent_first_stop_protection']}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
