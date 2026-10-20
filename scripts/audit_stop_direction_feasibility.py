"""固定三方向族的保护可行性短审计；不训练、不生成模型。"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stop_direction_feasibility import (
    clip_direction_simplex, parent_protection_matrices, positive_halfspace_certificate, select_verified_direction,
)
from experiments.stop_objective_audit import OBJECTIVES, audit_head_gradients
from scripts.audit_stop_objectives import CONFIG_SHA, REPORT_SHA, digest, load_verified_inputs


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('拒绝覆盖方向族审计目录')
    started = time.perf_counter()
    torch.set_num_threads(1)
    cache, weight, bias, rows, threshold, inputs = load_verified_inputs()
    _, full = audit_head_gradients(cache['features'], cache['lengths'], weight, bias,
                                   threshold=threshold, return_gradients=True)
    directions = torch.stack([-full[mode] / full[mode].norm() for mode in OBJECTIVES])
    objective_gradient = full['worst_prefix'] / full['worst_prefix'].norm()
    groups = {
        'full_2918': list(range(2918)),
        'original_gt_valid': [i for i, row in enumerate(rows) if row['teacher_index_prediction']['joint_valid']],
        'S4': [i for i, row in enumerate(rows) if row['scene_name'].startswith('S4_')],
        'S8': [i for i, row in enumerate(rows) if row['scene_name'].startswith('S8_')],
    }
    constraint_rows = []; labels = []
    for name, indices in groups.items():
        if name == 'full_2918':
            gradients = full
        else:
            _, gradients = audit_head_gradients(cache['features'][indices], cache['lengths'][indices],
                                                 weight, bias, threshold=threshold, return_gradients=True)
        for term in ('threshold_prefix', 'threshold_event'):
            gradient = gradients[term]
            constraint_rows.append((gradient / gradient.norm().clamp_min(1e-30)) @ directions.T)
            labels.append(name + ':' + term)
    metrics = torch.stack(constraint_rows)
    retained = [i for i, row in enumerate(rows) if row['deployment_endpoint']['joint_valid']]
    if len(retained) != 1144 or any(not rows[i]['stop_found'] for i in retained):
        raise ValueError('原合格样本身份错误')
    x = cache['features'][retained]
    logits = (x.double() @ weight.double().T).squeeze(-1) + bias.double()
    stops = torch.tensor([rows[i]['stop_length'] - 1 for i in retained])
    points, goals = cache['points'][retained], cache['goals'][retained]
    delta = points[:, :, 2] - goals[:, None, 2]
    joint = ((torch.linalg.vector_norm(points[:, :, :2] - goals[:, None, :2], dim=2) <= .3)
             & (torch.atan2(torch.sin(delta), torch.cos(delta)).abs() <= math.pi / 18))
    strict, compatible = parent_protection_matrices(x, logits, stops, joint, directions, threshold=threshold)
    scenarios = {
        'aggregate_only': metrics[:2],
        'guarded_metrics': metrics,
        'strict_index_protection': torch.cat((metrics, strict)),
        'goal_compatible_anchor_protection': torch.cat((metrics, compatible)),
        'diagnostic_only_strict_protection': strict,
        'diagnostic_only_goal_compatible_protection': compatible,
    }
    singular = torch.linalg.svdvals(directions)
    result = {
        'status': 'completed_requires_design_review', 'config_sha256': CONFIG_SHA,
        'prior_report_sha256': REPORT_SHA, 'input_sha256': inputs,
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
                        (Path(__file__).resolve(), ROOT / 'experiments/stop_direction_feasibility.py',
                         ROOT / 'experiments/stop_objective_audit.py', ROOT / 'experiments/stop_cross_effects.py',
                         ROOT / 'scripts/audit_stop_objectives.py', ROOT / 'experiments/terminal_alignment_audit.py')},
        'runtime': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'threads': 1},
        'settings': {'max_cuts': 128, 'clip_tolerance': 1e-12, 'unit_residual_tolerance': 1e-9,
                     'min_direction_norm': 1e-10, 'strict_descent_tolerance': 1e-9,
                     'max_certificate_subsets': 2000},
        'family': {'names': list(OBJECTIVES), 'unit_directions': directions.tolist(),
                   'singular_values': singular.tolist(),
                   'condition_number': float(singular.max() / singular.min()) if float(singular.min()) > 1e-12 else None},
        'threshold': threshold, 'metric_constraints': labels, 'normalized_metric_matrix': metrics.tolist(),
        'group_samples': {name: len(indices) for name, indices in groups.items()},
        'parent_protection': {'samples': len(retained), 'strict_rows': len(strict),
                              'compatible_rows': len(compatible)},
        'scenarios': {}, 'training_performed': False, 'model_exported': False,
        'automatic_training': False, 'closed_loop_evaluated': False,
    }
    for name, matrix in scenarios.items():
        polygon = clip_direction_simplex(matrix, torch.zeros(len(matrix)), directions @ objective_gradient)
        active = matrix[polygon['cuts']]
        polygon['active_halfspaces'] = active.tolist()
        certificate = positive_halfspace_certificate(active) if polygon['status'] == 'empty_family_polygon_float64' else None
        witness = select_verified_direction(polygon, matrix, directions, objective_gradient)
        if witness['status'] == 'verified_nonzero_witness':
            mixture = torch.tensor(witness['weights'], dtype=torch.double)
            slopes = metrics @ mixture / witness['mixture_norm']
            witness['all_guarded_normalized_metric_slopes'] = dict(zip(labels, slopes.tolist()))
        result['scenarios'][name] = {'constraint_rows': len(matrix), 'polygon': polygon, 'witness': witness,
                                    'infeasibility_certificate': certificate,
                                    'diagnostic_only': name.startswith('diagnostic_only_')}
    result['elapsed_seconds'] = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'elapsed_seconds': result['elapsed_seconds'], 'parent_protection': result['parent_protection'],
                      'scenarios': {name: {'polygon': value['polygon']['status'], 'witness_status': value['witness']['status'],
                                           'certificate': value['infeasibility_certificate']}
                                    for name, value in result['scenarios'].items()}}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
