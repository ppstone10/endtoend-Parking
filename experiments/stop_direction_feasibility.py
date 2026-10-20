"""三个既有方向凸组合的有界可行性检查；不是训练优化器。"""
from itertools import combinations

import torch

from experiments.stop_cross_effects import first_stop_preservation_bound


def _clip_polygon(vertices, normal, bound, tolerance):
    output = []
    previous = vertices[-1]
    previous_distance = float(previous @ normal - bound)
    for current in vertices:
        current_distance = float(current @ normal - bound)
        inside = current_distance <= tolerance
        previous_inside = previous_distance <= tolerance
        if inside != previous_inside:
            ratio = previous_distance / (previous_distance - current_distance)
            output.append(previous + max(0., min(1., ratio)) * (current - previous))
        if inside:
            output.append(current)
        previous, previous_distance = current, current_distance
    unique = []
    for value in output:
        if not unique or float((value - unique[-1]).norm()) > 1e-14:
            unique.append(value)
    if len(unique) > 1 and float((unique[0] - unique[-1]).norm()) <= 1e-14:
        unique.pop()
    return torch.stack(unique) if unique else torch.empty((0, 3), dtype=vertices.dtype)


def clip_direction_simplex(matrix, bounds, objective, *, max_cuts=128, tolerance=1e-12):
    """在三角形内裁剪半平面；硬预算，空域仅指当前float64有限族。"""
    if (matrix.ndim != 2 or matrix.shape[1] != 3 or bounds.shape != (len(matrix),)
            or objective.shape != (3,) or not isinstance(max_cuts, int) or max_cuts < 0
            or not 0 < tolerance < 1 or not all(torch.isfinite(v).all() for v in (matrix, bounds, objective))):
        raise ValueError('非法单纯形约束/预算')
    matrix = matrix.detach().cpu().double()
    bounds = bounds.detach().cpu().double()
    objective = objective.detach().cpu().double()
    vertices = torch.eye(3, dtype=torch.double)
    cuts = []
    for iteration in range(max_cuts + 1):
        residuals = matrix @ vertices.T - bounds[:, None]
        violation = float(residuals.max()) if residuals.numel() else 0.
        if violation <= tolerance:
            return {'status': 'polygon_complete', 'vertices': vertices.tolist(), 'cuts': cuts,
                    'max_vertex_violation': violation, 'objective_values': (vertices @ objective).tolist()}
        if iteration == max_cuts:
            return {'status': 'budget_exhausted', 'cuts': cuts, 'max_vertex_violation': violation}
        row = int(residuals.flatten().argmax()) // len(vertices)
        if row in cuts:
            return {'status': 'numerical_stall', 'cuts': cuts, 'max_vertex_violation': violation}
        cuts.append(row)
        vertices = _clip_polygon(vertices, matrix[row], bounds[row], tolerance)
        if not len(vertices):
            return {'status': 'empty_family_polygon_float64', 'cuts': cuts}
    raise AssertionError('预算循环错误')


def select_verified_direction(polygon, matrix, directions, objective_gradient):
    """只复核已返回顶点；不以近零抵消向量冒充非零可行方向。"""
    if polygon['status'] != 'polygon_complete':
        return {'status': 'no_verified_witness', 'reason': polygon['status']}
    scored = []
    for coefficients in polygon['vertices']:
        weights = torch.tensor(coefficients, dtype=torch.double).clamp_min(0)
        weights /= weights.sum()
        direction = weights @ directions
        norm = float(direction.norm())
        if norm <= 1e-10:
            continue
        direction = direction / norm
        residual = float((matrix @ weights).max()) / norm if len(matrix) else 0.
        slope = float(objective_gradient @ direction)
        if residual <= 1e-9 and slope < -1e-9:
            scored.append((float(objective_gradient @ (weights @ directions)), {
                'status': 'verified_nonzero_witness', 'weights': weights.tolist(),
                'direction': direction.tolist(), 'mixture_norm': norm,
                'max_normalized_constraint_violation': residual, 'unit_objective_slope': slope,
            }))
    if not scored:
        return {'status': 'no_verified_witness', 'reason': 'no_vertex_passes_unit_direction_residual_and_descent'}
    return min(scored, key=lambda item: item[0])[1]


def positive_halfspace_certificate(active_rows, *, max_subsets=2000):
    """非负切面组合在三个系数上均为正，证明当前单纯形族不相容。"""
    if (active_rows.ndim != 2 or active_rows.shape[1] != 3
            or not torch.isfinite(active_rows).all() or not isinstance(max_subsets, int) or max_subsets < 1):
        raise ValueError('非法证据切面/预算')
    active_rows = active_rows.detach().cpu().double()
    checked = 0
    for size in range(1, min(3, len(active_rows)) + 1):
        for indices in combinations(range(len(active_rows)), size):
            if checked >= max_subsets:
                return {'status': 'certificate_budget_exhausted', 'subsets_checked': checked}
            checked += 1
            rows = active_rows[list(indices)]
            coefficients = torch.linalg.lstsq(rows.T, torch.ones(3, dtype=torch.double), driver='gelsd').solution
            if not torch.isfinite(coefficients).all() or (coefficients < -1e-12).any():
                continue
            coefficients = coefficients.clamp_min(0)
            if float(coefficients.sum()) <= 1e-14:
                continue
            coefficients /= coefficients.sum()
            combined = coefficients @ rows
            gap = float(combined.min())
            if gap > 1e-9:
                return {'status': 'verified_family_infeasibility_certificate',
                        'active_offsets': list(indices), 'nonnegative_coefficients': coefficients.tolist(),
                        'combined_positive_row': combined.tolist(), 'positive_gap': gap,
                        'subsets_checked': checked}
    return {'status': 'no_certificate_found', 'subsets_checked': checked}


def parent_protection_matrices(features, logits, stops, joint_valid, directions, *, threshold):
    """原停点作为合格锚点；允许合格前缀越阈不等于生成训练标签。"""
    if (features.ndim != 3 or logits.shape != features.shape[:2] or joint_valid.shape != logits.shape
            or joint_valid.dtype != torch.bool or directions.shape != (3, features.shape[-1] + 1)
            or not torch.isfinite(features).all() or not torch.isfinite(directions).all()):
        raise ValueError('非法保护矩阵输入')
    first_stop_preservation_bound(logits, torch.zeros_like(logits), stops, threshold=threshold)
    stops = stops.long()
    if not joint_valid[torch.arange(len(stops)), stops].all():
        raise ValueError('保护锚点位姿不合格')
    x = features.double()
    # 正规化每个特征行只改变约束数值尺度，不改变不等式含义。
    velocity = (x @ directions[:, :-1].T + directions[:, -1]) / torch.sqrt(x.square().sum(2) + 1)[:, :, None]
    prefix = torch.arange(logits.shape[1])[None] < stops[:, None]
    event = -velocity[torch.arange(len(stops)), stops]
    strict = torch.cat((velocity[prefix], event))
    compatible = torch.cat((velocity[prefix & ~joint_valid], event))
    return strict, compatible
