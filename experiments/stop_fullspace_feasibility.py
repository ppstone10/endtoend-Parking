"""完整停止头空间的有界共同下降诊断；只迭代数学对偶量。"""
import math
import time

import torch

from experiments.stop_cross_effects import first_stop_preservation_bound


def fullspace_parent_constraints(features, logits, stops, joint_valid, *, threshold):
    if (features.ndim != 3 or features.shape[:2] != logits.shape or joint_valid.shape != logits.shape
            or joint_valid.dtype != torch.bool or not torch.isfinite(features).all()):
        raise ValueError('非法全空间保护输入')
    first_stop_preservation_bound(logits, torch.zeros_like(logits), stops, threshold=threshold)
    stops = stops.long()
    if not joint_valid[torch.arange(len(stops)), stops].all():
        raise ValueError('保护锚点位姿不合格')
    prefix = torch.arange(logits.shape[1])[None] < stops[:, None]

    def augmented(values):
        values = values.detach().cpu().double()
        rows = torch.cat((values, torch.ones(len(values), 1, dtype=torch.double)), dim=1)
        return rows / rows.norm(dim=1, keepdim=True)

    event = -augmented(features[torch.arange(len(stops)), stops])
    strict = torch.cat((augmented(features[prefix]), event))
    compatible = torch.cat((augmented(features[prefix & ~joint_valid]), event))
    return strict, compatible


def protected_common_descent(gradients, protection, *, max_iterations=256, max_seconds=8., progress=None):
    """Gd<=-t、Pd<=0、||d||<=1；合法对偶v的范数为t的上界。

    不限制d在旧梯度凸族内：保护锥的对偶行同样参与方向构造。
    预算耗尽不提供不可行结论；所有见证按单位方向复核完整P。
    """
    if (gradients.ndim != 2 or not len(gradients) or not gradients.shape[1]
            or protection.ndim != 2 or protection.shape[1] != gradients.shape[1]
            or not all(torch.isfinite(v).all() for v in (gradients, protection))
            or not isinstance(max_iterations, int) or max_iterations < 0
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('非法梯度/保护/预算')
    started = time.perf_counter()
    g = gradients.detach().cpu().double()
    p = protection.detach().cpu().double()
    lambdas = torch.full((len(g),), 1 / len(g), dtype=torch.double)
    active = []; multipliers = torch.empty(0, dtype=torch.double)
    p_norms = p.square().sum(1).clamp_min(1e-30)
    # 小规模Gd=-1可快速产生全空间候选；只接受完整P验证通过的解。
    # 不把最小二乘残差或rank当成原不等式可行/不可行证据。
    linear = torch.linalg.lstsq(g, -torch.ones(len(g), dtype=torch.double), driver='gelsd')
    candidate = linear.solution
    candidate_norm = float(candidate.norm())
    linear_info = {'rank': int(linear.rank), 'equation_residual': float((g @ candidate + 1).norm())}
    if candidate_norm > 1e-14:
        direction = candidate / candidate_norm
        slopes = g @ direction
        margin = float(-slopes.max())
        violation = float((p @ direction).max()) if len(p) else 0.
        if margin > 1e-8 and violation <= 1e-9:
            vector = lambdas @ g
            return {
                'status': 'verified_common_descent', 'iterations': 0,
                'elapsed_seconds': time.perf_counter() - started, 'linear_witness_check': linear_info,
                'witness': {'source': 'least_squares_candidate_fully_verified', 'direction': direction.tolist(),
                            'normalized_metric_slopes': slopes.tolist(), 'minimum_common_descent': margin,
                            'max_protection_violation': violation},
                'candidate_minimum_common_descent': margin, 'candidate_max_protection_violation': violation,
                'dual': {'group_coefficients': lambdas.tolist(), 'active_protection_indices': [],
                         'protection_coefficients': [], 'vector': vector.tolist(),
                         'unit_margin_upper_bound': float(vector.norm())},
            }
    status = 'iteration_budget_exhausted'
    witness = None
    for iteration in range(max_iterations + 1):
        group_vector = lambdas @ g
        vector = group_vector + (multipliers @ p[active] if active else 0.)
        norm = float(vector.norm())
        p_products = p @ vector
        if norm > 1e-14:
            direction = -vector / norm
            slopes = g @ direction
            margin = float(-slopes.max())
            violation = float(-p_products.min() / norm) if len(p) else 0.
            if margin > 1e-8 and violation <= 1e-9:
                witness = {'direction': direction.tolist(), 'normalized_metric_slopes': slopes.tolist(),
                           'minimum_common_descent': margin, 'max_protection_violation': violation}
                status = 'verified_common_descent'
                break
        else:
            margin, violation = 0., 0.
        if norm <= 1e-10:
            status = 'certified_small_unit_margin_upper_bound'
            break
        if progress and iteration % 64 == 0:
            progress(iteration, norm, violation)
        if time.perf_counter() - started >= max_seconds:
            status = 'time_budget_exhausted'
            break
        if iteration == max_iterations:
            break
        # 在lambda单纯形上精确线搜索；该步只改变数学权重。
        selected = int((g @ vector).argmin())
        change = g[selected] - group_vector
        squared = float(change @ change)
        step = max(0., min(1., -float(vector @ change) / squared)) if squared > 1e-30 else 0.
        lambdas *= 1 - step
        lambdas[selected] += step
        vector = vector + step * change
        # 保护乘子非负坐标下降，可减少已有乘子而非只累加新行。
        if len(p):
            products = p @ vector
            new_index = int(products.argmin())
            candidates = active + ([] if new_index in active else [new_index])
            values = torch.cat((multipliers, torch.zeros(1, dtype=torch.double))) if len(candidates) > len(active) else multipliers
            denominators = p_norms[candidates]
            deltas = torch.maximum(-values, -products[candidates] / denominators)
            changes = deltas * products[candidates] + .5 * deltas.square() * denominators
            choice = int(changes.argmin())
            if float(changes[choice]) < -1e-30:
                if choice == len(active):
                    active.append(new_index)
                    multipliers = torch.cat((multipliers, torch.zeros(1, dtype=torch.double)))
                multipliers[choice] += deltas[choice]
    # vector在循环起点由合法系数重构；保存完整数值证据而不保存模型参数。
    return {
        'status': status, 'iterations': iteration, 'elapsed_seconds': time.perf_counter() - started,
        'linear_witness_check': linear_info,
        'witness': witness, 'candidate_minimum_common_descent': margin,
        'candidate_max_protection_violation': violation,
        'dual': {'group_coefficients': lambdas.tolist(), 'active_protection_indices': active,
                 'protection_coefficients': multipliers.tolist(), 'vector': vector.tolist(),
                 'unit_margin_upper_bound': norm},
    }
