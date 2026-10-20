"""一次预定有限停止头更新的数值检查；无训练/步长搜索。"""
from copy import deepcopy
import math

import torch

from experiments.stop_objective_audit import objective_terms


def quarter_bound_head(parent_head, direction, bound):
    if (not math.isfinite(bound) or bound <= 0 or direction.shape != (parent_head.in_features + 1,)
            or not torch.isfinite(direction).all() or abs(float(direction.norm()) - 1.) > 1e-9
            or parent_head.weight.dtype != torch.float32):
        raise ValueError('非法单次更新输入/有限上界')
    head = deepcopy(parent_head).cpu()
    eta = bound / 4
    with torch.no_grad():
        head.weight.copy_((parent_head.weight.double() + eta * direction[:-1].double()[None]).float())
        head.bias.copy_((parent_head.bias.double() + eta * direction[-1].double()).float())
    return head, eta


def cached_float32_logits(head, features):
    if features.ndim != 3 or features.dtype != torch.float32 or head.weight.dtype != torch.float32:
        raise ValueError('缓存/停止头必须float32')
    values = []
    with torch.no_grad():
        for begin in range(0, len(features), 8):
            batch = features[begin:begin + 8]
            values.append(torch.stack([head(batch[:, step].contiguous()).squeeze(-1)
                                       for step in range(batch.shape[1])], dim=1))
    return torch.cat(values)


def finite_loss_changes(parent_logits, candidate_logits, lengths, groups, *, threshold):
    result = {}
    for dtype, name in ((torch.float32, 'float32'), (torch.float64, 'float64_reduction')):
        parent_terms = objective_terms(parent_logits.to(dtype), lengths, threshold)
        candidate_terms = objective_terms(candidate_logits.to(dtype), lengths, threshold)
        result[name] = {}
        for group, indices in groups.items():
            result[name][group] = {}
            for term in ('threshold_prefix', 'threshold_event'):
                before = float(parent_terms[term][indices].mean())
                after = float(candidate_terms[term][indices].mean())
                result[name][group][term] = {'parent': before, 'candidate': after, 'delta': after - before}
    return result
