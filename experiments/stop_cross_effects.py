"""完整池方向的交叉导数与固定线性头首次停止索引保护范围。"""
import math

import torch


REFERENCE_TERMS = ('original_prefix', 'original_tail', 'threshold_prefix', 'threshold_event')


def cross_directional_derivatives(group_gradients, full_gradients):
    """每一行固定完整池方向，每一列是当前组的参考损失。"""
    result = {}
    for name, gradient in full_gradients.items():
        if (gradient.ndim != 1 or not torch.isfinite(gradient).all()
                or any(group_gradients[key].shape != gradient.shape
                       or not torch.isfinite(group_gradients[key]).all() for key in REFERENCE_TERMS)):
            raise ValueError('非法交叉梯度')
        result[name] = {key: float(-torch.dot(group_gradients[key], gradient)) for key in REFERENCE_TERMS}
    return result


def first_stop_preservation_bound(logits, logit_velocity, stop_indices, *, threshold):
    """求z+eta*v保持既有首次停止索引的eta上界，不实际施加任何步长。

    开区间0<=eta<bound均保持；事件约束等号本身仍可停止，前缀等号则改变首索引。
    """
    if (logits.ndim != 2 or not logits.numel() or logit_velocity.shape != logits.shape
            or stop_indices.shape != (len(logits),) or not math.isfinite(threshold)
            or not 0 < threshold < 1 or not logits.is_floating_point()
            or not torch.isfinite(logits).all() or not torch.isfinite(logit_velocity).all()
            or not torch.isfinite(stop_indices).all() or not torch.equal(stop_indices, stop_indices.long())
            or torch.any(stop_indices < 0) or torch.any(stop_indices >= logits.shape[1])):
        raise ValueError('非法停点保护输入')
    cutoff = math.log(threshold / (1 - threshold))
    indices = stop_indices.long()
    steps = torch.arange(logits.shape[1], device=logits.device)[None]
    prefix = steps < indices[:, None]
    event = steps == indices[:, None]
    if torch.any(prefix & (logits >= cutoff)) or torch.any(event & (logits < cutoff)):
        raise ValueError('指定停点不是float64首次越阈')
    bounds = torch.full_like(logits, torch.inf)
    approaching_prefix = prefix & (logit_velocity > 0)
    leaving_event = event & (logit_velocity < 0)
    bounds[approaching_prefix] = (cutoff - logits[approaching_prefix]) / logit_velocity[approaching_prefix]
    bounds[leaving_event] = (logits[leaving_event] - cutoff) / -logit_velocity[leaving_event]
    tight = bounds.flatten().argmin()
    value = float(bounds.flatten()[tight])
    sample, step = divmod(int(tight), logits.shape[1])
    return {
        'samples': len(logits),
        'positive_step_supremum': value if math.isfinite(value) else None,
        'unbounded': not math.isfinite(value),
        'limiting_sample_offset': sample if math.isfinite(value) else None,
        'limiting_point_index': step if math.isfinite(value) else None,
        'limiting_constraint': ('prefix_crossing' if step < int(indices[sample]) else 'event_lost')
                               if math.isfinite(value) else None,
        'samples_with_prefix_scores_increasing': int(approaching_prefix.any(1).sum()),
        'samples_with_event_score_decreasing': int(leaving_event.any(1).sum()),
    }
