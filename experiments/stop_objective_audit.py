"""固定停止头目标审计；不包含优化器或模型更新。"""
import math

import torch
from torch.nn import functional as F


OBJECTIVES = ('cumulative', 'prefix_event', 'worst_prefix')


def objective_terms(logits, lengths, threshold):
    """返回每样本损失；pre/tail按共同H分母，便于重建原BCE。"""
    if (logits.ndim != 2 or not logits.numel() or not logits.is_floating_point()
            or lengths.shape != (len(logits),) or lengths.device != logits.device
            or not torch.isfinite(logits).all() or not torch.isfinite(lengths).all()
            or not torch.equal(lengths, lengths.long())
            or torch.any(lengths < 1) or torch.any(lengths > logits.shape[1])
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('非法logits/长度/阈值')
    terminal = lengths.long() - 1
    steps = torch.arange(logits.shape[1], device=logits.device)[None]
    prefix = steps < terminal[:, None]
    labels = (~prefix).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pre = (bce * prefix).mean(1)
    tail = (bce * ~prefix).mean(1)
    event_logit = logits.gather(1, terminal[:, None]).squeeze(1)
    cutoff = math.log(threshold / (1 - threshold))
    # 空前缀使用有限常数，避免max(-inf)在后续产生非有限中间量。
    maximum = logits.masked_fill(~prefix, torch.finfo(logits.dtype).min).amax(1)
    has_prefix = terminal > 0
    worst = torch.where(has_prefix, F.softplus(maximum - cutoff), torch.zeros_like(maximum))
    event = F.softplus(cutoff - event_logit)
    return {
        'cumulative': .2 * (pre + tail),
        'prefix_event': .2 * (bce * (steps <= terminal[:, None])).sum(1) / lengths,
        'worst_prefix': .2 * (worst + event) / (1 + has_prefix.to(logits.dtype)),
        'original_prefix': pre, 'original_tail': tail,
        'threshold_prefix': worst, 'threshold_event': event,
    }


def stop_objective(logits, lengths, *, objective, threshold=.26):
    if objective not in OBJECTIVES:
        raise ValueError('未知停止目标')
    return objective_terms(logits, lengths, threshold)[objective].mean()


def audit_head_gradients(features, lengths, weight, bias, *, threshold=.26, chunk_size=64,
                         return_gradients=False):
    """独立float64叶张量求导，不接触源head；分块不改变样本权重。"""
    if (features.ndim != 3 or not len(features) or weight.shape != (1, features.shape[-1])
            or bias.shape != (1,) or lengths.shape != (len(features),)
            or not isinstance(chunk_size, int) or chunk_size < 1
            or not all(torch.isfinite(v).all() for v in (features, weight, bias))):
        raise ValueError('非法特征/head/分块')
    w = weight.detach().cpu().double().clone().requires_grad_(True)
    b = bias.detach().cpu().double().clone().requires_grad_(True)
    totals = {}; gradients = {}
    for begin in range(0, len(features), chunk_size):
        x = features[begin:begin + chunk_size].detach().cpu().double()
        terms = objective_terms((x @ w.T).squeeze(-1) + b,
                                lengths[begin:begin + chunk_size].detach().cpu(), threshold)
        for name, values in terms.items():
            loss = values.sum() / len(features)
            gw, gb = torch.autograd.grad(loss, (w, b), retain_graph=True)
            gradient = torch.cat((gw.flatten(), gb))
            totals[name] = totals.get(name, 0.) + float(loss.detach())
            gradients[name] = gradients.get(name, torch.zeros_like(gradient)) + gradient.detach()
    result = {}
    for objective in OBJECTIVES:
        grad = gradients[objective]
        result[objective] = {
            'loss': totals[objective], 'gradient_norm': float(grad.norm()),
            'directional_derivatives': {
                name: float(-torch.dot(gradients[name], grad))
                for name in ('original_prefix', 'original_tail', 'threshold_prefix', 'threshold_event')
            },
        }
    # 显式选择时返回独立计算出的向量；默认旧入口输出结构不变。
    return (result, gradients) if return_gradients else result
