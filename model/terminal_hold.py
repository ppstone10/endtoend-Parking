"""可选专家终点保持：不改变有效前缀与停止标签。"""
from __future__ import annotations

import math
import torch

from .network import _yaw_residual


def hold_terminal_targets(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """在新张量中用专家末位姿补齐后缀，供一致的teacher forcing使用。"""
    if (target.ndim != 3 or target.shape[-1] != 3 or not target.shape[0]
            or not target.shape[1] or mask.shape != target.shape[:2]):
        raise ValueError('终点保持要求非空target=(B,T,3)、mask=(B,T)')
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(mask).all()):
        raise ValueError('终点保持输入必须有限')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('终点保持mask必须为0/1')
    if bool((mask[:, 1:] > mask[:, :-1]).any()):
        raise ValueError('终点保持mask必须是有效连续前缀')
    lengths = mask.sum(1).long()
    if bool((lengths <= 0).any()):
        raise ValueError('终点保持每条样本至少需要一个有效点')
    final = target.gather(1, (lengths - 1)[:, None, None].expand(-1, 1, 3))
    suffix = torch.arange(target.shape[1], device=target.device)[None, :] >= lengths[:, None]
    return torch.where(suffix[:, :, None], final, target)


def terminal_hold_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *,
    yaw_weight: float = 10.0, yaw_error_mode: str = 'raw',
) -> torch.Tensor:
    """每个有后缀样本等权，约束专家有效前缀之后的预测保持专家末位姿。"""
    if pred.shape != target.shape or not bool(torch.isfinite(pred).all()):
        raise ValueError('终点保持pred/target必须同形且有限')
    if not math.isfinite(yaw_weight) or yaw_weight < 0:
        raise ValueError('yaw_weight必须为有限非负数')
    held = hold_terminal_targets(target, mask)
    suffix = mask == 0
    counts = suffix.sum(1)
    positions = (pred[..., :2] - held[..., :2]).square().sum(-1)
    yaws = _yaw_residual(pred[..., 2], held[..., 2], yaw_error_mode).square()
    per_sequence = ((positions + yaw_weight * yaws) * suffix).sum(1) / counts.clamp(min=1)
    observed = counts > 0
    # 可反传的0；无后缀样本不稀释有后缀样本的权重。
    return (per_sequence * observed).sum() / observed.sum().clamp(min=1)
