"""MineParkingNet 验证集开环轨迹指标。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Iterable

import numpy as np
import torch

from training.trainer import Batch


@dataclass(frozen=True)
class OpenLoopMetrics:
    samples: int
    valid_points: int
    ade_m: float
    fde_m: float
    yaw_mae_rad: float
    inference_ms_per_sample: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_open_loop_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    masks: np.ndarray | torch.Tensor,
    *,
    inference_ms_per_sample: float = 0.0,
) -> OpenLoopMetrics:
    """按目标有效前缀计算 ADE、FDE 与环绕航向 MAE。"""
    predicted = _array(predictions)
    expected = _array(targets)
    mask = _array(masks)
    if predicted.ndim != 3 or predicted.shape[2] != 3:
        raise ValueError("predictions 必须是 (B,N,3)")
    if expected.ndim != 3 or expected.shape[2] != 3:
        raise ValueError("targets 必须是 (B,N,3)")
    if predicted.shape[0] != expected.shape[0] or mask.shape != expected.shape[:2]:
        raise ValueError("predictions、targets 与 masks 的 batch/shape 不一致")
    if not (np.isfinite(predicted).all() and np.isfinite(expected).all() and np.isfinite(mask).all()):
        raise ValueError("开环指标输入必须为有限值")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("masks 只能包含 0/1")
    active = mask.astype(bool)
    if np.any(active[:, 1:] & ~active[:, :-1]):
        raise ValueError("masks 的有效点必须是连续前缀")
    lengths = active.sum(axis=1)
    if np.any(lengths == 0):
        raise ValueError("每条样本至少需要一个有效点")
    if predicted.shape[1] < int(lengths.max()):
        raise ValueError("预测 horizon 小于目标有效轨迹长度")

    position_errors: list[np.ndarray] = []
    yaw_errors: list[np.ndarray] = []
    final_errors: list[float] = []
    for index, length_value in enumerate(lengths):
        length = int(length_value)
        delta_xy = predicted[index, :length, :2] - expected[index, :length, :2]
        errors = np.linalg.norm(delta_xy, axis=1)
        position_errors.append(errors)
        final_errors.append(float(errors[-1]))
        yaw_delta = predicted[index, :length, 2] - expected[index, :length, 2]
        yaw_errors.append(np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))))

    all_positions = np.concatenate(position_errors)
    all_yaws = np.concatenate(yaw_errors)
    return OpenLoopMetrics(
        samples=int(expected.shape[0]),
        valid_points=int(all_positions.size),
        ade_m=float(all_positions.mean()),
        fde_m=float(np.mean(final_errors)),
        yaw_mae_rad=float(all_yaws.mean()),
        inference_ms_per_sample=float(inference_ms_per_sample),
    )


def evaluate_open_loop(
    model: torch.nn.Module,
    batches: Iterable[Batch],
    *,
    device: str | torch.device = "cpu",
) -> OpenLoopMetrics:
    """对已准备 batch 执行无梯度推理并聚合统一指标。"""
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    elapsed = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in batches:
            bev, goal, state, target, mask = (
                tensor.to(target_device) for tensor in batch
            )
            started = time.perf_counter()
            forward_with_stop = getattr(model, "forward_with_stop", None)
            if callable(forward_with_stop):
                output = forward_with_stop(bev, goal, state).points
            else:
                output = model(bev, goal, state)
            elapsed += time.perf_counter() - started
            sample_count += int(bev.shape[0])
            predictions.append(output.detach().cpu())
            targets.append(target.detach().cpu())
            masks.append(mask.detach().cpu())
    if sample_count == 0:
        raise ValueError("评估 batches 不能为空")
    return compute_open_loop_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(masks),
        inference_ms_per_sample=elapsed * 1000.0 / sample_count,
    )


def _array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
