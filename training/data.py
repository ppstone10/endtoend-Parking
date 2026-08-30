"""训练与开环评估共享的数据坐标转换和 batch 准备。"""

from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np
import torch

from .trainer import Batch


def to_local(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """将全局轨迹点转换到车辆起始局部坐标系。"""
    values = np.asarray(points, dtype=np.float32)
    delta_x = values[:, 0] - x
    delta_y = values[:, 1] - y
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    local = np.empty_like(values)
    local[:, 0] = cos_yaw * delta_x + sin_yaw * delta_y
    local[:, 1] = -sin_yaw * delta_x + cos_yaw * delta_y
    yaw_delta = values[:, 2] - yaw
    local[:, 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    return local


def prepare_batches(
    data: dict,
    *,
    horizon: int,
    batch_size: int,
) -> tuple[Batch, ...]:
    """转换 NPZ 数据并补齐到模型 horizon；禁止截断有效轨迹。"""
    if horizon <= 0 or batch_size <= 0:
        raise ValueError("horizon 和 batch_size 必须为正")
    required = {"bevs", "goals", "states", "trajs", "masks", "dt"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"数据集缺少字段：{', '.join(missing)}")

    bevs = np.asarray(data["bevs"])
    goals = np.asarray(data["goals"])
    states = np.asarray(data["states"])
    trajectories = np.asarray(data["trajs"])
    masks = np.asarray(data["masks"])
    sample_count = bevs.shape[0]
    if sample_count == 0:
        raise ValueError("数据集不能为空")
    if not (
        goals.shape == (sample_count, 3)
        and states.ndim == 2
        and states.shape[0] == sample_count
        and states.shape[1] >= 5
        and trajectories.ndim == 3
        and trajectories.shape[0] == sample_count
        and trajectories.shape[2] == 3
        and masks.shape == trajectories.shape[:2]
    ):
        raise ValueError("数据集数组形状不符合训练契约")
    lengths = _prefix_lengths(masks)
    if int(lengths.max()) > horizon:
        raise ValueError(
            f"模型 horizon={horizon} 小于数据有效轨迹长度 {int(lengths.max())}"
        )

    local_goals = np.empty_like(goals)
    padded_trajectories = np.zeros((sample_count, horizon, 3), dtype=np.float32)
    padded_masks = np.zeros((sample_count, horizon), dtype=np.float32)
    for index in range(sample_count):
        x, y, yaw = (float(value) for value in states[index, :3])
        local_goals[index] = to_local(goals[index : index + 1], x, y, yaw)[0]
        length = int(lengths[index])
        padded_trajectories[index, :length] = to_local(
            trajectories[index, :length], x, y, yaw
        )
        padded_masks[index, :length] = 1.0

    batches: list[Batch] = []
    for start in range(0, sample_count, batch_size):
        selection = slice(start, start + batch_size)
        batches.append(
            (
                torch.as_tensor(bevs[selection], dtype=torch.float32),
                torch.as_tensor(local_goals[selection], dtype=torch.float32),
                torch.as_tensor(states[selection, 3:5], dtype=torch.float32),
                torch.as_tensor(padded_trajectories[selection], dtype=torch.float32),
                torch.as_tensor(padded_masks[selection], dtype=torch.float32),
            )
        )
    return tuple(batches)


def epoch_batches(
    batches: tuple[Batch, ...],
    *,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> tuple[Batch, ...]:
    """按 epoch 确定性重排样本，并保持原 batch 容量。

    输入 batch 保持不变；重排后的各字段共享同一索引顺序。
    """
    if not batches:
        return ()
    if not shuffle:
        return batches
    sizes = [int(batch[0].shape[0]) for batch in batches]
    if any(size <= 0 for size in sizes):
        raise ValueError("batch 不能为空")
    if any(any(int(tensor.shape[0]) != size for tensor in batch) for batch, size in zip(batches, sizes)):
        raise ValueError("batch 内字段的样本数必须一致")
    cumulative: list[int] = []
    total = 0
    for size in sizes:
        total += size
        cumulative.append(total)
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(total, generator=generator).tolist()
    capacity = sizes[0]
    shuffled: list[Batch] = []
    for start in range(0, total, capacity):
        fields: list[torch.Tensor] = []
        selected = permutation[start : start + capacity]
        for field_index in range(len(batches[0])):
            rows: list[torch.Tensor] = []
            for global_index in selected:
                batch_index = bisect_right(cumulative, global_index)
                previous_total = cumulative[batch_index - 1] if batch_index else 0
                rows.append(batches[batch_index][field_index][global_index - previous_total])
            fields.append(torch.stack(rows, dim=0))
        shuffled.append(tuple(fields))  # type: ignore[arg-type]
    return tuple(shuffled)


def model_horizon(model: torch.nn.Module) -> int:
    """读取 v0 或变长模型的公开 horizon。"""
    value = getattr(model, "horizon", getattr(model, "max_horizon", None))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("模型没有有效的 horizon/max_horizon")
    return value


def validate_model_dataset(model: torch.nn.Module, data: dict) -> None:
    """在运行前核对模型与数据的通道数和轨迹时间步长。"""
    bevs = np.asarray(data.get("bevs"))
    if bevs.ndim != 4:
        raise ValueError("数据集 bevs 必须是 (B,C,H,W)")
    first_convolution = next(
        (module for module in model.modules() if isinstance(module, torch.nn.Conv2d)),
        None,
    )
    if first_convolution is None or first_convolution.in_channels != bevs.shape[1]:
        raise ValueError("模型 bev_channels 与数据集不一致")
    model_dt = getattr(model, "dt", None)
    data_dt = float(np.asarray(data.get("dt")).reshape(-1)[0])
    if model_dt is None or not np.isclose(float(model_dt), data_dt):
        raise ValueError("模型 dt 与数据集轨迹 dt 不一致")


def _prefix_lengths(masks: np.ndarray) -> np.ndarray:
    if not np.isfinite(masks).all() or not np.logical_or(masks == 0, masks == 1).all():
        raise ValueError("mask 只能包含有限的 0/1")
    active = masks.astype(bool)
    if np.any(active[:, 1:] & ~active[:, :-1]):
        raise ValueError("mask 的有效点必须是连续前缀")
    lengths = active.sum(axis=1)
    if np.any(lengths == 0):
        raise ValueError("每条样本至少需要一个有效轨迹点")
    return lengths
