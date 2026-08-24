"""NPZ 数据集统计与 BEV/专家轨迹叠加抽检。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def summarize_dataset(data: dict[str, Any]) -> dict[str, Any]:
    """统计样本数、轨迹长度、倒车距离比例与任务分层数量。"""
    trajs = np.asarray(data["trajs"])
    masks = np.asarray(data["masks"])
    if trajs.shape[:2] != masks.shape:
        raise ValueError("trajs 与 masks 形状不一致")

    lengths: list[float] = []
    reverse_distance = 0.0
    total_distance = 0.0
    for trajectory, mask in zip(trajs, masks):
        count = int(np.count_nonzero(mask))
        points = trajectory[:count]
        if len(points) < 2:
            lengths.append(0.0)
            continue
        delta = np.diff(points[:, :2], axis=0)
        segment_length = np.linalg.norm(delta, axis=1)
        heading = points[:-1, 2]
        signed_progress = delta[:, 0] * np.cos(heading) + delta[:, 1] * np.sin(heading)
        length = float(segment_length.sum())
        lengths.append(length)
        total_distance += length
        reverse_distance += float(segment_length[signed_progress < 0.0].sum())

    metadata = data.get("task_meta") or []
    return {
        "sample_count": int(trajs.shape[0]),
        "trajectory_length_m": _distribution(lengths),
        "reverse_distance_ratio": (
            reverse_distance / total_distance if total_distance > 0.0 else 0.0
        ),
        "scene_counts": _metadata_counts(metadata, "scene_name"),
        "task_type_counts": _metadata_counts(metadata, "task_type"),
        "noise_level_counts": _noise_counts(metadata),
    }


def render_sample_overlay(data: dict[str, Any], index: int, path: str | Path) -> None:
    """保存一张 occupancy/target BEV 与局部专家轨迹叠加图。"""
    import matplotlib.pyplot as plt

    bev_meta = data.get("bev_meta")
    if not isinstance(bev_meta, dict):
        raise ValueError("绘制叠加图需要 schema v2 bev_meta")
    bevs = np.asarray(data["bevs"])
    if not 0 <= index < len(bevs):
        raise IndexError("样本索引越界")
    channels = list(bev_meta["channels"])
    front, back, left, right = (float(value) for value in bev_meta["extent"])
    occupancy = bevs[index, channels.index("occupancy")]

    figure, axis = plt.subplots(figsize=(6.0, 6.0))
    axis.imshow(
        occupancy,
        cmap="Greys",
        origin="upper",
        extent=(-right, left, -back, front),
        interpolation="nearest",
    )
    if "target" in channels:
        target = np.ma.masked_where(
            bevs[index, channels.index("target")] <= 0.0,
            bevs[index, channels.index("target")],
        )
        axis.imshow(
            target,
            cmap="Oranges",
            alpha=0.55,
            origin="upper",
            extent=(-right, left, -back, front),
            interpolation="nearest",
        )

    count = int(np.count_nonzero(data["masks"][index]))
    trajectory = np.asarray(data["trajs"][index, :count])
    state = np.asarray(data["states"][index])
    local = _to_local(trajectory, float(state[0]), float(state[1]), float(state[2]))
    axis.plot(local[:, 1], local[:, 0], color="#1f77b4", linewidth=2.0, label="Expert")
    axis.scatter([0.0], [0.0], color="#2ca02c", marker="o", label="Start")
    axis.set(
        xlabel="Left / m",
        ylabel="Forward / m",
        title=f"Sample {index}: BEV and expert trajectory",
    )
    axis.set_aspect("equal")
    axis.legend(loc="best")
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _metadata_counts(metadata: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(item.get(key, "unknown")) for item in metadata)
    return dict(sorted(counts.items()))


def _noise_counts(metadata: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(item.get("difficulty", {}).get("noise_level", "unknown"))
        for item in metadata
    )
    return dict(sorted(counts.items()))


def _to_local(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    local = np.empty_like(points)
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local[:, 0] = cos_yaw * dx + sin_yaw * dy
    local[:, 1] = -sin_yaw * dx + cos_yaw * dy
    local[:, 2] = points[:, 2] - yaw
    return local
