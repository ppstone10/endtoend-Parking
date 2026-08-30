"""分组开环指标与预测—专家局部轨迹叠加图。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from training.data import to_local

from .style import COLORS, save_fig, setup_style


def save_grouped_metrics(
    groups: dict[str, dict[str, dict[str, Any]]], output: str | Path
) -> list[str]:
    """绘制任务、场景、方向和噪声的 ADE/FDE 对比。"""
    setup_style()
    dimensions = (
        ("task_type", "Task type"),
        ("scene", "Scene"),
        ("maneuver", "Requested maneuver"),
        ("noise_level", "Sensor noise"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)
    for axis, (dimension, title) in zip(axes.flat, dimensions):
        entries = groups.get(dimension, {})
        labels = list(entries)
        positions = np.arange(len(labels))
        height = 0.36
        ade = [float(entries[label]["ade_m"]) for label in labels]
        fde = [float(entries[label]["fde_m"]) for label in labels]
        axis.barh(positions - height / 2, ade, height, color=COLORS["actual"], label="ADE")
        axis.barh(positions + height / 2, fde, height, color=COLORS["plan"], label="FDE")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Position error (m)")
        axis.set_title(title)
        axis.legend()
    written = save_fig(figure, str(output))
    plt.close(figure)
    return written


def save_prediction_overlays(
    data: dict[str, Any],
    predictions: np.ndarray,
    sample_metrics: list[dict[str, Any]],
    indices: list[int],
    output: str | Path,
    *,
    title: str,
) -> list[str]:
    """在 BEV 上叠加选定样本的自由滚动预测与专家轨迹。"""
    if not indices:
        raise ValueError("至少需要一个叠加样本")
    bev_meta = data.get("bev_meta")
    metadata = data.get("task_meta")
    if not isinstance(bev_meta, dict) or not isinstance(metadata, list):
        raise ValueError("预测叠加需要 schema v2 的 bev_meta/task_meta")
    channels = list(bev_meta["channels"])
    front, back, left, right = (float(value) for value in bev_meta["extent"])
    columns = min(3, len(indices))
    rows = math.ceil(len(indices) / columns)
    setup_style()
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.1 * columns, 4.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(title)
    for axis, index in zip(axes.flat, indices):
        state = np.asarray(data["states"][index], dtype=np.float64)
        length = int(np.count_nonzero(data["masks"][index]))
        expert_world = np.asarray(data["trajs"][index, :length], dtype=np.float64)
        expert = to_local(expert_world, float(state[0]), float(state[1]), float(state[2]))
        goal_world = np.asarray(data["goals"][index], dtype=np.float64)
        goal = to_local(goal_world[None, :], float(state[0]), float(state[1]), float(state[2]))[0]
        metric = sample_metrics[index]
        predicted_length = metric.get("predicted_length")
        display_length = int(predicted_length) if predicted_length is not None else length
        predicted = np.asarray(predictions[index, :display_length], dtype=np.float64)
        occupancy = np.asarray(data["bevs"][index, channels.index("occupancy")])
        axis.imshow(
            occupancy,
            cmap="Greys",
            origin="upper",
            extent=(-right, left, -back, front),
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        if "target" in channels:
            target = np.ma.masked_where(
                data["bevs"][index, channels.index("target")] <= 0.0,
                data["bevs"][index, channels.index("target")],
            )
            axis.imshow(
                target,
                cmap="Oranges",
                alpha=0.55,
                origin="upper",
                extent=(-right, left, -back, front),
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
        axis.plot(expert[:, 1], expert[:, 0], "--", color=COLORS["expert"], linewidth=2.2, label="Expert")
        axis.plot(predicted[:, 1], predicted[:, 0], "-", color=COLORS["plan"], linewidth=2.0, label="Prediction")
        axis.scatter([0.0], [0.0], marker="s", color=COLORS["actual"], s=45, label="Start")
        axis.scatter([goal[1]], [goal[0]], marker="*", color=COLORS["spot"], s=90, label="Goal")
        axis.scatter([predicted[-1, 1]], [predicted[-1, 0]], marker="x", color=COLORS["plan"], s=55)
        axis.set_xlim(-right, left)
        axis.set_ylim(-back, front)
        axis.invert_xaxis()
        axis.set_aspect("equal")
        axis.set_xlabel("Left (m, positive shown left)")
        axis.set_ylabel("Forward (m)")
        item = metadata[index]
        yaw_deg = np.degrees(float(metric["yaw_mae_rad"]))
        axis.set_title(
            f"#{index} {item.get('scene_name', 'unknown')} / {item.get('task_type', 'unknown')}\n"
            f"ADE {metric['ade_m']:.2f} m · FDE {metric['fde_m']:.2f} m · yaw {yaw_deg:.1f}°\n"
            f"target {length} pts · predicted stop {display_length} pts"
        )
        axis.legend(loc="upper right", fontsize=7)
    for axis in axes.flat[len(indices):]:
        axis.set_visible(False)
    written = save_fig(figure, str(output))
    plt.close(figure)
    return written
