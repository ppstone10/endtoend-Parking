"""BEV 保真度退化曲线图（感知层随噪声档退化）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, save_fig, setup_style


def save_bev_fidelity_degradation(
    per_scene: dict[str, dict[str, dict]],
    output: str | Path,
) -> list[str]:
    """按场景×噪声档绘制 occupancy IoU 与 target 命中率退化曲线。

    per_scene[scene][noise_level] = {"occupancy_iou": .., "target_hit_rate": ..}
    """
    if not per_scene:
        raise ValueError("至少需要一个场景")
    scenes = sorted(per_scene)
    noise_levels = _noise_order(per_scene)
    if not noise_levels:
        raise ValueError("没有可绘制的噪声档")

    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
    positions = np.arange(len(noise_levels))
    for index, scene in enumerate(scenes):
        series = per_scene[scene]
        iou = [float(series[level]["occupancy_iou"]) for level in noise_levels]
        hit = [float(series[level]["target_hit_rate"]) for level in noise_levels]
        color = _scene_color(index)
        axes[0].plot(positions, iou, marker="o", label=scene, color=color)
        axes[1].plot(positions, hit, marker="s", label=scene, color=color, linestyle="--")
    for axis in axes:
        axis.set_xticks(positions, noise_levels)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Occupancy IoU")
    axes[0].set_title("BEV occupancy fidelity vs noise")
    axes[1].set_ylabel("Target hit rate")
    axes[1].set_title("BEV target fidelity vs noise")
    axes[0].legend(fontsize=8, ncol=2)
    written = save_fig(figure, str(output))
    plt.close(figure)
    return written


def _noise_order(per_scene: dict[str, dict[str, dict]]) -> list[str]:
    """返回稳定噪声档顺序（clean/low/high 优先）。"""
    known = ["clean", "low", "high"]
    seen: list[str] = []
    for scene in sorted(per_scene):
        for level in per_scene[scene]:
            if level not in seen:
                seen.append(level)
    return [level for level in known if level in seen] + [
        level for level in seen if level not in known
    ]


def _scene_color(index: int) -> str:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    ]
    return palette[index % len(palette)]


__all__ = ["save_bev_fidelity_degradation"]