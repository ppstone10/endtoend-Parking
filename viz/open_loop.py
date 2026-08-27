"""开环模型指标对比图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, save_fig, setup_style


def save_open_loop_comparison(results: dict[str, dict], output: str | Path) -> list[str]:
    """将多个模型的 ADE/FDE/航向 MAE 保存为 PNG+PDF。"""
    if not results:
        raise ValueError("至少需要一个模型结果")
    labels = list(results)
    ade = [float(results[label]["ade_m"]) for label in labels]
    fde = [float(results[label]["fde_m"]) for label in labels]
    yaw_degrees = [np.degrees(float(results[label]["yaw_mae_rad"])) for label in labels]
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    positions = np.arange(len(labels))
    width = 0.36
    axes[0].bar(positions - width / 2, ade, width, label="ADE", color=COLORS["actual"])
    axes[0].bar(positions + width / 2, fde, width, label="FDE", color=COLORS["plan"])
    axes[0].set_ylabel("Position error (m)")
    axes[0].set_title("Open-loop position error")
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].legend()
    axes[1].bar(positions, yaw_degrees, color=COLORS["expert"])
    axes[1].set_ylabel("Yaw MAE (deg)")
    axes[1].set_title("Open-loop heading error")
    axes[1].set_xticks(positions, labels, rotation=20, ha="right")
    written = save_fig(figure, str(output))
    plt.close(figure)
    return written
