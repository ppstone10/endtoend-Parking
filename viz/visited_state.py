"""L2 访问状态轨迹质量图。

- 左图：网络 vs 专家 在访问状态上的 ADE/FDE/航向 MAE（随时间/距离退化）；
- 右图：近端（<3m）网络预测长度、终点距目标、终点航向误差、方向切换。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, save_fig, setup_style


def save_visited_state_report(report: dict, output: str | Path) -> list[str]:
    """根据 L2 报告绘制退化曲线，返回 PNG+PDF 路径。"""
    replans = report.get("replans", [])
    if not replans:
        raise ValueError("报告没有逐重规划记录，无法绘图")
    near_threshold = float(report.get("near_threshold_m", 3.0))

    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    # 左图：访问状态对误差 vs 距目标距离。
    xs = []
    ade = []
    fde = []
    for row in replans:
        if "vs_ade_m" not in row:
            continue
        xs.append(float(row["d_goal_m"]))
        ade.append(float(row["vs_ade_m"]))
        fde.append(float(row["vs_fde_m"]))
    if xs:
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        axes[0].plot(xs, np.asarray(ade)[order], marker="o", ms=3,
                     color=COLORS["actual"], label="ADE")
        axes[0].plot(xs, np.asarray(fde)[order], marker="s", ms=3,
                     color=COLORS["plan"], label="FDE")
        axes[0].axvline(near_threshold, color="gray", linestyle="--", alpha=0.6,
                        label=f"near {near_threshold}m")
        axes[0].set_ylabel("Error vs expert (m)")
        axes[0].set_xlabel("Distance to goal (m)")
        axes[0].set_title("Visited-state open-loop error")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

    # 右图：近端网络预测终点质量。
    near_rows = [row for row in replans if float(row.get("d_goal_m", 99.0)) < near_threshold]
    if near_rows:
        end_goal = [abs(float(row.get("net_end_to_goal_m", 0.0))) for row in near_rows]
        yaw_err = [abs(float(row.get("net_end_yaw_err_deg", 0.0))) for row in near_rows]
        lengths = [float(row.get("net_length_m", 0.0)) for row in near_rows]
        xs_near = [float(row["d_goal_m"]) for row in near_rows]
        order = np.argsort(xs_near)
        xs_near = np.asarray(xs_near)[order]
        axes[1].plot(xs_near, np.asarray(end_goal)[order], marker="o", ms=3,
                     color=COLORS["actual"], label="end->goal (m)")
        axes[1].plot(xs_near, np.asarray(lengths)[order], marker="s", ms=3,
                     color=COLORS["plan"], label="net length (m)")
        axes[1].plot(xs_near, np.asarray(yaw_err)[order], marker="^", ms=3,
                     color=COLORS["spot"], label="end yaw err (deg)")
        axes[1].set_xlabel("Distance to goal (m)")
        axes[1].set_title("Near-goal network trajectory quality")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    written = save_fig(figure, str(output))
    plt.close(figure)
    return written


__all__ = ["save_visited_state_report"]