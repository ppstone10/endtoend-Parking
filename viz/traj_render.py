"""轨迹渲染：专家/规划/实际三线叠加与单回合总图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from interfaces import GoalPose
from runtime.recorder import EpisodeRecord
from .style import COLORS, LINESTYLES, setup_style
from .world_render import draw_goal, draw_vehicle, render_world


def draw_trajectory(
    ax,
    points: np.ndarray,
    kind: str = "actual",
    label: str | None = None,
    with_markers: bool = False,
    linewidth: float = 1.8,
) -> None:
    """渲染一条 (N,3) 轨迹。

    kind ∈ {expert, plan, actual} 决定颜色与线型；with_markers 为真时
    在轨迹点上叠加圆形标记（论文图中的"规划轨迹点"离散点效果）。
    """
    if points is None or len(points) == 0:
        return
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 1:
        ax.plot(pts[:, 0], pts[:, 1], "o", color=COLORS[kind], label=label)
        return
    ax.plot(pts[:, 0], pts[:, 1], LINESTYLES[kind], color=COLORS[kind], linewidth=linewidth, label=label)
    if with_markers:
        ax.plot(pts[:, 0], pts[:, 1], "o", color=COLORS[kind], markersize=2.5, alpha=0.7)


def render_episode(
    record: EpisodeRecord,
    env,
    goal: GoalPose,
    vehicle_length: float,
    vehicle_width: float,
    title: str = "",
    out_path: str | None = None,
) -> plt.Figure:
    """渲染单回合总图：世界 + 最终规划轨迹 + 实际轨迹 + 起终点 + 末态车辆。

    out_path 提供时保存 PNG+PDF（不带扩展名）。
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    render_world(ax, env)
    draw_goal(ax, goal)

    if record.states:
        start = record.states[0]
        ax.plot(start.x, start.y, "ks", markersize=8, label="Start")
    if record.plans:
        draw_trajectory(ax, record.plans[-1], kind="plan", label="Planned trajectory", with_markers=True)
    if record.states:
        actual = np.array([[s.x, s.y, s.yaw] for s in record.states])
        draw_trajectory(ax, actual, kind="actual", label="Actual trajectory")
        draw_vehicle(ax, record.states[-1], vehicle_length, vehicle_width, label="Final vehicle")

    if record.collisions and record.collisions[-1]:
        title = f"{title} [COLLISION]".strip()

    ax.set_title(title or "Closed-loop parking episode")
    ax.legend(loc="upper right")
    if out_path:
        from .style import save_fig

        save_fig(fig, out_path)
    return fig
