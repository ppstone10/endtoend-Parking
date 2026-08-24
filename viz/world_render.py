"""世界俯视图渲染：障碍、边界、车位、车辆矩形。

当前支持 RectangleObstacle；多边形障碍（M2 障碍体系）扩展时在本模块
增加分支，调用方接口不变。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle

from interfaces import GoalPose, VehicleState
from .style import COLORS


def render_world(ax, env, spots: list[GoalPose] | None = None, spot_size: tuple[float, float] = (7.0, 3.5)) -> None:
    """在 ax 上渲染环境：边界、障碍物、车位框。

    spots 为目标车位位姿列表（含单个目标），spot_size 为车位框尺寸
    (length, width)。
    """
    half = env.world_size / 2.0
    ax.plot(
        [-half, half, half, -half, -half],
        [-half, -half, half, half, -half],
        color="black",
        linewidth=1.0,
        label="Boundary" if not getattr(ax, "_world_legend", False) else None,
    )
    ax._world_legend = True

    for i, obs in enumerate(env.obstacles):
        ax.add_patch(
            Rectangle(
                (obs.x_min, obs.y_min),
                obs.x_max - obs.x_min,
                obs.y_max - obs.y_min,
                facecolor=COLORS["obstacle"],
                alpha=0.6,
                edgecolor="black",
                linewidth=0.5,
            )
        )
    if env.obstacles:
        ax.add_patch(
            Rectangle(
                (env.obstacles[0].x_min, env.obstacles[0].y_min),
                0.0,
                0.0,
                facecolor=COLORS["obstacle"],
                alpha=0.6,
                label="Obstacle",
            )
        )

    if spots:
        l, w = spot_size
        for spot in spots:
            ax.add_patch(
                Rectangle(
                    (spot.x - l / 2.0, spot.y - w / 2.0),
                    l,
                    w,
                    fill=False,
                    edgecolor=COLORS["spot"],
                    linewidth=1.5,
                    linestyle="--",
                )
            )
        ax.plot([], [], color=COLORS["spot"], linestyle="--", linewidth=1.5, label="Parking spot")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def draw_vehicle(ax, state: VehicleState, length: float, width: float, label: str | None = None, alpha: float = 1.0) -> None:
    """渲染车辆矩形与朝向箭头。"""
    cos_yaw, sin_yaw = np.cos(state.yaw), np.sin(state.yaw)
    corners_local = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]]
    )
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    corners = corners_local @ rot.T + np.array([state.x, state.y])
    ax.add_patch(
        plt.Polygon(corners, closed=True, facecolor=COLORS["vehicle"], alpha=0.35 * alpha, edgecolor=COLORS["vehicle"], linewidth=1.2)
    )
    arrow_len = length * 0.5
    ax.add_patch(
        FancyArrow(
            state.x,
            state.y,
            arrow_len * cos_yaw,
            arrow_len * sin_yaw,
            width=0.05,
            head_width=0.4,
            head_length=0.5,
            color=COLORS["vehicle"],
            alpha=alpha,
        )
    )
    if label:
        ax.plot([], [], color=COLORS["vehicle"], linewidth=1.2, label=label)


def draw_goal(ax, goal: GoalPose, tol_pos: float = 0.3) -> None:
    """渲染目标位姿（箭头）与容差圆。"""
    ax.add_patch(plt.Circle((goal.x, goal.y), tol_pos, fill=False, color=COLORS["spot"], linestyle=":", linewidth=1.0))
    ax.add_patch(
        FancyArrow(
            goal.x,
            goal.y,
            1.5 * np.cos(goal.yaw),
            1.5 * np.sin(goal.yaw),
            width=0.05,
            head_width=0.4,
            head_length=0.5,
            color=COLORS["spot"],
        )
    )
    ax.plot([], [], color=COLORS["spot"], linewidth=1.2, label="Goal pose")
