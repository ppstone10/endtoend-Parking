"""世界俯视图渲染：障碍（矩形/多边形/圆）、边界、车位、车辆、悬崖纹理。

渲染按障碍 kind 分样式：cliff 斜纹填充、berm 深色条带、rock 圆斑、
vehicle 半透明、line 虚线框、wall/其余常规灰色。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

from interfaces import GoalPose, VehicleState
from sim.spots import ParkingSpot
from .style import COLORS

# kind → 渲染样式
_KIND_STYLE = {
    "cliff": {"facecolor": "#4a4a4a", "alpha": 0.9, "hatch": "xx", "edgecolor": "black"},
    "berm": {"facecolor": "#8b4513", "alpha": 0.95, "edgecolor": "black"},
    "rock": {"facecolor": "#a9a9a9", "alpha": 0.8, "edgecolor": "dimgray"},
    "vehicle": {"facecolor": "#4682b4", "alpha": 0.5, "edgecolor": "steelblue"},
    "equipment": {"facecolor": "#b8860b", "alpha": 0.7, "edgecolor": "darkgoldenrod"},
    "line": {"facecolor": "none", "alpha": 1.0, "edgecolor": "#ffcc00", "linestyle": "--"},
    "wall": {"facecolor": COLORS["obstacle"], "alpha": 0.6, "edgecolor": "black"},
}


def _draw_obstacle(ax, obs) -> None:
    style = _KIND_STYLE.get(obs.kind, _KIND_STYLE["wall"])
    if hasattr(obs, "vertices"):
        ax.add_patch(Polygon(np.array(obs.vertices), closed=True, linewidth=0.8, **style))
    elif hasattr(obs, "radius"):
        ax.add_patch(Circle((obs.x, obs.y), obs.radius, linewidth=0.8, **style))
    else:
        ax.add_patch(
            Rectangle(
                (obs.x_min, obs.y_min),
                obs.x_max - obs.x_min,
                obs.y_max - obs.y_min,
                linewidth=0.8,
                **style,
            )
        )


def render_world(
    ax,
    env,
    spots: list[GoalPose] | list[ParkingSpot] | None = None,
    spot_size: tuple[float, float] = (7.0, 3.5),
) -> None:
    """在 ax 上渲染环境：边界、障碍物（按 kind 样式）、车位框。

    spots 为 ParkingSpot 列表（优先，含占用状态与编号）或 GoalPose 列表；
    spot_size 仅对 GoalPose 列表生效。
    """
    half = env.world_size / 2.0
    ax.plot(
        [-half, half, half, -half, -half],
        [-half, -half, half, half, -half],
        color="black",
        linewidth=1.0,
    )

    for obs in env.obstacles:
        _draw_obstacle(ax, obs)

    if spots:
        for spot in spots:
            if isinstance(spot, ParkingSpot):
                l, w = spot.size
                pose = spot.pose
                face = "#cccccc" if not spot.occupied else "#999999"
                ax.add_patch(
                    Polygon(
                        spot.footprint_corners(), closed=True,
                        facecolor=face, edgecolor=COLORS["spot"],
                        linewidth=1.2, linestyle="--", alpha=0.6,
                    )
                )
                ax.text(pose.x, pose.y, spot.id, ha="center", va="center", fontsize=7, color="black")
            else:
                pose = spot
                l, w = spot_size
                ax.add_patch(
                    Rectangle(
                        (pose.x - l / 2.0, pose.y - w / 2.0), l, w,
                        fill=False, edgecolor=COLORS["spot"], linewidth=1.5, linestyle="--",
                    )
                )

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
