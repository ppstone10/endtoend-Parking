"""二维矿区泊车环境。

环境由障碍物列表与一块矩形地图边界构成，提供碰撞检测与解析射线投射。
坐标系：环境使用全局世界坐标；车辆中心局部坐标由调用方决定。

碰撞与点云语义分离（见 sim/obstacles.py）：
- is_free / has_collision 只检查 forbidden 障碍与地图边界；
- raycast 只与 emits_points 障碍及地图边界求交（悬崖不挡射线）。
"""

from __future__ import annotations

import math

import numpy as np

from interfaces import GoalPose
from .obstacles import Obstacle, RectangleObstacle

__all__ = ["ParkingEnvironment", "Obstacle", "RectangleObstacle"]


class ParkingEnvironment:
    """二维矿区泊车环境。

    world_size 为地图正方形边长（米），obstacles 为 Obstacle 列表
    （RectangleObstacle/PolygonObstacle/CircleObstacle），parking_spots
    为可用泊车位姿列表。
    """

    def __init__(
        self,
        world_size: float = 40.0,
        obstacles: list[Obstacle] | None = None,
        parking_spots: list[GoalPose] | None = None,
    ) -> None:
        self.world_size = world_size
        half = world_size / 2.0
        self.boundary = RectangleObstacle(-half, half, -half, half)
        self.obstacles = list(obstacles) if obstacles is not None else []
        self.parking_spots = parking_spots if parking_spots is not None else []

    def is_free(self, x: float, y: float) -> bool:
        """判断全局坐标点是否在地图内且不在任何 forbidden 障碍物内。"""
        b = self.boundary
        if x < b.x_min or x > b.x_max or y < b.y_min or y > b.y_max:
            return False
        for obs in self.obstacles:
            if obs.forbidden and obs.contains_point(x, y):
                return False
        return True

    def has_collision(self, xs: np.ndarray, ys: np.ndarray) -> bool:
        """批量判断点序列是否发生碰撞（越界或进入 forbidden 障碍物）。"""
        return not np.all([self.is_free(float(x), float(y)) for x, y in zip(xs, ys)])

    def raycast(self, origin: np.ndarray, angle: float, max_range: float) -> float:
        """解析射线投射：返回射线遇到（emits_points）障碍物或出界的距离。

        与障碍物边界/地图边界的求交均为解析解，无步进量化误差；
        射线穿过非 emits_points 障碍（如悬崖）。用于模拟 LiDAR。
        """
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = math.cos(angle), math.sin(angle)
        best = self._world_exit_distance(ox, oy, dx, dy)
        if best <= 0.0:
            return 0.0  # 起点在地图外
        for obs in self.obstacles:
            if not obs.emits_points:
                continue
            if obs.contains_point(ox, oy):
                return 0.0  # 起点在障碍物内
            t = obs.ray_entry_distance(ox, oy, dx, dy)
            if t is not None and t < best:
                best = t
        if best < max_range:
            return float(best)
        return float(max_range)

    def _world_exit_distance(self, ox: float, oy: float, dx: float, dy: float) -> float:
        """射线离开正方形地图的距离（射线与地图边界求交）。"""
        half = self.world_size / 2.0
        t_exit = math.inf
        if dx > 1e-12:
            t_exit = min(t_exit, (half - ox) / dx)
        elif dx < -1e-12:
            t_exit = min(t_exit, (-half - ox) / dx)
        if dy > 1e-12:
            t_exit = min(t_exit, (half - oy) / dy)
        elif dy < -1e-12:
            t_exit = min(t_exit, (-half - oy) / dy)
        return t_exit
