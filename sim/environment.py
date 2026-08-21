"""二维矿区泊车环境。

环境由若干轴对齐矩形障碍物与一块矩形地图边界构成，提供碰撞检测与模拟点云支持。
坐标系：车辆中心局部坐标最终由调用方决定；环境本身使用全局世界坐标。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces import GoalPose


@dataclass(frozen=True)
class RectangleObstacle:
    """轴对齐矩形障碍物，坐标为全局世界坐标（米）。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class ParkingEnvironment:
    """二维矿区泊车环境。

    world_size 为地图正方形边长（米），obstacles 为轴对齐矩形障碍物列表，
    parking_spots 为可用泊车位姿列表。
    """

    def __init__(
        self,
        world_size: float = 40.0,
        obstacles: list[RectangleObstacle] | None = None,
        parking_spots: list[GoalPose] | None = None,
    ) -> None:
        self.world_size = world_size
        half = world_size / 2.0
        self.boundary = RectangleObstacle(-half, half, -half, half)
        self.obstacles = obstacles if obstacles is not None else []
        self.parking_spots = parking_spots if parking_spots is not None else []

    def is_free(self, x: float, y: float) -> bool:
        """判断全局坐标点是否在地图内且不在任何障碍物内。"""
        b = self.boundary
        if x < b.x_min or x > b.x_max or y < b.y_min or y > b.y_max:
            return False
        for obs in self.obstacles:
            if obs.x_min <= x <= obs.x_max and obs.y_min <= y <= obs.y_max:
                return False
        return True

    def has_collision(self, xs: np.ndarray, ys: np.ndarray) -> bool:
        """批量判断点序列是否发生碰撞（越界或进入障碍物）。"""
        return not np.all([self.is_free(float(x), float(y)) for x, y in zip(xs, ys)])

    def raycast(self, origin: np.ndarray, angle: float, max_range: float) -> float:
        """从 origin 沿 angle 方向发射射线，返回遇到障碍物前的距离（米）。

        用于模拟 LiDAR。若射线先出地图边界或命中障碍物则返回该距离。
        """
        step = 0.05
        distance = 0.0
        direction = np.array([np.cos(angle), np.sin(angle)])
        while distance < max_range:
            distance += step
            probe = origin + direction * distance
            if not self.is_free(float(probe[0]), float(probe[1])):
                return distance
        return max_range