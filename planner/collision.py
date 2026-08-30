"""履带钻机居中矩形外廓与连续扫掠碰撞检查。"""

from __future__ import annotations

import math

import numpy as np

from interfaces import Trajectory, VehicleState
from sim import ParkingEnvironment
from sim.footprint import rectangle_pose_is_free


class RectangleFootprintCollisionChecker:
    """检查完整定向矩形及相邻中心位姿之间的连续扫掠。"""

    def __init__(
        self,
        env: ParkingEnvironment,
        *,
        vehicle_length: float,
        vehicle_width: float,
        collision_margin: float,
        resolution: float,
    ) -> None:
        self.env = env
        self.half_length = vehicle_length / 2.0 + collision_margin
        self.half_width = vehicle_width / 2.0 + collision_margin
        self.resolution = resolution
        self.corner_radius = float(np.hypot(self.half_length, self.half_width))
        self._local_corners = np.array(
            [
                [self.half_length, self.half_width],
                [self.half_length, -self.half_width],
                [-self.half_length, -self.half_width],
                [-self.half_length, self.half_width],
            ],
            dtype=np.float64,
        )

    def pose_free(self, x: float, y: float, yaw: float) -> bool:
        return rectangle_pose_is_free(
            self.env,
            x,
            y,
            yaw,
            half_length=self.half_length,
            half_width=self.half_width,
        )

    def swept_segment_free(self, start_pose: np.ndarray, end_pose: np.ndarray) -> bool:
        start_x, start_y, start_yaw = (float(value) for value in start_pose)
        end_x, end_y, end_yaw = (float(value) for value in end_pose)
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        delta_yaw = self._angle_delta(end_yaw, start_yaw)
        swept_distance = math.hypot(delta_x, delta_y) + self.corner_radius * abs(delta_yaw)
        steps = max(1, int(np.ceil(swept_distance / self.resolution)))
        for index in range(steps + 1):
            fraction = index / steps
            x = start_x + fraction * delta_x
            y = start_y + fraction * delta_y
            yaw = self._norm_angle(start_yaw + fraction * delta_yaw)
            if not self.pose_free(x, y, yaw):
                return False
        return True

    def check_trajectory(
        self, state: VehicleState, trajectory: Trajectory
    ) -> tuple[bool, str | None]:
        """从当前状态开始审查全部位姿和相邻点连续扫掠。"""
        points = np.asarray(trajectory.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return False, "empty_trajectory"
        if not np.isfinite(points).all():
            return False, "nonfinite_trajectory"
        previous = np.asarray([state.x, state.y, state.yaw], dtype=np.float64)
        if not self.pose_free(*previous):
            return False, "current_pose_collision"
        for point in points:
            if not self.swept_segment_free(previous, point):
                return False, "swept_collision"
            previous = point
        return True, None

    def rectangle_corners(self, x: float, y: float, yaw: float) -> np.ndarray:
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float64,
        )
        return self._local_corners @ rotation.T + np.array([x, y], dtype=np.float64)

    def point_in_rectangle(
        self, px: float, py: float, x: float, y: float, yaw: float
    ) -> bool:
        dx, dy = px - x, py - y
        local_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        return (
            abs(local_x) <= self.half_length + 1e-9
            and abs(local_y) <= self.half_width + 1e-9
        )

    @staticmethod
    def _norm_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _angle_delta(cls, target: float, source: float) -> float:
        return cls._norm_angle(target - source)


__all__ = ["RectangleFootprintCollisionChecker"]
