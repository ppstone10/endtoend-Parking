"""履带钻机居中矩形外廓与连续扫掠碰撞检查。"""

from __future__ import annotations

import numpy as np

from sim import CircleObstacle, ParkingEnvironment, PolygonObstacle, RectangleObstacle


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

    def pose_free(self, x: float, y: float, yaw: float) -> bool:
        corners = self.rectangle_corners(x, y, yaw)
        half_world = self.env.world_size / 2.0
        if np.any(np.abs(corners[:, 0]) > half_world) or np.any(
            np.abs(corners[:, 1]) > half_world
        ):
            return False

        vehicle_bbox = (
            float(corners[:, 0].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].min()),
            float(corners[:, 1].max()),
        )
        for obstacle in self.env.obstacles:
            if not obstacle.forbidden or not self._bbox_overlaps(
                vehicle_bbox, obstacle.bbox
            ):
                continue
            if isinstance(obstacle, CircleObstacle):
                if self._circle_intersects_rectangle(obstacle, x, y, yaw):
                    return False
                continue
            if isinstance(obstacle, RectangleObstacle):
                obstacle_vertices = np.array(
                    [
                        [obstacle.x_min, obstacle.y_min],
                        [obstacle.x_max, obstacle.y_min],
                        [obstacle.x_max, obstacle.y_max],
                        [obstacle.x_min, obstacle.y_max],
                    ],
                    dtype=np.float64,
                )
            elif isinstance(obstacle, PolygonObstacle):
                obstacle_vertices = np.asarray(obstacle.vertices, dtype=np.float64)
            else:
                if any(
                    obstacle.contains_point(float(px), float(py))
                    for px, py in corners
                ):
                    return False
                continue
            if self._polygons_intersect(
                corners,
                obstacle_vertices,
                lambda px, py: obstacle.contains_point(float(px), float(py)),
                lambda px, py: self.point_in_rectangle(
                    float(px), float(py), x, y, yaw
                ),
            ):
                return False
        return True

    def swept_segment_free(self, start_pose: np.ndarray, end_pose: np.ndarray) -> bool:
        start = np.asarray(start_pose, dtype=np.float64)
        end = np.asarray(end_pose, dtype=np.float64)
        delta_xy = end[:2] - start[:2]
        delta_yaw = self._angle_delta(float(end[2]), float(start[2]))
        swept_distance = float(np.linalg.norm(delta_xy)) + self.corner_radius * abs(
            delta_yaw
        )
        steps = max(1, int(np.ceil(swept_distance / self.resolution)))
        for index in range(steps + 1):
            fraction = index / steps
            x, y = start[:2] + fraction * delta_xy
            yaw = self._norm_angle(float(start[2]) + fraction * delta_yaw)
            if not self.pose_free(float(x), float(y), yaw):
                return False
        return True

    def rectangle_corners(self, x: float, y: float, yaw: float) -> np.ndarray:
        local = np.array(
            [
                [self.half_length, self.half_width],
                [self.half_length, -self.half_width],
                [-self.half_length, -self.half_width],
                [-self.half_length, self.half_width],
            ],
            dtype=np.float64,
        )
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float64,
        )
        return local @ rotation.T + np.array([x, y], dtype=np.float64)

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

    def _circle_intersects_rectangle(
        self, obstacle: CircleObstacle, x: float, y: float, yaw: float
    ) -> bool:
        dx, dy = obstacle.x - x, obstacle.y - y
        local_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        closest_x = float(np.clip(local_x, -self.half_length, self.half_length))
        closest_y = float(np.clip(local_y, -self.half_width, self.half_width))
        return (
            (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2
            <= obstacle.radius**2
        )

    @staticmethod
    def _bbox_overlaps(first, second) -> bool:
        return not (
            first[1] < second[0]
            or second[1] < first[0]
            or first[3] < second[2]
            or second[3] < first[2]
        )

    @classmethod
    def _polygons_intersect(
        cls,
        vehicle: np.ndarray,
        obstacle: np.ndarray,
        vehicle_point_in_obstacle,
        obstacle_point_in_vehicle,
    ) -> bool:
        if any(vehicle_point_in_obstacle(*point) for point in vehicle):
            return True
        if any(obstacle_point_in_vehicle(*point) for point in obstacle):
            return True
        for index in range(len(vehicle)):
            a1, a2 = vehicle[index], vehicle[(index + 1) % len(vehicle)]
            for other_index in range(len(obstacle)):
                b1 = obstacle[other_index]
                b2 = obstacle[(other_index + 1) % len(obstacle)]
                if cls._segments_intersect(a1, a2, b1, b2):
                    return True
        return False

    @staticmethod
    def _segments_intersect(a1, a2, b1, b2) -> bool:
        def orientation(p, q, r):
            return float(
                (q[0] - p[0]) * (r[1] - p[1])
                - (q[1] - p[1]) * (r[0] - p[0])
            )

        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
            )

        o1 = orientation(a1, a2, b1)
        o2 = orientation(a1, a2, b2)
        o3 = orientation(b1, b2, a1)
        o4 = orientation(b1, b2, a2)
        if o1 * o2 < 0.0 and o3 * o4 < 0.0:
            return True
        return (
            (abs(o1) <= 1e-9 and on_segment(a1, b1, a2))
            or (abs(o2) <= 1e-9 and on_segment(a1, b2, a2))
            or (abs(o3) <= 1e-9 and on_segment(b1, a1, b2))
            or (abs(o4) <= 1e-9 and on_segment(b1, a2, b2))
        )

    @staticmethod
    def _norm_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _angle_delta(cls, target: float, source: float) -> float:
        return cls._norm_angle(target - source)


__all__ = ["RectangleFootprintCollisionChecker"]
