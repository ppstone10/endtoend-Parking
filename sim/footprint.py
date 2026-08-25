"""仿真、任务采样与规划共用的定向矩形位姿碰撞判定。"""

from __future__ import annotations

import math

import numpy as np

from .environment import ParkingEnvironment
from .obstacles import CircleObstacle, PolygonObstacle, RectangleObstacle


def rectangle_pose_is_free(
    env: ParkingEnvironment,
    x: float,
    y: float,
    yaw: float,
    *,
    half_length: float,
    half_width: float,
) -> bool:
    """精确检查有向矩形是否在地图内且不与 forbidden 障碍相交。"""
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    extent_x = abs(cos_yaw) * half_length + abs(sin_yaw) * half_width
    extent_y = abs(sin_yaw) * half_length + abs(cos_yaw) * half_width
    half_world = env.world_size / 2.0
    if (
        x - extent_x < -half_world
        or x + extent_x > half_world
        or y - extent_y < -half_world
        or y + extent_y > half_world
    ):
        return False

    vehicle_bbox = (x - extent_x, x + extent_x, y - extent_y, y + extent_y)
    corners: np.ndarray | None = None
    for obstacle in env.obstacles:
        if not obstacle.forbidden or not _bbox_overlaps(vehicle_bbox, obstacle.bbox):
            continue
        if isinstance(obstacle, CircleObstacle):
            if _circle_intersects_rectangle(
                obstacle, x, y, cos_yaw, sin_yaw, half_length, half_width
            ):
                return False
            continue
        if isinstance(obstacle, RectangleObstacle):
            if _axis_aligned_rectangle_intersects(
                obstacle,
                x,
                y,
                cos_yaw,
                sin_yaw,
                half_length,
                half_width,
                extent_x,
                extent_y,
            ):
                return False
            continue
        if isinstance(obstacle, PolygonObstacle):
            obstacle_vertices = np.asarray(obstacle.vertices, dtype=np.float64)
        else:
            if corners is None:
                corners = rectangle_corners(x, y, yaw, half_length, half_width)
            if any(
                obstacle.contains_point(float(px), float(py)) for px, py in corners
            ):
                return False
            continue
        if corners is None:
            corners = rectangle_corners(x, y, yaw, half_length, half_width)
        if _polygons_intersect(
            corners,
            obstacle_vertices,
            lambda px, py: obstacle.contains_point(float(px), float(py)),
            lambda px, py: point_in_rectangle(
                float(px), float(py), x, y, cos_yaw, sin_yaw, half_length, half_width
            ),
        ):
            return False
    return True


def rectangle_corners(
    x: float, y: float, yaw: float, half_length: float, half_width: float
) -> np.ndarray:
    local = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64
    )
    return local @ rotation.T + np.asarray([x, y], dtype=np.float64)


def point_in_rectangle(
    px: float,
    py: float,
    x: float,
    y: float,
    cos_yaw: float,
    sin_yaw: float,
    half_length: float,
    half_width: float,
) -> bool:
    dx, dy = px - x, py - y
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    return (
        abs(local_x) <= half_length + 1e-9
        and abs(local_y) <= half_width + 1e-9
    )


def _circle_intersects_rectangle(
    obstacle: CircleObstacle,
    x: float,
    y: float,
    cos_yaw: float,
    sin_yaw: float,
    half_length: float,
    half_width: float,
) -> bool:
    dx, dy = obstacle.x - x, obstacle.y - y
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    closest_x = min(max(local_x, -half_length), half_length)
    closest_y = min(max(local_y, -half_width), half_width)
    return (
        (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2
        <= obstacle.radius**2
    )


def _axis_aligned_rectangle_intersects(
    obstacle: RectangleObstacle,
    x: float,
    y: float,
    cos_yaw: float,
    sin_yaw: float,
    half_length: float,
    half_width: float,
    extent_x: float,
    extent_y: float,
) -> bool:
    obstacle_x = (obstacle.x_min + obstacle.x_max) / 2.0
    obstacle_y = (obstacle.y_min + obstacle.y_max) / 2.0
    obstacle_half_x = (obstacle.x_max - obstacle.x_min) / 2.0
    obstacle_half_y = (obstacle.y_max - obstacle.y_min) / 2.0
    dx, dy = obstacle_x - x, obstacle_y - y
    if abs(dx) > extent_x + obstacle_half_x + 1e-9:
        return False
    if abs(dy) > extent_y + obstacle_half_y + 1e-9:
        return False
    projected_u = abs(dx * cos_yaw + dy * sin_yaw)
    obstacle_on_u = obstacle_half_x * abs(cos_yaw) + obstacle_half_y * abs(sin_yaw)
    if projected_u > half_length + obstacle_on_u + 1e-9:
        return False
    projected_v = abs(-dx * sin_yaw + dy * cos_yaw)
    obstacle_on_v = obstacle_half_x * abs(sin_yaw) + obstacle_half_y * abs(cos_yaw)
    return projected_v <= half_width + obstacle_on_v + 1e-9


def _bbox_overlaps(first, second) -> bool:
    return not (
        first[1] < second[0]
        or second[1] < first[0]
        or first[3] < second[2]
        or second[3] < first[2]
    )


def _polygons_intersect(
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
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


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


__all__ = ["rectangle_pose_is_free"]
