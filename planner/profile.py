"""前进/倒车分限速的梯形速度剖面。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .smoothing import infer_segment_directions


@dataclass(frozen=True)
class VelocityProfile:
    """与轨迹点等长的带符号速度和累计时间。"""

    speeds: np.ndarray
    times: np.ndarray

    def __post_init__(self) -> None:
        speeds = np.asarray(self.speeds, dtype=float)
        times = np.asarray(self.times, dtype=float)
        if speeds.ndim != 1 or times.shape != speeds.shape:
            raise ValueError("speeds 与 times 必须为等长一维数组")
        if not np.all(np.isfinite(speeds)) or not np.all(np.isfinite(times)):
            raise ValueError("速度剖面不能含非有限数")
        object.__setattr__(self, "speeds", speeds)
        object.__setattr__(self, "times", times)


def trapezoidal_velocity_profile(
    points: np.ndarray,
    *,
    max_speed: float = 1.5,
    reverse_speed: float = 0.75,
    acceleration: float = 0.8,
    deceleration: float = 1.0,
    max_omega: float = 0.35,
    directions: np.ndarray | None = None,
) -> VelocityProfile:
    """构造含换向停车和原地旋转耗时的梯形线速度剖面。"""
    poses = np.asarray(points, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2:
        raise ValueError("points 必须是至少两点的 (N, 3) 数组")
    if not np.all(np.isfinite(poses)):
        raise ValueError("points 不能含非有限数")
    limits = (max_speed, reverse_speed, acceleration, deceleration, max_omega)
    if any(value <= 0.0 or not np.isfinite(value) for value in limits):
        raise ValueError("速度与加减速限制必须为有限正数")
    if reverse_speed > max_speed:
        raise ValueError("reverse_speed 不能高于 max_speed")

    segment_directions = (
        infer_segment_directions(poses)
        if directions is None
        else _validate_directions(directions, len(poses) - 1)
    )
    ds = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    dyaw = np.abs(
        np.arctan2(
            np.sin(np.diff(poses[:, 2])),
            np.cos(np.diff(poses[:, 2])),
        )
    )
    pivot_segments = (ds <= 1e-12) & (dyaw > 1e-12)
    pivot_points = np.unique(
        np.concatenate(
            (np.flatnonzero(pivot_segments), np.flatnonzero(pivot_segments) + 1)
        )
    )
    point_directions = np.empty(len(poses), dtype=np.int8)
    point_directions[0] = segment_directions[0]
    point_directions[1:] = segment_directions
    cusp_indices = np.flatnonzero(segment_directions[1:] != segment_directions[:-1]) + 1

    point_limits = np.where(point_directions > 0, max_speed, reverse_speed).astype(float)
    point_limits[0] = 0.0
    point_limits[-1] = 0.0
    point_limits[cusp_indices] = 0.0
    point_limits[pivot_points] = 0.0

    magnitudes = point_limits.copy()
    for index in range(1, len(magnitudes)):
        reachable = np.sqrt(max(0.0, magnitudes[index - 1] ** 2 + 2.0 * acceleration * ds[index - 1]))
        magnitudes[index] = min(magnitudes[index], reachable)
    for index in range(len(magnitudes) - 2, -1, -1):
        stoppable = np.sqrt(max(0.0, magnitudes[index + 1] ** 2 + 2.0 * deceleration * ds[index]))
        magnitudes[index] = min(magnitudes[index], stoppable)

    speeds = magnitudes * point_directions
    speeds[cusp_indices] = 0.0
    speeds[pivot_points] = 0.0
    speeds[0] = 0.0
    speeds[-1] = 0.0
    times = np.zeros(len(poses), dtype=float)
    for index, distance in enumerate(ds):
        boundary_speed = magnitudes[index] + magnitudes[index + 1]
        if pivot_segments[index]:
            delta_t = dyaw[index] / max_omega
        elif distance <= 1e-12:
            delta_t = 1e-9
        elif boundary_speed > 1e-12:
            delta_t = 2.0 * distance / boundary_speed
        else:
            delta_t = np.sqrt(2.0 * distance / acceleration) + np.sqrt(2.0 * distance / deceleration)
        times[index + 1] = times[index] + max(float(delta_t), 1e-9)
    return VelocityProfile(speeds=speeds, times=times)


def _validate_directions(directions: np.ndarray, expected: int) -> np.ndarray:
    values = np.asarray(directions, dtype=np.int8)
    if values.shape != (expected,) or not np.all(np.isin(values, (-1, 1))):
        raise ValueError(f"directions 必须是长度 {expected} 且只含 -1/1 的一维数组")
    return values


__all__ = ["VelocityProfile", "trapezoidal_velocity_profile"]
