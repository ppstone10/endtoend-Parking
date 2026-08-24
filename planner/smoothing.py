"""碰撞安全的三次 Hermite 捷径平滑。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from interfaces import Trajectory


PoseFree = Callable[[float, float, float], bool]


def path_length(points: np.ndarray) -> float:
    """计算位置折线总长。"""
    poses = _validate_points(points)
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)))


def smooth_trajectory(
    trajectory: Trajectory,
    pose_free: PoseFree,
    *,
    step_size: float = 0.1,
    max_curvature: float = 1.6,
    attempts: int = 80,
    seed: int = 0,
) -> Trajectory:
    """用三次捷径反复缩短轨迹，不跨越前进/倒车换向点。"""
    if step_size <= 0.0 or max_curvature <= 0.0:
        raise ValueError("step_size 与 max_curvature 必须为正数")
    if attempts < 0:
        raise ValueError("attempts 不能为负数")
    points = _validate_points(trajectory.points).copy()
    if len(points) < 3 or attempts == 0:
        return Trajectory(points=points, dt=trajectory.dt)

    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        if len(points) < 3:
            break
        i, j = sorted(rng.choice(len(points), size=2, replace=False).tolist())
        if j - i < 2:
            continue
        segment_directions = infer_segment_directions(points)
        if np.any(segment_directions[i:j] != segment_directions[i]):
            continue
        direction = int(segment_directions[i])
        candidate = _hermite_shortcut(points[i], points[j], direction, step_size)
        if path_length(candidate) >= path_length(points[i : j + 1]) - 1e-9:
            continue
        if _max_curvature(candidate) > max_curvature + 1e-9:
            continue
        if not all(pose_free(float(x), float(y), float(yaw)) for x, y, yaw in candidate):
            continue
        points = np.concatenate([points[:i], candidate, points[j + 1 :]], axis=0)

    return Trajectory(points=points, dt=trajectory.dt)


def infer_segment_directions(points: np.ndarray) -> np.ndarray:
    """根据位移与车身朝向内积推断每条边的前进/倒车方向。"""
    poses = _validate_points(points)
    deltas = np.diff(poses[:, :2], axis=0)
    headings = np.column_stack([np.cos(poses[:-1, 2]), np.sin(poses[:-1, 2])])
    alignment = np.sum(deltas * headings, axis=1)
    directions = np.where(alignment >= 0.0, 1, -1).astype(np.int8)
    zero_edges = np.linalg.norm(deltas, axis=1) <= 1e-12
    for index in np.flatnonzero(zero_edges):
        directions[index] = directions[index - 1] if index > 0 else 1
    return directions


def _hermite_shortcut(
    start: np.ndarray,
    goal: np.ndarray,
    direction: int,
    step_size: float,
) -> np.ndarray:
    chord = float(np.linalg.norm(goal[:2] - start[:2]))
    count = max(2, int(np.ceil(max(chord, step_size) * 2.0 / step_size)))
    ts = np.linspace(0.0, 1.0, count + 1)
    travel_start = direction * np.array([np.cos(start[2]), np.sin(start[2])])
    travel_goal = direction * np.array([np.cos(goal[2]), np.sin(goal[2])])
    tangent_scale = chord * 0.5
    m0, m1 = tangent_scale * travel_start, tangent_scale * travel_goal

    t = ts[:, None]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    xy = h00 * start[:2] + h10 * m0 + h01 * goal[:2] + h11 * m1

    derivative = (
        (6.0 * t**2 - 6.0 * t) * start[:2]
        + (3.0 * t**2 - 4.0 * t + 1.0) * m0
        + (-6.0 * t**2 + 6.0 * t) * goal[:2]
        + (3.0 * t**2 - 2.0 * t) * m1
    )
    travel_yaw = np.arctan2(derivative[:, 1], derivative[:, 0])
    yaws = _normalize_angles(travel_yaw if direction > 0 else travel_yaw + np.pi)
    poses = np.column_stack([xy, yaws])
    poses[0] = start
    poses[-1] = goal
    return poses


def _max_curvature(points: np.ndarray) -> float:
    ds = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    dyaw = _normalize_angles(np.diff(points[:, 2]))
    valid = ds > 1e-9
    if not np.any(valid):
        return 0.0
    return float(np.max(np.abs(dyaw[valid]) / ds[valid]))


def _validate_points(points: np.ndarray) -> np.ndarray:
    poses = np.asarray(points, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2:
        raise ValueError("轨迹必须是至少两点的 (N, 3) 数组")
    if not np.all(np.isfinite(poses)):
        raise ValueError("轨迹不能含非有限数")
    return poses


def _normalize_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


__all__ = ["infer_segment_directions", "path_length", "smooth_trajectory"]
