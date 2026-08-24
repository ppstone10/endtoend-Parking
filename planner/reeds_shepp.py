"""Reeds–Shepp 48 词族枚举、最短路候选与弧长采样。

公式源自 Reeds & Shepp (1990) 第 8 节；48 词族按 12 个基本公式
及原型/时间反演/镜像/双对称组织。实现交叉核对了 OMPL 与
PythonRobotics 的成熟开源实现，归属与许可说明见 THIRD_PARTY_NOTICES.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, asin, atan2, cos, fmod, hypot, pi, sin, sqrt
from typing import Callable, Sequence

import numpy as np


_EPS = 1e-12
_SYMMETRIES = ("identity", "timeflip", "reflect", "timeflip_reflect")


@dataclass(frozen=True)
class ReedsSheppWord:
    """48 词族表的一项。"""

    base: str
    symmetry: str

    @property
    def name(self) -> str:
        return f"{self.base}:{self.symmetry}"


@dataclass(frozen=True)
class ReedsSheppPath:
    """一条 Reeds–Shepp 候选；lengths 为带方向的米制段长。"""

    modes: tuple[str, ...]
    lengths: tuple[float, ...]
    turning_radius: float
    word: ReedsSheppWord

    @property
    def total_length(self) -> float:
        return float(sum(abs(length) for length in self.lengths))

    @property
    def cusp_count(self) -> int:
        signs = [1 if length >= 0.0 else -1 for length in self.lengths if abs(length) > _EPS]
        return sum(lhs != rhs for lhs, rhs in zip(signs, signs[1:]))

    def sample(
        self,
        start: Sequence[float],
        step_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """从 start 按弧长等距采样，返回 ``(N,3)`` 位姿和 ``(N,)`` 方向。"""
        if step_size <= 0.0:
            raise ValueError("step_size 必须为正数")
        x, y, yaw = _pose_tuple(start, "start")
        first_direction = _direction(self.lengths[0]) if self.lengths else 1
        points: list[tuple[float, float, float]] = [(x, y, _mod2pi(yaw))]
        directions: list[int] = [first_direction]

        for mode, metric_length in zip(self.modes, self.lengths):
            if abs(metric_length) <= _EPS:
                continue
            origin = (x, y, yaw)
            count = max(1, int(np.ceil(abs(metric_length) / step_size)))
            for index in range(1, count + 1):
                distance = metric_length * index / count
                x, y, yaw = _interpolate_segment(origin, mode, distance, self.turning_radius)
                points.append((x, y, _mod2pi(yaw)))
                directions.append(_direction(metric_length))

        return np.asarray(points, dtype=float), np.asarray(directions, dtype=np.int8)


_Solver = Callable[[float, float, float], tuple[list[float], list[str]] | None]


def _mod2pi(angle: float) -> float:
    value = fmod(angle, 2.0 * pi)
    if value < -pi:
        value += 2.0 * pi
    elif value > pi:
        value -= 2.0 * pi
    return value


def _pose_tuple(pose: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(pose) != 3:
        raise ValueError(f"{name} 必须含 [x, y, yaw]")
    values = tuple(float(value) for value in pose)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 必须为有限数")
    return values


def _direction(length: float) -> int:
    return 1 if length >= 0.0 else -1


def _ls_l(x: float, y: float, phi: float):
    u, t = hypot(x - sin(phi), y - 1.0 + cos(phi)), atan2(y - 1.0 + cos(phi), x - sin(phi))
    if -_EPS <= t <= pi + _EPS:
        v = _mod2pi(phi - t)
        if -_EPS <= v <= pi + _EPS:
            return [t, u, v], ["L", "S", "L"]
    return None


def _ls_r(x: float, y: float, phi: float):
    u1 = hypot(x + sin(phi), y - 1.0 - cos(phi))
    t1 = atan2(y - 1.0 - cos(phi), x + sin(phi))
    if u1 * u1 >= 4.0 - _EPS:
        u = sqrt(max(0.0, u1 * u1 - 4.0))
        t = _mod2pi(t1 + atan2(2.0, u))
        v = _mod2pi(t - phi)
        if t >= -_EPS and v >= -_EPS:
            return [t, u, v], ["L", "S", "R"]
    return None


def _l_x_r_x_l(x: float, y: float, phi: float):
    u1 = hypot(x - sin(phi), y - 1.0 + cos(phi))
    theta = atan2(y - 1.0 + cos(phi), x - sin(phi))
    if u1 <= 4.0 + _EPS:
        a = acos(np.clip(0.25 * u1, -1.0, 1.0))
        t = _mod2pi(a + theta + pi / 2.0)
        u = _mod2pi(pi - 2.0 * a)
        v = _mod2pi(phi - t - u)
        return [t, -u, v], ["L", "R", "L"]
    return None


def _l_x_r_l(x: float, y: float, phi: float):
    u1 = hypot(x - sin(phi), y - 1.0 + cos(phi))
    theta = atan2(y - 1.0 + cos(phi), x - sin(phi))
    if u1 <= 4.0 + _EPS:
        a = acos(np.clip(0.25 * u1, -1.0, 1.0))
        t = _mod2pi(a + theta + pi / 2.0)
        u = _mod2pi(pi - 2.0 * a)
        v = _mod2pi(-phi + t + u)
        return [t, -u, -v], ["L", "R", "L"]
    return None


def _l_r_x_l(x: float, y: float, phi: float):
    u1 = hypot(x - sin(phi), y - 1.0 + cos(phi))
    theta = atan2(y - 1.0 + cos(phi), x - sin(phi))
    if _EPS < u1 <= 4.0 + _EPS:
        u = acos(np.clip(1.0 - u1 * u1 / 8.0, -1.0, 1.0))
        a = asin(np.clip(2.0 * sin(u) / u1, -1.0, 1.0))
        t = _mod2pi(-a + theta + pi / 2.0)
        v = _mod2pi(t - u - phi)
        return [t, u, -v], ["L", "R", "L"]
    return None


def _l_r_x_l_r(x: float, y: float, phi: float):
    u1 = hypot(x + sin(phi), y - 1.0 - cos(phi))
    theta = atan2(y - 1.0 - cos(phi), x + sin(phi))
    if u1 <= 2.0 + _EPS:
        a = acos(np.clip((u1 + 2.0) / 4.0, -1.0, 1.0))
        t = _mod2pi(theta + a + pi / 2.0)
        u = _mod2pi(a)
        v = _mod2pi(phi - t + 2.0 * u)
        if t >= -_EPS and u >= -_EPS and v >= -_EPS:
            return [t, u, -u, -v], ["L", "R", "L", "R"]
    return None


def _l_x_r_l_x_r(x: float, y: float, phi: float):
    u1 = hypot(x + sin(phi), y - 1.0 - cos(phi))
    theta = atan2(y - 1.0 - cos(phi), x + sin(phi))
    u2 = (20.0 - u1 * u1) / 16.0
    if -_EPS <= u2 <= 1.0 + _EPS and u1 > _EPS:
        u = acos(np.clip(u2, -1.0, 1.0))
        a = asin(np.clip(2.0 * sin(u) / u1, -1.0, 1.0))
        t = _mod2pi(theta + a + pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= -_EPS and v >= -_EPS:
            return [t, -u, -u, v], ["L", "R", "L", "R"]
    return None


def _l_x_r90_s_l(x: float, y: float, phi: float):
    u1 = hypot(x - sin(phi), y - 1.0 + cos(phi))
    theta = atan2(y - 1.0 + cos(phi), x - sin(phi))
    if u1 >= 2.0 - _EPS:
        root = sqrt(max(0.0, u1 * u1 - 4.0))
        u = root - 2.0
        t = _mod2pi(theta + atan2(2.0, root) + pi / 2.0)
        v = _mod2pi(t - phi + pi / 2.0)
        if t >= -_EPS and v >= -_EPS:
            return [t, -pi / 2.0, -u, -v], ["L", "R", "S", "L"]
    return None


def _l_x_r90_s_r(x: float, y: float, phi: float):
    u1 = hypot(x + sin(phi), y - 1.0 - cos(phi))
    theta = atan2(y - 1.0 - cos(phi), x + sin(phi))
    if u1 >= 2.0 - _EPS:
        t = _mod2pi(theta + pi / 2.0)
        u = u1 - 2.0
        v = _mod2pi(phi - t - pi / 2.0)
        if t >= -_EPS and v >= -_EPS:
            return [t, -pi / 2.0, -u, -v], ["L", "R", "S", "R"]
    return None


def _l_s_r90_x_l(x: float, y: float, phi: float):
    u1 = hypot(x - sin(phi), y - 1.0 + cos(phi))
    theta = atan2(y - 1.0 + cos(phi), x - sin(phi))
    if u1 >= 2.0 - _EPS:
        root = sqrt(max(0.0, u1 * u1 - 4.0))
        u = root - 2.0
        t = _mod2pi(theta - atan2(root, 2.0) + pi / 2.0)
        v = _mod2pi(t - phi - pi / 2.0)
        if t >= -_EPS and v >= -_EPS:
            return [t, u, pi / 2.0, -v], ["L", "S", "R", "L"]
    return None


def _l_s_l90_x_r(x: float, y: float, phi: float):
    u1 = hypot(x + sin(phi), y - 1.0 - cos(phi))
    theta = atan2(y - 1.0 - cos(phi), x + sin(phi))
    if u1 >= 2.0 - _EPS:
        t = _mod2pi(theta)
        u = u1 - 2.0
        v = _mod2pi(phi - t - pi / 2.0)
        if t >= -_EPS and v >= -_EPS:
            return [t, u, pi / 2.0, -v], ["L", "S", "L", "R"]
    return None


def _l_x_r90_s_l90_x_r(x: float, y: float, phi: float):
    u1 = hypot(x + sin(phi), y - 1.0 - cos(phi))
    theta = atan2(y - 1.0 - cos(phi), x + sin(phi))
    if u1 >= 4.0 - _EPS:
        root = sqrt(max(0.0, u1 * u1 - 4.0))
        u = root - 4.0
        t = _mod2pi(theta + atan2(2.0, root) + pi / 2.0)
        v = _mod2pi(t - phi)
        if t >= -_EPS and v >= -_EPS:
            return [t, -pi / 2.0, -u, -pi / 2.0, v], ["L", "R", "S", "L", "R"]
    return None


_BASE_SOLVERS: tuple[tuple[str, _Solver], ...] = (
    ("LSL", _ls_l),
    ("LSR", _ls_r),
    ("LxRxL", _l_x_r_x_l),
    ("LxRL", _l_x_r_l),
    ("LRxL", _l_r_x_l),
    ("LRxLR", _l_r_x_l_r),
    ("LxRLxR", _l_x_r_l_x_r),
    ("LxR90SL", _l_x_r90_s_l),
    ("LxR90SR", _l_x_r90_s_r),
    ("LSR90xL", _l_s_r90_x_l),
    ("LSL90xR", _l_s_l90_x_r),
    ("LxR90SL90xR", _l_x_r90_s_l90_x_r),
)

REEDS_SHEPP_WORDS: tuple[ReedsSheppWord, ...] = tuple(
    ReedsSheppWord(base, symmetry)
    for base, _solver in _BASE_SOLVERS
    for symmetry in _SYMMETRIES
)


def reeds_shepp_paths(
    start: Sequence[float],
    goal: Sequence[float],
    turning_radius: float,
) -> list[ReedsSheppPath]:
    """枚举去重后的可行候选，按米制总长从短到长返回。"""
    if turning_radius <= 0.0 or not np.isfinite(turning_radius):
        raise ValueError("turning_radius 必须为有限正数")
    sx, sy, syaw = _pose_tuple(start, "start")
    gx, gy, gyaw = _pose_tuple(goal, "goal")
    dx, dy = gx - sx, gy - sy
    c, s = cos(syaw), sin(syaw)
    x = (c * dx + s * dy) / turning_radius
    y = (-s * dx + c * dy) / turning_radius
    phi = _mod2pi(gyaw - syaw)

    if hypot(x, y) <= _EPS and abs(phi) <= _EPS:
        word = REEDS_SHEPP_WORDS[0]
        return [ReedsSheppPath(("S",), (0.0,), turning_radius, word)]

    candidates: list[ReedsSheppPath] = []
    signatures: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
    solver_by_name = dict(_BASE_SOLVERS)
    for word in REEDS_SHEPP_WORDS:
        solver = solver_by_name[word.base]
        if word.symmetry == "identity":
            solved = solver(x, y, phi)
        elif word.symmetry == "timeflip":
            solved = solver(-x, y, -phi)
        elif word.symmetry == "reflect":
            solved = solver(x, -y, -phi)
        else:
            solved = solver(-x, -y, phi)
        if solved is None:
            continue
        normalized_lengths, modes = solved
        if "timeflip" in word.symmetry:
            normalized_lengths = [-length for length in normalized_lengths]
        if "reflect" in word.symmetry:
            modes = [_reflect(mode) for mode in modes]

        compact = [
            (mode, float(length) * turning_radius)
            for mode, length in zip(modes, normalized_lengths)
            if abs(length) > _EPS
        ]
        if not compact:
            continue
        metric_modes = tuple(item[0] for item in compact)
        metric_lengths = tuple(item[1] for item in compact)
        signature = (metric_modes, tuple(round(length / turning_radius, 10) for length in metric_lengths))
        if signature in signatures:
            continue
        path = ReedsSheppPath(metric_modes, metric_lengths, turning_radius, word)
        endpoint, _ = path.sample((sx, sy, syaw), step_size=max(path.total_length, 1.0))
        position_error = hypot(float(endpoint[-1, 0]) - gx, float(endpoint[-1, 1]) - gy)
        yaw_error = abs(_mod2pi(float(endpoint[-1, 2]) - gyaw))
        if position_error > 1e-7 or yaw_error > 1e-7:
            continue
        signatures.add(signature)
        candidates.append(path)

    candidates.sort(key=lambda path: (path.total_length, path.cusp_count, path.word.name))
    return candidates


def _reflect(mode: str) -> str:
    return {"L": "R", "R": "L", "S": "S"}[mode]


def _interpolate_segment(
    origin: tuple[float, float, float],
    mode: str,
    distance: float,
    radius: float,
) -> tuple[float, float, float]:
    x, y, yaw = origin
    if mode == "S":
        return x + distance * cos(yaw), y + distance * sin(yaw), yaw
    angle = distance / radius
    if mode == "L":
        return (
            x + radius * (sin(yaw + angle) - sin(yaw)),
            y + radius * (-cos(yaw + angle) + cos(yaw)),
            yaw + angle,
        )
    if mode == "R":
        return (
            x + radius * (-sin(yaw - angle) + sin(yaw)),
            y + radius * (cos(yaw - angle) - cos(yaw)),
            yaw - angle,
        )
    raise ValueError(f"未知 Reeds–Shepp 段类型: {mode}")


__all__ = ["REEDS_SHEPP_WORDS", "ReedsSheppPath", "ReedsSheppWord", "reeds_shepp_paths"]
