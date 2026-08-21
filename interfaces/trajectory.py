"""未来轨迹点与轨迹定义。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryPoint:
    """单个未来轨迹点，车辆中心局部坐标。"""

    x: float
    y: float
    yaw: float


@dataclass
class Trajectory:
    """未来 N 个局部轨迹点。

    points 形状为 (N, 3)，列为 [x, y, yaw]。dt 为相邻点时间间隔（秒）。
    """

    points: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"轨迹点形状必须为 (N, 3)，实际 {self.points.shape}")

    @property
    def horizon(self) -> int:
        return self.points.shape[0]

    def to_points(self) -> list[TrajectoryPoint]:
        return [TrajectoryPoint(float(p[0]), float(p[1]), float(p[2])) for p in self.points]