"""车辆运动状态与目标位姿定义。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    """车辆当前运动状态。

    坐标 (x, y, yaw) 为车辆中心局部坐标系的位姿，v 为线速度（米/秒），omega 为角速度（弧度/秒）。
    """

    x: float
    y: float
    yaw: float
    v: float = 0.0
    omega: float = 0.0

    @classmethod
    def from_array(cls, state: np.ndarray) -> "VehicleState":
        if state.shape[0] < 5:
            raise ValueError(f"状态数组至少需要 5 个元素，实际 {state.shape[0]}")
        x, y, yaw, v, omega = state[:5]
        return cls(float(x), float(y), float(yaw), float(v), float(omega))

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v, self.omega], dtype=np.float64)


@dataclass
class GoalPose:
    """目标泊车位姿。"""

    x: float
    y: float
    yaw: float