"""轮式基础接口；直接导入，避免改变历史五维状态与差速指令。"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real

import numpy as np


def _finite_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值，不能是布尔值")
    return float(value)


@dataclass(frozen=True)
class WheeledControlCmd:
    """请求后轴有符号纵向速度v与等效前轮转角steering_angle（rad）。"""

    v: float
    steering_angle: float

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _finite_number(getattr(self, field.name), field.name))


@dataclass(frozen=True)
class WheeledVehicleState:
    """世界系车体中心位姿、后轴纵向速度与实际转角。

    v不是转弯时中心速度的模长。转角必须显式保存，停车时不能用omega恢复。
    此数组与旧VehicleState虽同为五维，最后一维含义不同，禁止隐式互换。
    """

    x: float
    y: float
    yaw: float
    v: float = 0.0
    steering_angle: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _finite_number(getattr(self, field.name), field.name))

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v, self.steering_angle], dtype=np.float64)

    @classmethod
    def from_array(cls, values) -> WheeledVehicleState:
        """仅用于已确认轮式契约的数组；持久化需另外携带配置身份。"""
        array = np.asarray(values, dtype=object)
        if array.shape != (5,):
            raise ValueError("轮式状态数组必须恰为(x,y,yaw,v,steering_angle)五维")
        return cls(*array)


@dataclass(frozen=True)
class WheeledTrajectory:
    """绑定车型身份的世界系车体中心参考；不接受旧Trajectory隐式转换。"""

    points: np.ndarray
    dt: float
    model_identity: str

    def __post_init__(self) -> None:
        dt = _finite_number(self.dt, "参考dt")
        if dt <= 0:
            raise ValueError("参考dt必须为正")
        if (not isinstance(self.model_identity, str) or len(self.model_identity) != 64
                or any(c not in "0123456789abcdef" for c in self.model_identity)):
            raise ValueError("参考必须绑定完整轮式配置SHA256身份")
        points = np.array(self.points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("轮式参考必须是非空有限N×3车体中心位姿")
        points.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "dt", dt)
