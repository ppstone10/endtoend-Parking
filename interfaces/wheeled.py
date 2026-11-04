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
