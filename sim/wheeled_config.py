"""轮式独立配置及严格身份，不改变历史VehicleConfig元数据。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any


_DECLARATIONS = {
    "schema_version": 1,
    "motion_model": "kinematic_bicycle",
    "pose_reference": "body_center",
    "speed_reference": "rear_axle_longitudinal",
    "actuation_model": "sampled_slew_zoh",
}


@dataclass(frozen=True)
class WheeledVehicleConfig:
    """刚性车架前轮转向的研究参数，角度单位统一为rad。

    前后悬分别从对应车轴量到外廓边界，位姿原点仍是几何中心。
    parameter_source/notes必须表明参数依据；示例不代表实车标定。
    """

    name: str
    model_version: str
    length: float
    width: float
    wheelbase: float
    front_overhang: float
    rear_overhang: float
    max_forward_speed: float
    max_reverse_speed: float
    max_steering_angle: float
    max_steering_rate: float
    max_acceleration: float
    max_deceleration: float
    collision_margin: float
    parameter_source: str
    parameter_notes: str

    def __post_init__(self) -> None:
        text_fields = {"name", "model_version", "parameter_source", "parameter_notes"}
        nonnegative_fields = {"front_overhang", "rear_overhang", "collision_margin"}
        for name, value in asdict(self).items():
            if name in text_fields:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} 必须是非空文字")
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数值，不能是布尔值")
            if value < 0 or (value == 0 and name not in nonnegative_fields):
                raise ValueError(f"{name} 的数值范围无效")
            object.__setattr__(self, name, float(value))
        if not math.isclose(self.length, self.wheelbase + self.front_overhang + self.rear_overhang,
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError("length 必须等于 wheelbase + front_overhang + rear_overhang")
        if self.max_steering_angle >= math.pi / 2:
            raise ValueError("max_steering_angle 必须严格小于pi/2")
        if not math.isfinite(self.min_turning_radius) or self.min_turning_radius <= 0:
            raise ValueError("派生最小转弯半径必须有限且为正")
        if not math.isfinite(max(self.max_forward_speed, self.max_reverse_speed) / self.min_turning_radius):
            raise ValueError("派生角速度上限必须有限")

    @property
    def center_from_rear(self) -> float:
        """后轴中心到车体中心沿车身x轴的有符号距离。"""
        return self.length / 2 - self.rear_overhang

    @property
    def min_turning_radius(self) -> float:
        """后轴中心的最小转弯半径；不是车身最外点扫掠半径。"""
        return self.wheelbase / math.tan(self.max_steering_angle)

    def to_metadata(self) -> dict[str, Any]:
        return {**_DECLARATIONS, "parameters": asdict(self)}

    @property
    def identity(self) -> str:
        encoded = json.dumps(self.to_metadata(), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> WheeledVehicleConfig:
        if not isinstance(metadata, dict) or set(metadata) != set(_DECLARATIONS) | {"parameters"}:
            raise ValueError("轮式配置声明字段缺失或未知，不接受旧履带元数据")
        for name, expected in _DECLARATIONS.items():
            if type(metadata[name]) is not type(expected) or metadata[name] != expected:
                raise ValueError(f"不支持的轮式配置声明：{name}")
        if not isinstance(metadata["parameters"], dict):
            raise ValueError("parameters 必须是配置对象")
        try:
            return cls(**metadata["parameters"])
        except TypeError as exc:
            raise ValueError(f"轮式参数字段无效：{exc}") from exc


def load_wheeled_vehicle_config(path: str | Path) -> WheeledVehicleConfig:
    """加载专用版本配置；不将其注册为历史mining_truck预设。"""
    try:
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法加载轮式配置 {path}：{exc}") from exc
    return WheeledVehicleConfig.from_metadata(metadata)
