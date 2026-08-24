"""车辆参数统一配置。

VehicleConfig 为车辆尺寸与控制上限的唯一来源：规划器、MPC、车辆模型、
碰撞检测、可视化统一从同一 config 构造，避免尺寸散落不一致。
模块构造参数的默认值保留 4×2 兼容旧行为；脚本与实验层显式注入矿卡配置。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConfig:
    """车辆几何与控制上限。

    length/width 为车身矩形尺寸（米）；max_v/max_omega 为差分驱动控制上限。
    """

    name: str
    length: float
    width: float
    max_v: float
    max_omega: float

    def planner_kwargs(self) -> dict:
        """HybridAStarPlanner 构造参数（车辆尺寸部分）。"""
        return {"vehicle_length": self.length, "vehicle_width": self.width}

    def mpc_kwargs(self) -> dict:
        """MPCController 构造参数（控制上限部分）。"""
        return {"max_v": self.max_v, "max_omega": self.max_omega}

    def vehicle_model_kwargs(self) -> dict:
        """DifferentialDriveModel 构造参数。"""
        return {"max_v": self.max_v, "max_omega": self.max_omega}

    def collision_kwargs(self) -> dict:
        """ClosedLoopEngine 碰撞检测构造参数。"""
        return {"vehicle_length": self.length, "vehicle_width": self.width}


# 矿卡（默认）：典型中型矿卡尺寸 6×3m，低速泊车控制上限。
MINING_TRUCK = VehicleConfig("mining_truck", 6.0, 3.0, 2.0, 1.0)

# 兼容配置：项目初始阶段的 4×2 车辆，保留作回归测试。
LEGACY_4X2 = VehicleConfig("legacy_4x2", 4.0, 2.0, 2.0, 1.0)

VEHICLE_PRESETS: dict[str, VehicleConfig] = {
    MINING_TRUCK.name: MINING_TRUCK,
    LEGACY_4X2.name: LEGACY_4X2,
}


def get_vehicle(name: str) -> VehicleConfig:
    """按名称取预设车辆配置。"""
    try:
        return VEHICLE_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"未知车辆预设 {name}，可选：{sorted(VEHICLE_PRESETS)}") from exc
