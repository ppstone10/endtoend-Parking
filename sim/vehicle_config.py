"""车辆参数统一配置与 JSON 加载。

VehicleConfig 为车辆尺寸与控制上限的唯一来源：规划器、MPC、车辆模型、
碰撞检测、可视化统一从同一 config 构造，避免尺寸散落不一致。
模块构造参数的默认值保留 4×2 兼容旧行为；脚本与实验层显式注入矿卡配置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VehicleConfig:
    """车辆几何与控制上限。

    length/width 为以履带几何中心居中的矩形外廓（米）；max_v/max_omega
    为底盘执行上限；plan_* 与其余字段只控制专家规划搜索。
    """

    name: str
    length: float
    width: float
    max_v: float
    max_omega: float
    model_version: str = "legacy_v1"
    plan_v: float = 0.5
    plan_max_omega: float = 0.8
    collision_margin: float = 0.0
    xy_resolution: float = 0.5
    yaw_resolution_deg: float = 30.0
    motion_resolution: float = 0.1
    collision_check_resolution: float = 0.1
    analytic_expansion_distance: float | None = None
    enable_pivot: bool = False
    pivot_omega: float = 0.35
    rotation_penalty: float = 2.0
    direction_mismatch_penalty: float = 2.0
    max_planning_time_s: float = 8.0

    def __post_init__(self) -> None:
        positive = {
            "length": self.length,
            "width": self.width,
            "max_v": self.max_v,
            "max_omega": self.max_omega,
            "plan_v": self.plan_v,
            "plan_max_omega": self.plan_max_omega,
            "xy_resolution": self.xy_resolution,
            "yaw_resolution_deg": self.yaw_resolution_deg,
            "motion_resolution": self.motion_resolution,
            "collision_check_resolution": self.collision_check_resolution,
            "pivot_omega": self.pivot_omega,
            "rotation_penalty": self.rotation_penalty,
            "direction_mismatch_penalty": self.direction_mismatch_penalty,
            "max_planning_time_s": self.max_planning_time_s,
        }
        invalid = [
            key
            for key, value in positive.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(f"车辆配置必须为有限正数：{', '.join(invalid)}")
        if not math.isfinite(self.collision_margin) or self.collision_margin < 0.0:
            raise ValueError("collision_margin 必须为有限非负数")
        if (
            self.analytic_expansion_distance is not None
            and (
                not math.isfinite(self.analytic_expansion_distance)
                or self.analytic_expansion_distance < 0.0
            )
        ):
            raise ValueError("analytic_expansion_distance 必须为有限非负数或 None")
        if self.plan_v > self.max_v:
            raise ValueError("plan_v 不能高于底盘 max_v")
        if self.plan_max_omega > self.max_omega:
            raise ValueError("plan_max_omega 不能高于底盘 max_omega")
        if self.enable_pivot and self.pivot_omega > self.plan_max_omega:
            raise ValueError("pivot_omega 不能高于 plan_max_omega")
        if self.direction_mismatch_penalty < 1.0:
            raise ValueError("direction_mismatch_penalty 不能小于 1")
        if not self.name or not self.model_version:
            raise ValueError("name 与 model_version 不能为空")

    def planner_kwargs(self) -> dict:
        """HybridAStarPlanner 构造参数。"""
        return {
            "vehicle_length": self.length,
            "vehicle_width": self.width,
            "plan_v": self.plan_v,
            "max_omega": self.plan_max_omega,
            "collision_margin": self.collision_margin,
            "xy_resolution": self.xy_resolution,
            "yaw_resolution": math.radians(self.yaw_resolution_deg),
            "motion_resolution": self.motion_resolution,
            "collision_check_resolution": self.collision_check_resolution,
            "analytic_expansion_distance": self.analytic_expansion_distance,
            "enable_pivot": self.enable_pivot,
            "pivot_omega": self.pivot_omega,
            "rotation_penalty": self.rotation_penalty,
            "direction_mismatch_penalty": self.direction_mismatch_penalty,
            "max_planning_time_s": self.max_planning_time_s,
            "vehicle_model_name": self.name,
            "vehicle_model_version": self.model_version,
            "vehicle_model_metadata": self.to_metadata(),
        }

    def mpc_kwargs(self) -> dict:
        """MPCController 构造参数（控制上限部分）。"""
        return {"max_v": self.max_v, "max_omega": self.max_omega}

    def vehicle_model_kwargs(self) -> dict:
        """DifferentialDriveModel 构造参数。"""
        return {"max_v": self.max_v, "max_omega": self.max_omega}

    def collision_kwargs(self) -> dict:
        """ClosedLoopEngine 碰撞检测构造参数。"""
        return {"vehicle_length": self.length, "vehicle_width": self.width}

    def to_metadata(self) -> dict[str, Any]:
        """返回可稳定写入任务元数据的完整理论/标定配置。"""
        return asdict(self)


def load_vehicle_config(path: str | Path) -> VehicleConfig:
    """从可编辑 JSON 文件加载并严格校验车辆配置。"""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法加载车辆配置 {config_path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("车辆配置根节点必须是 JSON 对象")
    try:
        return VehicleConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"车辆配置字段无效：{exc}") from exc


# 默认理论履带钻机：从可编辑配置加载；并保留 MINING_TRUCK 名称兼容旧调用。
_DEFAULT_DRILL_RIG_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "vehicles"
    / "tracked_drill_rig.json"
)
MINING_DRILL_RIG = load_vehicle_config(_DEFAULT_DRILL_RIG_CONFIG)
MINING_TRUCK = MINING_DRILL_RIG

# 兼容配置：项目初始阶段的 4×2 车辆，保留作回归测试。
LEGACY_4X2 = VehicleConfig("legacy_4x2", 4.0, 2.0, 2.0, 1.0)

VEHICLE_PRESETS: dict[str, VehicleConfig] = {
    MINING_DRILL_RIG.name: MINING_DRILL_RIG,
    "mining_truck": MINING_DRILL_RIG,
    LEGACY_4X2.name: LEGACY_4X2,
}


def get_vehicle(name: str) -> VehicleConfig:
    """按名称取预设车辆配置。"""
    try:
        return VEHICLE_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"未知车辆预设 {name}，可选：{sorted(VEHICLE_PRESETS)}") from exc
