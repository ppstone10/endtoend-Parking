"""Task 驱动专家数据组件工厂。"""

from __future__ import annotations

import math

import numpy as np

from interfaces import CameraIntrinsics
from planner import HybridAStarPlanner
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import (
    MINING_DRILL_RIG,
    SimulatedCamera,
    SimulatedLiDAR,
    VehicleConfig,
    get_noise_profile,
)

from .pipeline import SensorBEVPipeline


def bev_contract_from_checkpoint(model_config) -> dict:
    """从 checkpoint 的 model_config 读取 BEV 输入契约（供运行时构造同构感知链路）。"""
    config = dict(model_config or {})
    return {
        "geometry_target": str(config.get("target_channel", "image")) == "geometry",
        "target_channel": str(config.get("target_channel", "image")),
        "height_mode": str(config.get("height_mode", "installation")),
    }


def assert_pipeline_matches_contract(pipeline, model_config) -> None:
    """校验感知管道与模型声明的 BEV 输入契约一致，不一致立即报错。

    这是一道防静默错配的关卡：用几何 target 训练的模型若在运行时收到相机
    target，开环指标会正常而闭环表现会莫名退化，极难定位。
    """
    contract = bev_contract_from_checkpoint(model_config)
    pipeline_target = (
        "geometry" if getattr(pipeline, "target_override", False) else "image"
    )
    if pipeline_target != contract["target_channel"]:
        raise ValueError(
            "感知管道 target 通道与 checkpoint 输入契约不一致："
            f"管道 {pipeline_target} vs 契约 {contract['target_channel']}"
        )


def build_task_components(
    task,
    vehicle_config: VehicleConfig = MINING_DRILL_RIG,
    *,
    model_config=None,
):
    """按 Task 场景、噪声和车辆配置构造规划器与传感器管道。

    ``model_config`` 为已载入 checkpoint 的模型配置：给出时按其中的 BEV 输入契约
    （``target_channel``）构造同构管道，避免"训练用几何 target、运行时喂相机 target"
    这类静默错配；不给时保持既有语义（相机渲染 target）。
    """
    if model_config is None:
        return build_planner_and_pipeline(task, vehicle_config, geometry_target=False)
    contract = bev_contract_from_checkpoint(model_config)
    return build_planner_and_pipeline(
        task, vehicle_config, geometry_target=contract["geometry_target"]
    )


def build_planner_and_pipeline(
    task,
    vehicle_config: VehicleConfig = MINING_DRILL_RIG,
    *,
    geometry_target: bool = True,
    target_channel: str | None = None,
):
    """按 Task 场景构造规划器与感知管道。

    **感知链路始终是传感器链路**：LiDAR 点云 → occupancy/height/density，
    车辆轮廓由 `BEVFusion` 绘制。

    ``geometry_target=True``（默认）只把 **target 通道**换成目标车位几何栅格化
    （`rasterize_goal_target`，0.25m 栅格下 6×3m 车位约 24×12 格）：
    target 的语义就是"目标车位矩形在哪"，而"相机渲染 → 单应反投影 → 逐格采样"
    会因相机视野与采样漏采目标（实测单前视 64.3%、四视角环视 9.3% 空帧）。
    ``False`` 时 target 走相机链路，用于对照。
    """
    use_geometry = geometry_target if target_channel is None else target_channel == "geometry"
    planner = HybridAStarPlanner(task.scene.env, **vehicle_config.planner_kwargs())
    return planner, _build_sensor_pipeline(
        task, vehicle_config, geometry_target=use_geometry
    )


def _build_sensor_pipeline(
    task,
    vehicle_config: VehicleConfig,
    *,
    geometry_target: bool = False,
):
    """构造"传感器 → 融合 BEV"管道（相机渲染 target 通道）。"""
    profile = get_noise_profile(task.difficulty.noise_level)
    seed_sequence = np.random.SeedSequence([task.seed, 2, 8])
    lidar_seed, camera_seed = (
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(2)
    )
    intrinsics = CameraIntrinsics(
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        image_width=640,
        image_height=480,
    )
    lidar_range = math.hypot(
        max(task.scene.bev_config.extent[:2]),
        max(task.scene.bev_config.extent[2:]),
    )
    return SensorBEVPipeline(
        lidar_sensor=SimulatedLiDAR(
            task.scene.env,
            beams=360,
            max_range=lidar_range,
            noise=profile,
            seed=lidar_seed,
        ),
        camera_sensor=SimulatedCamera(
            task.scene.env,
            intrinsics,
            parking_area=(vehicle_config.length, vehicle_config.width),
            noise=profile,
            seed=camera_seed,
        ),
        lidar2bev=LiDAR2BEV(config=task.scene.bev_config),
        camera2bev=Camera2BEV(config=task.scene.bev_config),
        bev_fusion=BEVFusion(
            vehicle_length=vehicle_config.length,
            vehicle_width=vehicle_config.width,
        ),
        target_override=geometry_target,
    )
