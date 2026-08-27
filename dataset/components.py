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


def build_task_components(task, vehicle_config: VehicleConfig = MINING_DRILL_RIG):
    """按 Task 场景、噪声和车辆配置构造规划器与传感器管道。"""
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
    pipeline = SensorBEVPipeline(
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
    )
    planner = HybridAStarPlanner(task.scene.env, **vehicle_config.planner_kwargs())
    return planner, pipeline
