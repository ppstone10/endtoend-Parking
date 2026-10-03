"""传感器 → 融合 BEV 管道。

将模拟 LiDAR/Camera 传感器与 Sensor2BEV 转换、融合组合为单一入口，
供数据集生成复用。
"""

from __future__ import annotations

import numpy as np

from interfaces import BEVConfig, BEVTensor, GoalPose
from .target_bev import rasterize_goal_target


class SensorBEVPipeline:
    """组合传感器采集与 BEV 转换的适配器。

    lidar_sensor/camera_sensor 为 sim 中的模拟传感器；
    lidar2bev/camera2bev/bev_fusion 为 sensor2bev 模块的转换与融合组件。

    ``target_override`` 为 True 时，target 通道改用**目标车位几何栅格化**
    （`dataset.target_bev.rasterize_goal_target`）替换相机反投影结果：
    相机视野与逐格采样都会漏采目标，而 target 的语义就是"目标车位矩形在哪"，
    故以几何栅格化为主口径，相机链路保留供对照。
    """

    def __init__(
        self,
        lidar_sensor,
        camera_sensor,
        lidar2bev,
        camera2bev,
        bev_fusion,
        *,
        target_override: bool = False,
    ) -> None:
        self.lidar_sensor = lidar_sensor
        self.camera_sensor = camera_sensor
        self.lidar2bev = lidar2bev
        self.camera2bev = camera2bev
        self.bev_fusion = bev_fusion
        self.target_override = bool(target_override)
        self.bev_config = BEVConfig(
            resolution=lidar2bev.resolution,
            extent=lidar2bev.extent,
        )
        camera_config = BEVConfig(
            resolution=camera2bev.resolution,
            extent=camera2bev.extent,
        )
        if camera_config != self.bev_config:
            raise ValueError("SensorBEVPipeline 要求 LiDAR/Camera 使用同一 BEV 配置")
        self._goals: list[GoalPose] = []

    def capture_bev(self, x: float, y: float, yaw: float) -> BEVTensor:
        """采集一帧融合 BEV。"""
        lidar_frame = self.lidar_sensor.capture(x, y, yaw)
        camera_frame = self.camera_sensor.capture(x, y, yaw)
        lidar_bev = self.lidar2bev.to_bev(lidar_frame, x, y, yaw)
        camera_bev = self.camera2bev.to_bev(camera_frame, x, y, yaw)
        override = None
        if self.target_override:
            override = np.zeros(self.bev_config.shape, dtype=np.float32)
            for goal in self._goals:
                override = np.maximum(
                    override,
                    rasterize_goal_target(
                        goal,
                        self.bev_fusion.vehicle_length,
                        self.bev_fusion.vehicle_width,
                        x,
                        y,
                        yaw,
                        self.bev_config,
                    ),
                )
        return self.bev_fusion.fuse(lidar_bev, camera_bev, target_override=override)

    def set_target_goals(self, goals: list[GoalPose]) -> None:
        """设置当前监督样本的目标：既渲染到 Camera 通道，也用于几何 target。"""
        self._goals = list(goals)
        self.camera_sensor.env.parking_spots = list(goals)
