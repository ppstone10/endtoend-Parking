"""传感器 → 融合 BEV 管道。

将模拟 LiDAR/Camera 传感器与 Sensor2BEV 转换、融合组合为单一入口，
供数据集生成复用。
"""

from __future__ import annotations

from interfaces import BEVTensor


class SensorBEVPipeline:
    """组合传感器采集与 BEV 转换的适配器。

    lidar_sensor/camera_sensor 为 sim 中的模拟传感器；
    lidar2bev/camera2bev/bev_fusion 为 sensor2bev 模块的转换与融合组件。
    """

    def __init__(
        self,
        lidar_sensor,
        camera_sensor,
        lidar2bev,
        camera2bev,
        bev_fusion,
    ) -> None:
        self.lidar_sensor = lidar_sensor
        self.camera_sensor = camera_sensor
        self.lidar2bev = lidar2bev
        self.camera2bev = camera2bev
        self.bev_fusion = bev_fusion

    def capture_bev(self, x: float, y: float, yaw: float) -> BEVTensor:
        """采集一帧融合 BEV。"""
        lidar_frame = self.lidar_sensor.capture(x, y, yaw)
        camera_frame = self.camera_sensor.capture(x, y, yaw)
        lidar_bev = self.lidar2bev.to_bev(lidar_frame, x, y, yaw)
        camera_bev = self.camera2bev.to_bev(camera_frame, x, y, yaw)
        return self.bev_fusion.fuse(lidar_bev, camera_bev)