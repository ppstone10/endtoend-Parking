"""LiDAR 与 Camera BEV 融合。

将 LiDAR 的障碍物/高度/密度通道与 Camera 的目标区域通道组合为统一 BEV。
融合前校验两路 BEV 的分辨率与覆盖范围一致。
"""

from __future__ import annotations

import numpy as np

from interfaces import BEVTensor

CH_VEHICLE = "vehicle"


class BEVFusion:
    """传感器级 BEV 后融合。

    将 LiDAR BEV 与 Camera BEV 按通道拼接；可选绘制车辆轮廓通道。
    """

    def __init__(self, vehicle_length: float = 4.0, vehicle_width: float = 2.0) -> None:
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width

    def fuse(self, lidar_bev: BEVTensor, camera_bev: BEVTensor) -> BEVTensor:
        """融合两路 BEV，输出叠加通道的统一张量。"""
        if lidar_bev.resolution != camera_bev.resolution:
            raise ValueError("融合要求两路 BEV 分辨率一致")
        if lidar_bev.extent != camera_bev.extent:
            raise ValueError("融合要求两路 BEV 覆盖范围一致")

        data = np.concatenate([lidar_bev.data, camera_bev.data], axis=0)
        channels = list(lidar_bev.channels) + list(camera_bev.channels)
        vehicle = self._vehicle_outline(lidar_bev)
        if vehicle is not None:
            data = np.concatenate([data, vehicle], axis=0)
            channels.append(CH_VEHICLE)
        return BEVTensor(
            data=data,
            resolution=lidar_bev.resolution,
            extent=lidar_bev.extent,
            channels=channels,
        )

    def _vehicle_outline(self, bev: BEVTensor) -> np.ndarray | None:
        """在 BEV 中心绘制车辆轮廓通道（长×宽矩形）。"""
        front, back, left, right = bev.extent
        h, w = bev.height, bev.width
        outline = np.zeros((1, h, w), dtype=np.float32)
        # 车辆中心位于局部坐标原点 → 栅格行 front/res、列 right/res。
        row_c = front / bev.resolution
        col_c = right / bev.resolution
        row_half = int(round(self.vehicle_length / 2.0 / bev.resolution))
        col_half = int(round(self.vehicle_width / 2.0 / bev.resolution))
        r0, r1 = max(0, int(round(row_c)) - row_half), min(h, int(round(row_c)) + row_half + 1)
        c0, c1 = max(0, int(round(col_c)) - col_half), min(w, int(round(col_c)) + col_half + 1)
        outline[0, r0:r1, c0:c1] = 1.0
        return outline