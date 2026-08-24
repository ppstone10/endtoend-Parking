"""LiDAR 点云到 BEV 的转换。

流程：ROI 裁剪 → 降采样 → 地面滤除 → 二维栅格投影。
生成的 BEV 以车辆为中心，通道语义见 BEVTensor.channels。
"""

from __future__ import annotations

import numpy as np

from interfaces import BEVConfig, BEVTensor, LiDARFrame

# 标准通道顺序
CH_OCCUPANCY = "occupancy"
CH_HEIGHT = "height"
CH_DENSITY = "density"


class LiDAR2BEV:
    """将 LiDAR 点云转换为多通道 BEV 张量。

    resolution 为每栅格米数；extent 为 (front, back, left, right) 米；
    ground_z 以下视为地面点并滤除；max_points 为降采样后的最大点数。
    """

    def __init__(
        self,
        resolution: float | None = None,
        extent: tuple[float, float, float, float] | None = None,
        ground_z: float = 0.1,
        max_points: int = 10000,
        *,
        config: BEVConfig | None = None,
    ) -> None:
        if config is not None and (resolution is not None or extent is not None):
            raise ValueError("config 不能与 resolution/extent 同时传入")
        if config is None:
            config = BEVConfig(
                resolution=0.25 if resolution is None else resolution,
                extent=(20.0, 20.0, 20.0, 20.0) if extent is None else extent,
            )
        self.config = config
        self.resolution = config.resolution
        self.extent = config.extent
        self.ground_z = ground_z
        self.max_points = max_points
        self.channels = [CH_OCCUPANCY, CH_HEIGHT, CH_DENSITY]

    def to_bev(self, frame: LiDARFrame, x: float, y: float, yaw: float) -> BEVTensor:
        """将点云转换到车辆中心局部系并生成 BEV。"""
        pts = self._roi_crop(frame.points, x, y, yaw)
        pts = self._downsample(pts)
        pts = self._remove_ground(pts)
        return self._project(pts)

    def _to_local(self, pts: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
        """全局点云变换到车辆中心局部系（绕 z 旋转并平移）。"""
        dx = pts[:, 0] - x
        dy = pts[:, 1] - y
        cos, sin = np.cos(-yaw), np.sin(-yaw)
        local = np.empty_like(pts)
        local[:, 0] = cos * dx - sin * dy
        local[:, 1] = sin * dx + cos * dy
        local[:, 2:] = pts[:, 2:]
        return local

    def _roi_crop(
        self, pts: np.ndarray, x: float, y: float, yaw: float
    ) -> np.ndarray:
        """裁剪出车辆局部坐标 ROI 范围内的点。"""
        local = self._to_local(pts, x, y, yaw)
        front, back, left, right = self.extent
        mask = (
            (local[:, 0] >= -back)
            & (local[:, 0] <= front)
            & (local[:, 1] >= -right)
            & (local[:, 1] <= left)
        )
        return local[mask]

    def _downsample(self, pts: np.ndarray) -> np.ndarray:
        """均匀随机降采样到 max_points 以内。"""
        if pts.shape[0] <= self.max_points:
            return pts
        indices = np.random.choice(pts.shape[0], self.max_points, replace=False)
        return pts[indices]

    def _remove_ground(self, pts: np.ndarray) -> np.ndarray:
        """滤除地面点（z 低于阈值的点）。"""
        return pts[pts[:, 2] > self.ground_z]

    def _project(self, pts: np.ndarray) -> BEVTensor:
        """将局部系点云投影到栅格，生成占据/高度/密度三通道。"""
        front, back, left, right = self.extent
        h = round((front + back) / self.resolution)
        w = round((left + right) / self.resolution)
        occupancy = np.zeros((h, w), dtype=np.float32)
        height = np.zeros((h, w), dtype=np.float32)
        density = np.zeros((h, w), dtype=np.float32)

        if pts.shape[0] == 0:
            data = np.stack([occupancy, height, density], axis=0)
            return BEVTensor(data=data, resolution=self.resolution, extent=self.extent, channels=self.channels)

        # 局部坐标 → 栅格行列：行对应前向(纵向)，列对应左向(横向)。
        col = np.floor((pts[:, 1] + right) / self.resolution).astype(int)
        row = np.floor((front - pts[:, 0]) / self.resolution).astype(int)
        valid = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        row, col, pts = row[valid], col[valid], pts[valid]

        # 高度通道取每个栅格内点的最大 z（归一化到地面高度之上）。
        for r, c, z in zip(row, col, pts[:, 2]):
            occupancy[r, c] = 1.0
            height[r, c] = max(height[r, c], float(z))
            density[r, c] += 1.0

        density = np.minimum(density, 5.0) / 5.0
        data = np.stack([occupancy, height, density], axis=0)
        return BEVTensor(data=data, resolution=self.resolution, extent=self.extent, channels=self.channels)
