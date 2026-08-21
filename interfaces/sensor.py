"""传感器帧与标定参数定义。

坐标约定：点云与相机位姿均使用车辆中心局部坐标系。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LiDARFrame:
    """单帧 LiDAR 点云。

    points 形状为 (N, 4)，列为 [x, y, z, intensity]，单位米，车辆中心局部系。
    """

    points: np.ndarray
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 4:
            raise ValueError(f"LiDAR 点云形状必须为 (N, 4)，实际 {self.points.shape}")

    @property
    def count(self) -> int:
        return self.points.shape[0]


@dataclass
class CameraIntrinsics:
    """相机内参。

    仅保留 IPM/单应变换所需的最小参数，完整畸变模型留待阶段二细化。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


@dataclass
class CameraFrame:
    """单帧相机图像。"""

    image: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        if self.image.ndim != 3 or self.image.shape[-1] not in (1, 3):
            raise ValueError(f"图像形状必须为 (H, W, C)，实际 {self.image.shape}")