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
class CameraView:
    """车体中心上方的单相机视角，yaw/pitch为弧度，height为米。"""

    image: np.ndarray
    intrinsics: CameraIntrinsics
    yaw: float = 0.0
    height: float = 1.5
    pitch: float = float(np.deg2rad(30.0))

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        _validate_camera_image(self.image, self.intrinsics)
        if not np.isfinite([self.yaw, self.height, self.pitch]).all() or self.height <= 0:
            raise ValueError("相机外参必须有限且height为正")


def _validate_camera_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> None:
    if intrinsics.image_width <= 0 or intrinsics.image_height <= 0:
        raise ValueError("相机图像尺寸必须为正")
    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"图像形状必须为 (H, W, C)，实际 {image.shape}")
    if image.shape[:2] != (intrinsics.image_height, intrinsics.image_width):
        raise ValueError("图像尺寸必须与相机内参标定一致")
    if not np.isfinite([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]).all():
        raise ValueError("相机内参必须有限")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0 or not np.isfinite(image).all():
        raise ValueError("相机焦距必须为正且图像必须有限")


@dataclass
class CameraFrame:
    """同步相机帧；views非空时image仅为预览，不能用于几何投影。"""

    image: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp: float = 0.0
    views: tuple[CameraView, ...] = ()

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        _validate_camera_image(self.image, self.intrinsics)
        self.views = tuple(self.views)
        if any(not isinstance(view, CameraView) for view in self.views):
            raise ValueError("views必须包含CameraView")
