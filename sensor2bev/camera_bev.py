"""Camera 图像到 BEV 的转换（IPM/单应变换）。

将相机图像中的地面目标（车位线、目标区域）反投影到车辆中心局部 BEV。
采用与 sim/camera_model.py 相同的单应推导，保持两个模块的投影几何一致：
    u = (fx * (-Y) + cx * zc) / zc
    v = (fy * (-sinP * X + h * cosP) + cy * zc) / zc
    zc = cosP * X + h * sinP
其中 (X, Y) 为 z=0 地面上车辆局部坐标（X 前向、Y 左向）。
"""

from __future__ import annotations

import numpy as np

from interfaces import BEVConfig, BEVTensor, CameraFrame

CH_TARGET = "target"


class Camera2BEV:
    """将相机图像转换为目标区域 BEV 通道。

    height 为相机离地高度（米），pitch 为向下俯仰角（弧度），
    threshold 为像素灰度阈值，高于阈值视为目标区域。
    """

    def __init__(
        self,
        resolution: float | None = None,
        extent: tuple[float, float, float, float] | None = None,
        height: float = 1.5,
        pitch: float = np.deg2rad(30.0),
        threshold: float = 100.0,
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
        self.height = height
        self.pitch = pitch
        self.threshold = threshold
        self.channels = [CH_TARGET]

    def _homography(self, intrinsics) -> np.ndarray:
        """构建地面→像素 3x3 单应矩阵（与 sim/camera_model.py 一致）。"""
        cos_p, sin_p = np.cos(self.pitch), np.sin(self.pitch)
        h = self.height
        k = intrinsics
        return np.array(
            [
                [k.cx * cos_p, -k.fx, k.cx * h * sin_p],
                [k.cy * cos_p - k.fy * sin_p, 0.0, k.cy * h * sin_p + k.fy * h * cos_p],
                [cos_p, 0.0, h * sin_p],
            ],
            dtype=np.float64,
        )

    def to_bev(self, frame: CameraFrame, x: float, y: float, yaw: float) -> BEVTensor:
        """将相机图像反投影到车辆中心局部 BEV，输出目标区域通道。"""
        hmat = self._homography(frame.intrinsics)
        image = frame.image
        if image.ndim == 3:
            gray = image.mean(axis=-1)
        else:
            gray = image

        front, back, left, right = self.extent
        h = round((front + back) / self.resolution)
        w = round((left + right) / self.resolution)
        target = np.zeros((h, w), dtype=np.float32)

        # 逐栅格中心前向投影到像素并采样灰度。
        rows = np.arange(h, dtype=np.float64)
        cols = np.arange(w, dtype=np.float64)
        centers_x = front - (rows + 0.5) * self.resolution  # 行 → 前向距离
        centers_y = -right + (cols + 0.5) * self.resolution  # 列 → 左向距离
        grid_x, grid_y = np.meshgrid(centers_x, centers_y, indexing="ij")
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        ones = np.ones_like(flat_x)
        hom = hmat @ np.stack([flat_x, flat_y, ones], axis=0)
        depth = hom[2]
        valid_depth = depth > 0.0
        u = hom[0] / np.maximum(depth, 1e-12)
        v = hom[1] / np.maximum(depth, 1e-12)

        img_h, img_w = gray.shape
        inside = (
            valid_depth
            & (u >= 0)
            & (u < img_w)
            & (v >= 0)
            & (v < img_h)
        )
        u_int = u[inside].astype(int)
        v_int = v[inside].astype(int)
        target.ravel()[np.flatnonzero(inside)] = (
            gray[v_int, u_int] > self.threshold
        ).astype(np.float32)

        return BEVTensor(
            data=target[None, :, :],
            resolution=self.resolution,
            extent=self.extent,
            channels=self.channels,
        )
