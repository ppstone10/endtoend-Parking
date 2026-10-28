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

from interfaces import BEVConfig, BEVTensor, CameraFrame, CameraView

from interfaces.bev import CAMERA_PROJECTION_CURRENT

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

    def _homography(self, intrinsics, *, height=None, pitch=None) -> np.ndarray:
        """构建地面→像素 3x3 单应矩阵（与 sim/camera_model.py 一致）。"""
        pitch = self.pitch if pitch is None else pitch
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        h = self.height if height is None else height
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
        front, back, left, right = self.extent
        h, w = self.config.shape
        rows, cols = np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(front-(rows+.5)*self.resolution,
                                    -right+(cols+.5)*self.resolution, indexing="ij")
        target = np.zeros((h, w), dtype=np.float32)
        views = frame.views or (CameraView(frame.image, frame.intrinsics,
                                          height=self.height, pitch=self.pitch),)
        for view in views:
            # BEV格心先由车体系转到该相机局部地面系，再采样该视角图像。
            c, s = np.cos(view.yaw), np.sin(view.yaw)
            local_x, local_y = c*grid_x+s*grid_y, -s*grid_x+c*grid_y
            hom = self._homography(view.intrinsics, height=view.height, pitch=view.pitch) @ np.stack(
                [local_x.ravel(), local_y.ravel(), np.ones(h*w)], axis=0)
            depth = hom[2]
            valid = depth > 1e-3
            u = np.divide(hom[0], depth, out=np.full(h*w, -1.), where=valid)
            v = np.divide(hom[1], depth, out=np.full(h*w, -1.), where=valid)
            gray = view.image.mean(axis=-1)
            img_h, img_w = gray.shape
            inside = valid & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            pixels = gray[v[inside].astype(int), u[inside].astype(int)] > self.threshold
            # 多视角只在同一车体BEV格上融合，重叠不会重复累计强度。
            indices = np.flatnonzero(inside)
            target.ravel()[indices] = np.maximum(target.ravel()[indices], pixels)
        return BEVTensor(data=target[None], resolution=self.resolution, extent=self.extent,
                         channels=self.channels,
                         camera_projection=CAMERA_PROJECTION_CURRENT if frame.views else None)
