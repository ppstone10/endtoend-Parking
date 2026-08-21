"""针孔相机模型（地面平面 ↔ 图像平面）。

提供从车辆局部地面坐标到图像像素的单应变换，用于模拟相机渲染。
单应 H 满足 [u, v, 1]^T ~ H * [X, Y, 1]^T，其中 (X, Y) 为 z=0 地面上的
车辆局部坐标（X 前向、Y 左向）。

推导（相机位于车辆中心上方 h 米，向下俯仰 pitch 角，朝向与车辆 yaw 一致）：
  u = (fx * (-Y) + cx * zc) / zc
  v = (fy * (-sinP * X + h * cosP) + cy * zc) / zc
  zc = cosP * X + h * sinP
"""

from __future__ import annotations

import numpy as np

from interfaces import CameraIntrinsics


class CameraModel:
    """针孔相机模型。

    intrinsics 为相机内参，height 为相机离地高度（米），
    pitch 为向下俯仰角（弧度，>0 表示朝前下方看）。
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        height: float = 1.5,
        pitch: float = np.deg2rad(30.0),
    ) -> None:
        self.intrinsics = intrinsics
        self.height = height
        self.pitch = pitch

    def homography(self) -> np.ndarray:
        """返回地面→像素的 3x3 单应矩阵。"""
        k = self.intrinsics
        cos_p, sin_p = np.cos(self.pitch), np.sin(self.pitch)
        h = self.height
        return np.array(
            [
                [k.cx * cos_p, -k.fx, k.cx * h * sin_p],
                [k.cy * cos_p - k.fy * sin_p, 0.0, k.cy * h * sin_p + k.fy * h * cos_p],
                [cos_p, 0.0, h * sin_p],
            ],
            dtype=np.float64,
        )

    def project(self, X: float, Y: float) -> tuple[float, float] | None:
        """将地面点 (X, Y) 投影到像素坐标。

        点位于相机前方（深度 > 0）时返回 (u, v)，否则返回 None。
        """
        hmat = self.homography()
        p = hmat @ np.array([X, Y, 1.0], dtype=np.float64)
        if p[2] <= 0.0:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])