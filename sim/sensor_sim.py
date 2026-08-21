"""模拟 LiDAR 传感器。

通过环境射线投射生成 LiDARFrame。车辆位于 (x, y, yaw)，以 yaw 为前向。
"""

from __future__ import annotations

import numpy as np

from interfaces import LiDARFrame
from .environment import ParkingEnvironment


class SimulatedLiDAR:
    """基于环境射线投射的模拟 LiDAR。

    beams 为每圈射线数，max_range 为最大量程（米），z 为安装高度（米）。
    """

    def __init__(
        self, env: ParkingEnvironment, beams: int = 360, max_range: float = 20.0, z: float = 1.0
    ) -> None:
        self.env = env
        self.beams = beams
        self.max_range = max_range
        self.z = z

    def capture(self, x: float, y: float, yaw: float) -> LiDARFrame:
        """采集一帧点云。

        每个波束产生一个点 [x, y, z, intensity]，命中障碍物时 intensity 为 1，未命中为 0。
        """
        origin = np.array([x, y])
        angles = np.linspace(0.0, 2.0 * np.pi, self.beams, endpoint=False) + yaw
        points = np.empty((self.beams, 4), dtype=np.float32)
        for i, angle in enumerate(angles):
            dist = self.env.raycast(origin, float(angle), self.max_range)
            hit = dist < self.max_range
            points[i, 0] = x + dist * np.cos(angle)
            points[i, 1] = y + dist * np.sin(angle)
            points[i, 2] = self.z
            points[i, 3] = 1.0 if hit else 0.0
        return LiDARFrame(points=points)