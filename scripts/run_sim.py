"""阶段一数据流演示：仿真环境 → LiDAR → Sensor2BEV → BEV。

不依赖网络与控制器，验证传感器与 BEV 链路可用。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from interfaces import GoalPose
from sensor2bev import LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedLiDAR


def main() -> None:
    env = ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-6.0, x_max=-1.0, y_min=-15.0, y_max=15.0),
            RectangleObstacle(x_min=1.0, x_max=6.0, y_min=-15.0, y_max=15.0),
        ],
        parking_spots=[GoalPose(x=0.0, y=8.0, yaw=0.0)],
    )
    lidar = SimulatedLiDAR(env, beams=360, max_range=20.0, z=1.0)
    bev_converter = LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))

    x, y, yaw = 0.0, -8.0, 0.0
    frame = lidar.capture(x, y, yaw)
    bev = bev_converter.to_bev(frame, x, y, yaw)

    print(f"LiDAR 点数: {frame.count}")
    print(f"BEV 形状: {bev.shape}, 通道: {bev.channels}")
    print(f"占据栅格数: {int((bev.data[0] > 0).sum())}")
    print("演示完成")


if __name__ == "__main__":
    main()