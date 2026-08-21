"""阶段一数据流演示：环境 → LiDAR/Camera → Sensor2BEV → 融合 BEV。

验证双传感器 BEV 链路与融合可用。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from interfaces import CameraIntrinsics, GoalPose
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR


def main() -> None:
    env = ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-6.0, x_max=-1.0, y_min=-15.0, y_max=15.0),
            RectangleObstacle(x_min=1.0, x_max=6.0, y_min=-15.0, y_max=15.0),
        ],
        parking_spots=[GoalPose(x=6.0, y=0.0, yaw=0.0)],
    )
    lidar = SimulatedLiDAR(env, beams=360, max_range=20.0, z=1.0)
    intrinsics = CameraIntrinsics(
        fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
    )
    camera = SimulatedCamera(env, intrinsics, height=1.5, pitch=np.deg2rad(30.0))
    lidar2bev = LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))
    camera2bev = Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))
    fusion = BEVFusion()

    x, y, yaw = 0.0, -8.0, 0.0
    lidar_frame = lidar.capture(x, y, yaw)
    camera_frame = camera.capture(x, y, yaw)
    lidar_bev = lidar2bev.to_bev(lidar_frame, x, y, yaw)
    camera_bev = camera2bev.to_bev(camera_frame, x, y, yaw)
    fused_bev = fusion.fuse(lidar_bev, camera_bev)

    print(f"LiDAR 点数: {lidar_frame.count}")
    print(f"图像尺寸: {camera_frame.image.shape}, 目标像素数: {(camera_frame.image > 0).sum()}")
    print(f"LiDAR BEV: {lidar_bev.shape} 通道 {lidar_bev.channels}")
    print(f"Camera BEV: {camera_bev.shape} 通道 {camera_bev.channels}")
    print(f"融合 BEV: {fused_bev.shape} 通道 {fused_bev.channels}")
    print(f"融合后占据栅格: {int((fused_bev.data[0] > 0).sum())}")
    print("演示完成")


if __name__ == "__main__":
    main()