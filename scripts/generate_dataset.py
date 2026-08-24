"""阶段三演示：生成小批量训练样本并打印统计。

用法：
    python scripts/generate_dataset.py [count]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator, SensorBEVPipeline
from interfaces import BEVConfig, CameraIntrinsics
from planner import HybridAStarPlanner
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR


def build_pipeline(env, bev_config: BEVConfig | None = None) -> SensorBEVPipeline:
    bev_config = bev_config or BEVConfig()
    intrinsics = CameraIntrinsics(
        fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
    )
    return SensorBEVPipeline(
        lidar_sensor=SimulatedLiDAR(env, beams=360, max_range=20.0),
        camera_sensor=SimulatedCamera(env, intrinsics),
        lidar2bev=LiDAR2BEV(config=bev_config),
        camera2bev=Camera2BEV(config=bev_config),
        bev_fusion=BEVFusion(),
    )


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    env = ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=-6.0, y_max=-2.0),
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=2.0, y_max=6.0),
        ],
    )
    planner = HybridAStarPlanner(env=env)
    generator = DatasetGenerator(env=env, planner=planner, sensor_pipeline=build_pipeline(env))
    samples = generator.generate(count=count)

    print(f"生成 {len(samples)} 条训练样本")
    for i, sample in enumerate(samples):
        traj = sample.expert_trajectory
        length = float(np.sum(np.linalg.norm(np.diff(traj.points[:, :2], axis=0), axis=1)))
        print(
            f"  [{i}] start=({sample.state.x:.1f},{sample.state.y:.1f},{np.degrees(sample.state.yaw):+.0f}°) "
            f"goal=({sample.goal.x:.1f},{sample.goal.y:.1f},{np.degrees(sample.goal.yaw):+.0f}°) "
            f"轨迹点={traj.horizon} 长度={length:.1f}m "
            f"BEV={sample.bev.shape} 通道={sample.bev.channels}"
        )


if __name__ == "__main__":
    main()
