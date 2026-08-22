"""数据集生成测试。"""

import unittest

import numpy as np

from dataset import DatasetGenerator, SensorBEVPipeline
from interfaces import CameraIntrinsics, GoalPose, VehicleState
from planner import HybridAStarPlanner
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR


def _build_generator(seed: int = 0) -> DatasetGenerator:
    env = ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=-6.0, y_max=-2.0),
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=2.0, y_max=6.0),
        ],
    )
    planner = HybridAStarPlanner(env=env)
    intrinsics = CameraIntrinsics(
        fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
    )
    pipeline = SensorBEVPipeline(
        lidar_sensor=SimulatedLiDAR(env, beams=360, max_range=20.0),
        camera_sensor=SimulatedCamera(env, intrinsics),
        lidar2bev=LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
        camera2bev=Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
        bev_fusion=BEVFusion(),
    )
    return DatasetGenerator(env=env, planner=planner, sensor_pipeline=pipeline, seed=seed)


class TestDatasetGenerator(unittest.TestCase):
    def test_generate_samples(self):
        generator = _build_generator(seed=1)
        samples = generator.generate(count=3)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertEqual(sample.bev.shape[0], 5)  # 融合 BEV 5 通道
            self.assertEqual(sample.bev.channels[-1], "vehicle")
            self.assertGreater(sample.expert_trajectory.horizon, 2)
            self.assertIsInstance(sample.goal, GoalPose)
            self.assertIsInstance(sample.state, VehicleState)

    def test_sample_poses_free(self):
        generator = _build_generator(seed=2)
        samples = generator.generate(count=2)
        env = generator.env
        for sample in samples:
            # 起始与目标车辆矩形中心应在自由空间。
            self.assertTrue(env.is_free(sample.state.x, sample.state.y))
            self.assertTrue(env.is_free(sample.goal.x, sample.goal.y))
            # 专家轨迹各点自由。
            for px, py in sample.expert_trajectory.points[:, :2]:
                self.assertTrue(env.is_free(float(px), float(py)))


if __name__ == "__main__":
    unittest.main()