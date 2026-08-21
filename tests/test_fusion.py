"""Camera→BEV 与融合测试。"""

import unittest

import numpy as np

from interfaces import CameraIntrinsics, GoalPose
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR


def _setup_env():
    return ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-6.0, x_max=-1.0, y_min=-15.0, y_max=15.0),
            RectangleObstacle(x_min=1.0, x_max=6.0, y_min=-15.0, y_max=15.0),
        ],
        parking_spots=[GoalPose(x=6.0, y=0.0, yaw=0.0)],
    )


class TestSimulatedCamera(unittest.TestCase):
    def setUp(self):
        self.env = _setup_env()
        self.intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        )
        self.camera = SimulatedCamera(self.env, self.intrinsics)

    def test_capture_shape(self):
        frame = self.camera.capture(0.0, -8.0, 0.0)
        self.assertEqual(frame.image.shape, (480, 640, 1))
        # 泊车位在车辆前方视野内，应存在目标像素。
        self.assertGreater((frame.image > 0).sum(), 0)


class TestCamera2BEV(unittest.TestCase):
    def setUp(self):
        self.env = _setup_env()
        self.intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        )
        self.camera = SimulatedCamera(self.env, self.intrinsics)
        self.converter = Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))

    def test_target_channel(self):
        frame = self.camera.capture(0.0, -8.0, 0.0)
        bev = self.converter.to_bev(frame, 0.0, -8.0, 0.0)
        self.assertEqual(bev.shape[0], 1)
        self.assertEqual(bev.channels, ["target"])
        # 泊车位在车辆前方，目标区域应出现在 BEV 前向半区。
        target = bev.data[0]
        front_half = target[: target.shape[0] // 2]
        self.assertGreater((front_half > 0).sum(), 0)


class TestBEVFusion(unittest.TestCase):
    def setUp(self):
        self.env = _setup_env()
        self.intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        )
        self.lidar = SimulatedLiDAR(self.env, beams=360, max_range=20.0)
        self.camera = SimulatedCamera(self.env, self.intrinsics)
        self.lidar2bev = LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))
        self.camera2bev = Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))
        self.fusion = BEVFusion()

    def test_fuse_channels(self):
        x, y, yaw = 0.0, -8.0, 0.0
        lidar_bev = self.lidar2bev.to_bev(self.lidar.capture(x, y, yaw), x, y, yaw)
        camera_bev = self.camera2bev.to_bev(self.camera.capture(x, y, yaw), x, y, yaw)
        fused = self.fusion.fuse(lidar_bev, camera_bev)
        self.assertEqual(fused.channels, ["occupancy", "height", "density", "target", "vehicle"])
        self.assertEqual(fused.shape[0], 5)
        # 车辆轮廓通道中心应有占据。
        self.assertGreater((fused.data[4] > 0).sum(), 0)

    def test_fuse_mismatch_raises(self):
        lidar_bev = LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)).to_bev(
            self.lidar.capture(0.0, 0.0, 0.0), 0.0, 0.0, 0.0
        )
        camera_bev = Camera2BEV(resolution=0.5, extent=(10.0, 10.0, 10.0, 10.0)).to_bev(
            self.camera.capture(0.0, 0.0, 0.0), 0.0, 0.0, 0.0
        )
        with self.assertRaises(ValueError):
            self.fusion.fuse(lidar_bev, camera_bev)


if __name__ == "__main__":
    unittest.main()