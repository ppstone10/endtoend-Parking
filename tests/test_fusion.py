"""Camera→BEV 与融合测试。"""

import unittest

import numpy as np

from interfaces import BEVConfig, CameraIntrinsics, GoalPose
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR


def _setup_env(parking_x: float = 5.0, parking_y: float = 0.0):
    """返回泊车位位于车辆正前方视野内的环境。

    车辆默认在原点 (0,0)，泊车位 (parking_x, parking_y) 正前方，侧墙平行 x 轴。
    """
    return ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=-6.0, y_max=-2.0),
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=2.0, y_max=6.0),
        ],
        parking_spots=[GoalPose(x=parking_x, y=parking_y, yaw=0.0)],
    )


def _setup_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
    )


class TestSimulatedCamera(unittest.TestCase):
    def setUp(self):
        self.env = _setup_env()
        self.intrinsics = _setup_intrinsics()
        self.camera = SimulatedCamera(self.env, self.intrinsics)

    def test_capture_shape(self):
        frame = self.camera.capture(0.0, 0.0, 0.0)
        self.assertEqual(frame.image.shape, (480, 640, 1))
        # 泊车位在车辆正前方视野内，应存在目标像素。
        self.assertGreater((frame.image > 0).sum(), 0)

    def test_out_of_fov_no_wide_fill(self):
        # 泊车位在车辆左后方（相机视野外），不应产生横跨全图的错误填充。
        env = _setup_env(parking_x=-5.0, parking_y=-8.0)
        camera = SimulatedCamera(env, self.intrinsics)
        frame = camera.capture(0.0, 0.0, 0.0)
        white = frame.image[:, :, 0] > 0
        # 视野外目标要么完全不可见，要么仅出现在图像边缘小范围，不得铺满全宽。
        self.assertLess((white).sum(), 50)


class TestCamera2BEV(unittest.TestCase):
    def setUp(self):
        self.env = _setup_env()
        self.intrinsics = _setup_intrinsics()
        self.camera = SimulatedCamera(self.env, self.intrinsics)
        self.converter = Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))

    def test_target_channel(self):
        frame = self.camera.capture(0.0, 0.0, 0.0)
        bev = self.converter.to_bev(frame, 0.0, 0.0, 0.0)
        self.assertEqual(bev.shape[0], 1)
        self.assertEqual(bev.channels, ["target"])
        target = bev.data[0]
        self.assertGreater((target > 0).sum(), 0)

    def test_roundtrip_position(self):
        # 往返一致性：泊车位正前方 (5,0)，target BEV 应落回前方约 5m、中线附近。
        frame = self.camera.capture(0.0, 0.0, 0.0)
        bev = self.converter.to_bev(frame, 0.0, 0.0, 0.0)
        target = bev.data[0]
        rows, cols = np.where(target > 0)
        res = 0.2
        front = 10.0
        X = front - (rows + 0.5) * res
        Y = -10 + (cols + 0.5) * res
        self.assertTrue(np.any(X > 2.0), "目标应出现在前方 2m 之外")
        # 目标区域沿中线分布，Y 范围应较窄。
        self.assertLess(np.abs(Y).max(), 3.0)


class TestBEVFusion(unittest.TestCase):
    def setUp(self):
        self.env = _setup_env()
        self.intrinsics = _setup_intrinsics()
        self.lidar = SimulatedLiDAR(self.env, beams=360, max_range=20.0)
        self.camera = SimulatedCamera(self.env, self.intrinsics)
        self.lidar2bev = LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))
        self.camera2bev = Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0))
        self.fusion = BEVFusion()

    def test_fuse_channels(self):
        x, y, yaw = 0.0, 0.0, 0.0
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

    def test_shared_config_drives_both_converters(self):
        config = BEVConfig()
        lidar = LiDAR2BEV(config=config)
        camera = Camera2BEV(config=config)
        self.assertIs(lidar.config, config)
        self.assertIs(camera.config, config)
        self.assertEqual(lidar.resolution, 0.25)
        self.assertEqual(lidar.extent, (20.0, 20.0, 20.0, 20.0))

    def test_config_cannot_be_mixed_with_legacy_spatial_arguments(self):
        with self.assertRaises(ValueError):
            LiDAR2BEV(resolution=0.2, config=BEVConfig())


if __name__ == "__main__":
    unittest.main()
