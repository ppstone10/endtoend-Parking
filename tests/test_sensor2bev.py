"""Sensor2BEV 转换测试。"""

import unittest

import numpy as np

from interfaces import LiDARFrame
from sensor2bev import LiDAR2BEV


class TestLiDAR2BEV(unittest.TestCase):
    def setUp(self):
        self.converter = LiDAR2BEV(resolution=0.2, extent=(10, 10, 10, 10), ground_z=0.1)

    def test_empty_points(self):
        frame = LiDARFrame(points=np.zeros((0, 4), dtype=np.float32))
        bev = self.converter.to_bev(frame, 0.0, 0.0, 0.0)
        self.assertEqual(bev.shape[0], 3)
        self.assertEqual((bev.data[0] == 0).all(), True)

    def test_obstacle_creates_occupancy(self):
        # 在车辆前方 5 米处放置一个点，应产生占据。
        points = np.array([[5.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        frame = LiDARFrame(points=points)
        bev = self.converter.to_bev(frame, 0.0, 0.0, 0.0)
        self.assertEqual((bev.data[0] > 0).sum(), 1)
        self.assertEqual(bev.channels, ["occupancy", "height", "density"])

    def test_ground_removed(self):
        # 地面点（z=0）应被滤除，不产生占据。
        points = np.array([[5.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        frame = LiDARFrame(points=points)
        bev = self.converter.to_bev(frame, 0.0, 0.0, 0.0)
        self.assertEqual((bev.data[0] > 0).sum(), 0)

    def test_outside_roi_removed(self):
        # ROI 之外的点（后方 20 米）应被裁剪。
        points = np.array([[-20.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        frame = LiDARFrame(points=points)
        bev = self.converter.to_bev(frame, 0.0, 0.0, 0.0)
        self.assertEqual((bev.data[0] > 0).sum(), 0)


if __name__ == "__main__":
    unittest.main()