"""相机渲染的可见性鲁棒性测试。

回归背景：旧实现"任一角点投影出图就整帧丢弃"，导致部分角点越界时把本来可见
的目标区域也一起清空。现在只按**相机前方**角点的凸包渲染可见部分；
全部角点在相机后方（目标在车后）时正确地不渲染。
"""

from __future__ import annotations

import unittest

import numpy as np

from interfaces import CameraIntrinsics, GoalPose
from sim import ParkingEnvironment
from sim.sensor_camera import SimulatedCamera


def _camera() -> SimulatedCamera:
    return SimulatedCamera(
        ParkingEnvironment(world_size=60.0, obstacles=[]),
        CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        ),
        parking_area=(6.0, 3.0),
    )


class TestCameraVisibility(unittest.TestCase):
    def test_goal_ahead_is_rendered(self):
        camera = _camera()
        camera.env.parking_spots = [GoalPose(10.0, 0.0, 0.0)]
        image = camera.capture(0.0, 0.0, 0.0).image
        self.assertGreater(int(np.count_nonzero(image)), 0)

    def test_goal_behind_is_not_rendered(self):
        """目标在车后：全部角点深度为负 → 图像应为空（物理正确）。"""
        camera = _camera()
        camera.env.parking_spots = [GoalPose(-6.0, 0.0, 0.0)]
        image = camera.capture(0.0, 0.0, 0.0).image
        self.assertEqual(int(np.count_nonzero(image)), 0)

    def test_partially_behind_goal_renders_visible_part(self):
        """跨相机平面的目标：远处角点仍可见，应渲染裁剪后的可见部分。"""
        camera = _camera()
        camera.env.parking_spots = [GoalPose(4.0, 0.0, 0.0)]
        image = camera.capture(0.0, 0.0, 0.0).image
        # 目标中心 4m、长 6m → 近端角点在相机平面之后，远端角点可见。
        self.assertGreater(int(np.count_nonzero(image)), 0)

    def test_clip_keeps_far_half_of_straddling_rectangle(self):
        camera = _camera()
        # 局部坐标下的 6x3 矩形，中心在原点：相机平面约在 x = -0.87m。
        polygon = [
            np.array([-3.0, 1.5]),
            np.array([3.0, 1.5]),
            np.array([3.0, -1.5]),
            np.array([-3.0, -1.5]),
        ]
        clipped = camera._clip_to_camera_front(polygon)
        self.assertGreaterEqual(len(clipped), 3)
        self.assertTrue(all(pt[0] > -0.9 for pt in clipped))

    def test_clip_drops_fully_behind_polygon(self):
        camera = _camera()
        polygon = [
            np.array([-3.0, 1.5]),
            np.array([-1.5, 1.5]),
            np.array([-1.5, -1.5]),
            np.array([-3.0, -1.5]),
        ]
        self.assertEqual(camera._clip_to_camera_front(polygon), [])

    def test_no_parking_spots_renders_nothing(self):
        camera = _camera()
        camera.env.parking_spots = []
        image = camera.capture(0.0, 0.0, 0.0).image
        self.assertEqual(int(np.count_nonzero(image)), 0)

    def test_convex_hull_of_rectangle_is_itself(self):
        camera = _camera()
        rect = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        hull = camera._convex_hull(rect)
        self.assertEqual(len(hull), 4)
        self.assertEqual(set(hull), set(rect))

    def test_convex_hull_drops_interior_point(self):
        camera = _camera()
        points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]
        hull = camera._convex_hull(points)
        self.assertNotIn((1.0, 1.0), hull)
        self.assertEqual(len(hull), 4)

    def test_convex_hull_with_fewer_than_three_points(self):
        camera = _camera()
        self.assertEqual(camera._convex_hull([(0.0, 0.0), (1.0, 1.0)]), [(0.0, 0.0), (1.0, 1.0)])
        self.assertEqual(camera._convex_hull([(0.0, 0.0)]), [(0.0, 0.0)])

    def test_rendering_is_deterministic(self):
        camera = _camera()
        camera.env.parking_spots = [GoalPose(8.0, 1.0, 0.2)]
        first = camera.capture(0.0, 0.0, 0.0).image
        second = camera.capture(0.0, 0.0, 0.0).image
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
