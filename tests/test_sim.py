"""差分驱动模型与环境测试。"""

import unittest

import numpy as np

from interfaces import ControlCmd, GoalPose, VehicleState
from sim import ParkingEnvironment, RectangleObstacle, SimulatedLiDAR
from sim.vehicle_model import DifferentialDriveModel


class TestDifferentialDriveModel(unittest.TestCase):
    def setUp(self):
        self.model = DifferentialDriveModel(max_v=2.0, max_omega=1.0)

    def test_forward_motion(self):
        state = VehicleState(0.0, 0.0, 0.0)
        new_state = self.model.step(state, ControlCmd(1.0, 0.0), dt=1.0)
        self.assertAlmostEqual(new_state.x, 1.0)
        self.assertAlmostEqual(new_state.y, 0.0)

    def test_rotation(self):
        state = VehicleState(0.0, 0.0, 0.0)
        new_state = self.model.step(state, ControlCmd(0.0, 0.5), dt=1.0)
        self.assertAlmostEqual(new_state.yaw, 0.5)

    def test_speed_limit(self):
        state = VehicleState(0.0, 0.0, 0.0)
        new_state = self.model.step(state, ControlCmd(10.0, 0.0), dt=1.0)
        self.assertLessEqual(new_state.v, self.model.max_v)


class TestParkingEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = ParkingEnvironment(
            world_size=40.0,
            obstacles=[
                RectangleObstacle(x_min=-6.0, x_max=-1.0, y_min=-15.0, y_max=15.0)
            ],
        )

    def test_free_inside(self):
        self.assertTrue(self.env.is_free(10.0, 10.0))

    def test_occupied(self):
        self.assertFalse(self.env.is_free(-5.0, 0.0))

    def test_outside_boundary(self):
        self.assertFalse(self.env.is_free(21.0, 0.0))

    def test_raycast_hit_obstacle(self):
        # 障碍物位于车辆 x 负方向，射线朝 -x 应命中。
        dist = self.env.raycast(np.array([0.0, 0.0]), np.pi, 20.0)
        self.assertLess(dist, 20.0)


class TestSimulatedLiDAR(unittest.TestCase):
    def test_capture_shape(self):
        env = ParkingEnvironment(
            world_size=40.0,
            obstacles=[RectangleObstacle(x_min=-6.0, x_max=-1.0, y_min=-15.0, y_max=15.0)],
        )
        lidar = SimulatedLiDAR(env, beams=360, max_range=20.0)
        frame = lidar.capture(0.0, 0.0, 0.0)
        self.assertEqual(frame.points.shape, (360, 4))
        self.assertTrue((frame.points[:, 3] >= 0).all())


if __name__ == "__main__":
    unittest.main()