"""统一接口契约测试。"""

import unittest

import numpy as np

from interfaces import (
    BEVTensor,
    ControlCmd,
    GoalPose,
    LiDARFrame,
    Trajectory,
    VehicleState,
)


class TestLiDARFrame(unittest.TestCase):
    def test_shape_validation(self):
        good = np.zeros((10, 4), dtype=np.float32)
        frame = LiDARFrame(points=good)
        self.assertEqual(frame.count, 10)

        with self.assertRaises(ValueError):
            LiDARFrame(points=np.zeros((10, 3)))


class TestBEVTensor(unittest.TestCase):
    def test_valid_tensor(self):
        data = np.zeros((3, 100, 100), dtype=np.float32)
        bev = BEVTensor(data=data, resolution=0.2, extent=(10, 10, 10, 10), channels=["a", "b", "c"])
        self.assertEqual(bev.shape, (3, 100, 100))

    def test_channel_mismatch(self):
        data = np.zeros((3, 100, 100), dtype=np.float32)
        with self.assertRaises(ValueError):
            BEVTensor(data=data, resolution=0.2, extent=(10, 10, 10, 10), channels=["a"])

    def test_size_mismatch(self):
        data = np.zeros((3, 50, 50), dtype=np.float32)
        with self.assertRaises(ValueError):
            BEVTensor(data=data, resolution=0.2, extent=(10, 10, 10, 10), channels=["a", "b", "c"])


class TestVehicleState(unittest.TestCase):
    def test_roundtrip(self):
        state = VehicleState(1.0, 2.0, 0.5, 0.3, 0.1)
        restored = VehicleState.from_array(state.to_array())
        self.assertAlmostEqual(restored.x, state.x)
        self.assertAlmostEqual(restored.omega, state.omega)


class TestTrajectory(unittest.TestCase):
    def test_horizon(self):
        traj = Trajectory(points=np.zeros((20, 3)), dt=0.1)
        self.assertEqual(traj.horizon, 20)


class TestControlCmd(unittest.TestCase):
    def test_to_array(self):
        cmd = ControlCmd(1.0, 0.5)
        np.testing.assert_array_equal(cmd.to_array(), np.array([1.0, 0.5]))


if __name__ == "__main__":
    unittest.main()