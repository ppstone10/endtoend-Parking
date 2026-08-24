"""专家轨迹三次捷径平滑测试。"""

import unittest

import numpy as np

from interfaces import Trajectory
from planner.smoothing import path_length, smooth_trajectory


class TestTrajectorySmoothing(unittest.TestCase):
    def test_open_space_shortcut_is_safe_and_not_longer(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.7, 0.35], [2.0, -0.7, -0.35], [3.0, 0.0, 0.0]],
            dtype=float,
        )
        trajectory = Trajectory(points=points, dt=0.2)
        checked: list[tuple[float, float, float]] = []

        def pose_free(x: float, y: float, yaw: float) -> bool:
            checked.append((x, y, yaw))
            return True

        smoothed = smooth_trajectory(
            trajectory,
            pose_free,
            step_size=0.05,
            max_curvature=4.0,
            attempts=40,
            seed=3,
        )
        self.assertGreater(len(checked), 0)
        np.testing.assert_allclose(smoothed.points[0], trajectory.points[0])
        np.testing.assert_allclose(smoothed.points[-1], trajectory.points[-1])
        self.assertLessEqual(path_length(smoothed.points), path_length(trajectory.points) + 1e-6)

    def test_rejected_shortcuts_leave_original_unchanged(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.4, 0.2], [2.0, 0.0, 0.0]])
        trajectory = Trajectory(points=points, dt=0.1)
        smoothed = smooth_trajectory(trajectory, lambda _x, _y, _yaw: False, attempts=20, seed=1)
        np.testing.assert_allclose(smoothed.points, trajectory.points)
        self.assertIsNot(smoothed, trajectory)

    def test_shortcut_does_not_remove_direction_change(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        trajectory = Trajectory(points=points, dt=0.1)
        smoothed = smooth_trajectory(trajectory, lambda _x, _y, _yaw: True, attempts=60, seed=7)
        self.assertTrue(np.any(np.all(np.isclose(smoothed.points[:, :2], [2.0, 0.0]), axis=1)))


if __name__ == "__main__":
    unittest.main()
