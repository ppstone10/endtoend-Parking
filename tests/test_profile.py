"""梯形速度剖面测试。"""

import unittest

import numpy as np

from planner.profile import trapezoidal_velocity_profile


class TestVelocityProfile(unittest.TestCase):
    def test_forward_profile_respects_speed_and_acceleration_limits(self):
        points = np.column_stack([np.arange(11.0), np.zeros(11), np.zeros(11)])
        profile = trapezoidal_velocity_profile(
            points, max_speed=2.0, reverse_speed=0.8, acceleration=1.0, deceleration=1.0
        )
        self.assertAlmostEqual(float(profile.speeds[0]), 0.0)
        self.assertAlmostEqual(float(profile.speeds[-1]), 0.0)
        self.assertLessEqual(float(np.max(profile.speeds)), 2.0)
        self.assertGreater(float(np.max(profile.speeds)), 1.0)
        self.assertTrue(np.all(np.diff(profile.times) > 0.0))
        ds = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        self.assertTrue(np.all(profile.speeds[1:] ** 2 - profile.speeds[:-1] ** 2 <= 2.0 * ds + 1e-8))

    def test_reverse_segments_use_lower_signed_speed(self):
        points = np.column_stack([np.arange(0.0, -6.0, -1.0), np.zeros(6), np.zeros(6)])
        profile = trapezoidal_velocity_profile(
            points, max_speed=2.0, reverse_speed=0.6, acceleration=1.0, deceleration=1.0
        )
        self.assertTrue(np.all(profile.speeds[1:-1] < 0.0))
        self.assertLessEqual(float(np.max(np.abs(profile.speeds))), 0.6 + 1e-9)

    def test_direction_change_forces_stop(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        profile = trapezoidal_velocity_profile(
            points, max_speed=1.0, reverse_speed=0.5, acceleration=1.0, deceleration=1.0
        )
        self.assertAlmostEqual(float(profile.speeds[2]), 0.0)
        self.assertTrue(np.all(np.diff(profile.times) > 0.0))
        self.assertGreater(profile.speeds[1], 0.0)
        self.assertLess(profile.speeds[3], 0.0)

    def test_invalid_limits_raise(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            trapezoidal_velocity_profile(points, max_speed=0.0)

    def test_in_place_rotation_stops_and_uses_angular_duration(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.2],
                [1.0, 0.0, 0.4],
                [2.0, 0.0, 0.4],
            ]
        )
        profile = trapezoidal_velocity_profile(points, max_omega=0.2)

        np.testing.assert_allclose(profile.speeds[1:4], 0.0, atol=1e-12)
        self.assertAlmostEqual(profile.times[2] - profile.times[1], 1.0)
        self.assertAlmostEqual(profile.times[3] - profile.times[2], 1.0)


if __name__ == "__main__":
    unittest.main()
