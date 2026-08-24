"""Reeds–Shepp 48 词族与几何采样测试。"""

import unittest

import numpy as np

from planner.reeds_shepp import REEDS_SHEPP_WORDS, reeds_shepp_paths


def _angle_error(lhs: float, rhs: float) -> float:
    return float(abs((lhs - rhs + np.pi) % (2.0 * np.pi) - np.pi))


class TestReedsShepp(unittest.TestCase):
    def test_word_table_contains_48_unique_families(self):
        self.assertEqual(len(REEDS_SHEPP_WORDS), 48)
        self.assertEqual(len({word.name for word in REEDS_SHEPP_WORDS}), 48)

    def test_candidates_reach_exact_pose_with_bounded_sampling_step(self):
        start = (-1.0, -2.0, np.deg2rad(-20.0))
        goal = (4.0, 3.0, np.deg2rad(70.0))
        step_size = 0.1
        paths = reeds_shepp_paths(start, goal, turning_radius=0.625)
        self.assertGreater(len(paths), 0)
        self.assertEqual(paths, sorted(paths, key=lambda path: path.total_length))

        points, directions = paths[0].sample(start, step_size=step_size)
        self.assertLess(float(np.hypot(*(points[-1, :2] - goal[:2]))), 1e-6)
        self.assertLess(_angle_error(float(points[-1, 2]), goal[2]), 1e-6)
        self.assertLessEqual(float(np.max(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))), step_size + 1e-9)
        self.assertEqual(points.shape[0], directions.shape[0])
        self.assertTrue(set(np.unique(directions)).issubset({-1, 1}))

    def test_reverse_goal_uses_shortest_reverse_straight(self):
        paths = reeds_shepp_paths((0.0, 0.0, 0.0), (-4.0, 0.0, 0.0), turning_radius=1.0)
        points, directions = paths[0].sample((0.0, 0.0, 0.0), step_size=0.1)
        self.assertAlmostEqual(paths[0].total_length, 4.0, places=8)
        self.assertTrue(np.all(directions[1:] == -1))
        np.testing.assert_allclose(points[-1], [-4.0, 0.0, 0.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
