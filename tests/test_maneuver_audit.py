"""任务机动要求与专家轨迹方向一致性测试。"""

import unittest

import numpy as np

from dataset.maneuver import audit_maneuver_consistency
from sim.tasks import Maneuver


class TestManeuverAudit(unittest.TestCase):
    def test_requested_direction_must_cover_at_least_half_distance(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        forward = audit_maneuver_consistency(points, Maneuver.FORWARD)
        reverse = audit_maneuver_consistency(points, Maneuver.REVERSE)

        self.assertTrue(forward.consistent)
        self.assertAlmostEqual(forward.requested_distance_ratio, 0.8)
        self.assertFalse(reverse.consistent)
        self.assertAlmostEqual(reverse.requested_distance_ratio, 0.2)
        self.assertEqual(forward.direction_changes, 1)

    def test_equal_forward_and_reverse_distance_is_consistent_for_either_request(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        self.assertTrue(
            audit_maneuver_consistency(points, Maneuver.FORWARD).consistent
        )
        self.assertTrue(
            audit_maneuver_consistency(points, Maneuver.REVERSE).consistent
        )

    def test_audit_metadata_is_json_ready_and_records_policy(self):
        audit = audit_maneuver_consistency(
            np.array([[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
            Maneuver.REVERSE,
        )
        metadata = audit.to_metadata()
        self.assertEqual(metadata["requested_maneuver"], "reverse")
        self.assertEqual(metadata["minimum_requested_distance_ratio"], 0.5)
        self.assertEqual(metadata["requested_distance_ratio"], 1.0)
        self.assertTrue(metadata["consistent"])

    def test_invalid_or_stationary_trajectory_is_rejected(self):
        invalid_cases = (
            np.zeros((1, 3)),
            np.zeros((2, 2)),
            np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
            np.zeros((2, 3)),
        )
        for points in invalid_cases:
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    audit_maneuver_consistency(points, Maneuver.FORWARD)
        with self.assertRaisesRegex(ValueError, r"\[0.5, 1.0\]"):
            audit_maneuver_consistency(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                Maneuver.FORWARD,
                minimum_requested_distance_ratio=0.49,
            )


if __name__ == "__main__":
    unittest.main()
