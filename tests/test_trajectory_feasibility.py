"""履带钻机专家轨迹运动学与碰撞可行性审计测试。"""

import unittest

import numpy as np

from dataset.feasibility import (
    audit_trajectory_feasibility,
    require_trajectory_feasibility,
    summarize_trajectory_feasibility,
)
from sim import MINING_DRILL_RIG


class TestTrajectoryFeasibility(unittest.TestCase):
    def test_forward_and_fixed_center_pivot_are_feasible(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.07]]
        )
        audit = audit_trajectory_feasibility(
            points,
            dt=0.2,
            max_v=0.5,
            max_omega=0.35,
            pose_free=lambda _x, _y, _yaw: True,
            model_metadata=MINING_DRILL_RIG.to_metadata(),
        )

        self.assertTrue(audit.feasible)
        self.assertEqual(audit.pivot_segment_count, 1)
        self.assertEqual(audit.moving_segment_count, 1)
        self.assertAlmostEqual(audit.max_linear_speed_mps, 0.5)
        self.assertAlmostEqual(audit.max_angular_speed_radps, 0.35)

    def test_lateral_translation_or_collision_is_rejected(self):
        lateral = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
        lateral_audit = audit_trajectory_feasibility(
            lateral,
            dt=0.2,
            max_v=1.0,
            max_omega=1.0,
            pose_free=lambda _x, _y, _yaw: True,
            model_metadata=MINING_DRILL_RIG.to_metadata(),
        )
        collision_audit = audit_trajectory_feasibility(
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            dt=0.2,
            max_v=1.0,
            max_omega=1.0,
            pose_free=lambda x, _y, _yaw: x < 0.05,
            model_metadata=MINING_DRILL_RIG.to_metadata(),
        )

        self.assertFalse(lateral_audit.kinematically_feasible)
        self.assertFalse(collision_audit.collision_free)
        self.assertFalse(collision_audit.feasible)

    def test_archive_summary_requires_current_model_and_collision_evidence(self):
        points = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
        masks = np.ones((1, 2), dtype=np.float32)
        metadata = [
            {
                "dataset": {
                    "vehicle_model": MINING_DRILL_RIG.to_metadata(),
                    "feasibility_audit": {
                        "collision_free": True,
                        "feasible": True,
                    },
                }
            }
        ]
        summary = summarize_trajectory_feasibility(
            points,
            masks,
            dt=0.2,
            metadata=metadata,
            vehicle_config=MINING_DRILL_RIG,
        )
        require_trajectory_feasibility({"trajectory_feasibility": summary})
        self.assertEqual(summary["feasible_count"], 1)

        metadata[0]["dataset"].pop("feasibility_audit")
        rejected = summarize_trajectory_feasibility(
            points,
            masks,
            dt=0.2,
            metadata=metadata,
            vehicle_config=MINING_DRILL_RIG,
        )
        with self.assertRaisesRegex(ValueError, "可行性门禁失败"):
            require_trajectory_feasibility({"trajectory_feasibility": rejected})


if __name__ == "__main__":
    unittest.main()
