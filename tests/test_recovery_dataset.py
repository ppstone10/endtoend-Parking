"""闭环偏离状态专家重标注测试。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset.recovery import (
    RecoveryCandidate,
    build_recovery_sample,
    select_recovery_candidates,
    select_recovery_candidates_with_diagnostics,
)
from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState
from scripts.build_recovery_dataset import (
    _assert_unique_recovery_samples,
    _load_priority_indices,
    _validate_recovery_provenance,
)


class _Planner:
    plan_v = 1.0
    max_omega = 1.0

    def plan(self, state, goal):
        return Trajectory(
            np.asarray(
                [
                    [state.x, state.y, state.yaw],
                    [goal.x, goal.y, goal.yaw],
                ]
            ),
            dt=0.2,
        )

    def model_metadata(self):
        return {"name": "test", "model_version": "v1"}

    def _pose_free(self, x, y, yaw):
        return True

    def _swept_segment_free(self, start, end):
        return True


class _Pipeline:
    def set_target_goals(self, goals):
        self.goals = goals

    def capture_bev(self, x, y, yaw):
        return BEVTensor(
            np.zeros((1, 8, 8), dtype=np.float32),
            resolution=0.25,
            extent=(1.0, 1.0, 1.0, 1.0),
            channels=["occupancy"],
        )


class TestRecoveryCandidateSelection(unittest.TestCase):
    def test_selects_only_strided_safe_deviated_states_in_priority_order(self):
        states = [
            VehicleState(float(index), float((index + 1) % 2), 0.0)
            for index in range(1, 7)
        ]
        collisions = [False, False, False, True, False, False]
        expert = np.stack((np.arange(0.0, 8.0), np.zeros(8), np.zeros(8)), axis=1)
        selected = select_recovery_candidates(
            states,
            collisions,
            expert,
            stride=2,
            min_deviation_m=0.5,
        )
        self.assertEqual([item.rollout_step for item in selected], [3, 2, 6])
        self.assertEqual(selected[0].trigger, "collision_backtrack")
        self.assertAlmostEqual(selected[0].deviation_m, 0.0)
        self.assertTrue(all(np.isclose(item.deviation_m, 1.0) for item in selected[1:]))

    def test_selects_yaw_only_distribution_shift(self):
        selected = select_recovery_candidates(
            [VehicleState(0.0, 0.0, np.deg2rad(20.0))],
            [False],
            np.asarray([[0.0, 0.0, 0.0]]),
            stride=1,
            min_deviation_m=0.25,
            min_yaw_deviation_rad=np.deg2rad(5.0),
            yaw_radius_m=2.0,
        )
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0].position_deviation_m, 0.0)
        self.assertAlmostEqual(selected[0].yaw_deviation_rad, np.deg2rad(20.0))

    def test_includes_last_safe_state_before_collision_outside_stride(self):
        selected = select_recovery_candidates(
            [VehicleState(0.0, 1.0, 0.0), VehicleState(1.0, 1.0, 0.0)],
            [False, True],
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            stride=10,
            min_deviation_m=0.25,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].trigger, "collision_backtrack")

    def test_pre_collision_state_is_kept_even_without_pose_deviation(self):
        selected = select_recovery_candidates(
            [VehicleState(0.0, 0.0, 0.0), VehicleState(1.0, 0.0, 0.0)],
            [False, True],
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            stride=10,
            min_deviation_m=0.25,
            min_yaw_deviation_rad=np.deg2rad(5.0),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].trigger, "collision_backtrack")

    def test_backtracks_past_immediate_state_to_nearest_margin_safe_state(self):
        result = select_recovery_candidates_with_diagnostics(
            [
                VehicleState(1.0, 0.0, 0.0),
                VehicleState(2.0, 0.0, 0.0),
                VehicleState(3.0, 0.0, 0.0),
            ],
            [False, False, True],
            np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            stride=10,
            min_deviation_m=0.25,
            pose_free=lambda x, _y, _yaw: x <= 1.5,
            initial_state=VehicleState(0.0, 0.0, 0.0),
        )
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.rollout_step, 1)
        self.assertEqual(candidate.collision_backtrack_steps, 2)
        self.assertEqual(candidate.trigger, "collision_backtrack")
        self.assertEqual(
            result.diagnostics["immediate_pre_collision_margin_safe_events"], 0
        )
        self.assertEqual(
            result.diagnostics["last_margin_safe_backtrack_events"], 1
        )

    def test_uses_initial_state_when_first_executed_state_collides(self):
        result = select_recovery_candidates_with_diagnostics(
            [VehicleState(1.0, 0.0, 0.0)],
            [True],
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            stride=10,
            min_deviation_m=0.25,
            pose_free=lambda x, _y, _yaw: x == 0.0,
            initial_state=VehicleState(0.0, 0.0, 0.0),
        )
        self.assertEqual(result.candidates[0].rollout_step, 0)
        self.assertEqual(result.candidates[0].collision_backtrack_steps, 1)

    def test_filters_margin_unsafe_stride_candidate_with_diagnostic(self):
        result = select_recovery_candidates_with_diagnostics(
            [VehicleState(1.0, 1.0, 0.0)],
            [False],
            np.asarray([[1.0, 0.0, 0.0]]),
            stride=1,
            min_deviation_m=0.25,
            pose_free=lambda _x, _y, _yaw: False,
        )
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.diagnostics["stride_candidates_before_margin"], 1)
        self.assertEqual(result.diagnostics["stride_margin_unsafe_candidates"], 1)

    def test_rejects_invalid_expert_points(self):
        with self.assertRaisesRegex(ValueError, "expert_points"):
            select_recovery_candidates(
                [VehicleState(0.0, 0.0, 0.0)],
                [False],
                np.empty((0, 3)),
                stride=1,
                min_deviation_m=0.0,
            )

    def test_recovery_sample_relabels_current_state_and_audits_new_trajectory(self):
        source = {
            "task_id": "S1_parking_lot-T1-0001-deadbeef",
            "difficulty": {"maneuver": "reverse"},
            "dataset": {"source": "task", "vehicle_model": {"old": True}},
        }
        state = VehicleState(1.0, 2.0, 0.0, 0.2, 0.0)
        sample = build_recovery_sample(
            RecoveryCandidate(20, state, 0.7, 0.5, 0.1, "stride"),
            source_index=3,
            source_metadata=source,
            goal=GoalPose(1.1, 2.0, 0.0),
            planner=_Planner(),
            pipeline=_Pipeline(),
            checkpoint_identity="abc123",
        )
        self.assertIs(sample.state, state)
        self.assertEqual(sample.task_meta["difficulty"]["maneuver"], "forward")
        self.assertEqual(sample.task_meta["dataset"]["source"], "closed_loop_recovery")
        self.assertTrue(sample.task_meta["dataset"]["feasibility_audit"]["feasible"])
        self.assertEqual(sample.task_meta["recovery"]["source_dataset_index"], 3)
        self.assertEqual(sample.task_meta["recovery"]["collision_backtrack_steps"], 0)
        self.assertEqual(source["difficulty"]["maneuver"], "reverse")


class TestRecoverySupplementSelection(unittest.TestCase):
    def test_selects_only_failed_zero_recovery_tasks_from_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / ".checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "identity.json").write_text(
                json.dumps(
                    {
                        "plan_fingerprint": "plan",
                        "checkpoint_sha256": "checkpoint",
                    }
                ),
                encoding="utf-8",
            )
            records = [
                (3, "collision", 0),
                (4, "timeout", 0),
                (5, "collision", 1),
                (6, "oscillation", 0),
            ]
            for index, failure, recoveries in records:
                (checkpoint_dir / f"part-{index:05d}.done.json").write_text(
                    json.dumps(
                        {
                            "source_index": index,
                            "rollout_failure": failure,
                            "recovery_samples": recoveries,
                        }
                    ),
                    encoding="utf-8",
                )
            indices, evidence = _load_priority_indices(Path(temporary))
        self.assertEqual(indices, [3, 4])
        self.assertEqual(evidence["outcome_counts"], {"collision": 1, "timeout": 1})
        self.assertEqual(evidence["source_plan_fingerprint"], "plan")
        self.assertIn("evidence_sha256", evidence)

    def test_rejects_duplicate_recovery_source_states(self):
        metadata = [
            {"recovery": {"source_dataset_index": 3, "rollout_step": 10}},
            {"recovery": {"source_dataset_index": 3, "rollout_step": 10}},
        ]
        with self.assertRaisesRegex(ValueError, "重复来源状态"):
            _assert_unique_recovery_samples(metadata)

    def test_rejects_recovery_from_different_source_task(self):
        recovery = [
            {
                "recovery": {
                    "source_dataset_index": 0,
                    "source_task_id": "different",
                    "rollout_step": 10,
                }
            }
        ]
        with self.assertRaisesRegex(ValueError, "来源任务身份"):
            _validate_recovery_provenance(recovery, [{"task_id": "expected"}])


if __name__ == "__main__":
    unittest.main()
