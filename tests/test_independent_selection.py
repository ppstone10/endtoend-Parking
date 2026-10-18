import unittest
from pathlib import Path
import tempfile

import numpy as np

from scripts.build_selection_dataset import collect_metadata_ids, sample_selection_tasks
from scripts.analyze_prefix_kinematics import prefix_metrics, align_outcomes
from scripts.build_dataset import _prepare_checkpoints
from scripts.compare_selection_candidates import assess_candidate_metrics, compare_reports


class FakeTask:
    def __init__(self, identity):
        self.task_id = identity


class FakeSampler:
    def sample(self, scene, kind, sample_index, **kwargs):
        return FakeTask(f"{scene}-{kind}-{sample_index}")


class TestIndependentSelection(unittest.TestCase):
    def test_recovery_source_ids_are_excluded_recursively(self):
        ids = collect_metadata_ids({"task_id": "recovery", "recovery": {"source_task_id": "original"}})
        self.assertEqual(ids, {"recovery", "original"})

    def test_sampling_skips_reserved_identity_and_keeps_unique(self):
        meta = {"scene_name": "S1", "task_type": "T1", "difficulty": {
            "maneuver": "forward", "adjacent_occupancy": 0, "noise_level": "clean"}}
        tasks = sample_selection_tasks([meta, meta], FakeSampler(), {"S1-T1-0"})
        self.assertEqual([task.task_id for task in tasks], ["S1-T1-1", "S1-T1-2"])

    def test_final_test_scene_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "S9"):
            sample_selection_tasks([{"scene_name": "S9_mine_complex"}], FakeSampler(), set())

    def test_changed_selection_identity_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _prepare_checkpoints(output, {"seed": 1, "purpose": "checkpoint_selection_only"})
            with self.assertRaises(RuntimeError):
                _prepare_checkpoints(output, {"seed": 2, "purpose": "checkpoint_selection_only"})


class TestPrefixKinematics(unittest.TestCase):
    def test_straight_motion_is_kinematically_consistent(self):
        points = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
        result = prefix_metrics(points, dt=0.2)
        self.assertAlmostEqual(result["max_lateral_residual_m"], 0)
        self.assertAlmostEqual(result["max_speed_mps"], 0.5)

    def test_sideways_motion_is_detected(self):
        result = prefix_metrics(np.array([[0, 0, 0], [0.1, 0.05, 0]]), dt=0.2)
        self.assertAlmostEqual(result["max_lateral_residual_m"], 0.05)

    def test_pivot_preserves_zero_lateral_motion(self):
        result = prefix_metrics(np.array([[0, 0, 0], [0, 0, 0.1]]), dt=0.2)
        self.assertAlmostEqual(result["max_lateral_residual_m"], 0)
        self.assertAlmostEqual(result["max_yaw_rate_radps"], 0.5)

    def test_missing_and_duplicate_dataset_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            align_outcomes({"episodes": [{"meta": {"dataset_index": 0}}]}, 2)
        with self.assertRaises(ValueError):
            align_outcomes({"episodes": [{"meta": {"dataset_index": 0}}, {"meta": {"dataset_index": 0}}]}, 2)

    def test_changed_task_identity_is_rejected(self):
        report = {"episodes": [{"meta": {"dataset_index": 0, "task_id": "old"}}]}
        with self.assertRaisesRegex(ValueError, "身份"):
            align_outcomes(report, 1, ["new"])

    def test_nonfinite_timestep_is_rejected(self):
        with self.assertRaises(ValueError):
            prefix_metrics(np.zeros((2, 3)), dt=float("nan"))


class TestJointSelection(unittest.TestCase):
    def test_zero_collision_with_no_success_is_not_admitted(self):
        decision = assess_candidate_metrics({"success_rate": 0.66, "collision_rate": 0.28}, {"success_rate": 0, "collision_rate": 0})
        self.assertEqual(decision["status"], "tradeoff_or_regression_not_admitted")

    def test_success_alone_is_insufficient(self):
        decision = assess_candidate_metrics({"success_rate": 0.66, "collision_rate": 0.28}, {"success_rate": 0.8, "collision_rate": 0.2})
        self.assertEqual(decision["status"], "improves_baseline_not_admitted")

    def test_both_thresholds_require_main_protocol_confirmation(self):
        decision = assess_candidate_metrics({"success_rate": 0.66, "collision_rate": 0.28}, {"success_rate": 0.7, "collision_rate": 0.1})
        self.assertEqual(decision["status"], "selection_gate_passed_requires_main_protocol")
        self.assertFalse(decision["deployment_modified"])

    def test_protocol_change_is_rejected(self):
        from copy import deepcopy
        report = {"status": "completed", "experiment": {"name": "E5", "safety_mode": "none"},
                  "overall": {"reconstruct_failures": 0, "evaluated_samples": 1},
                  "protocol": {"data": "a", "evaluated_indices": [0], "max_steps": 600,
                               "replan_every": 10, "control_seed": 0, "mpc_dt": 0.1}}
        candidate = deepcopy(report)
        candidate["protocol"]["replan_every"] = 1
        with self.assertRaisesRegex(ValueError, "协议"):
            compare_reports(report, candidate)


if __name__ == "__main__":
    unittest.main()
