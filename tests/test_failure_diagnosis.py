"""失败诊断只观察实际执行，首步不得丢失真实初态。"""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from interfaces import ControlCmd, Trajectory, VehicleState
from runtime.recorder import EpisodeRecord
from scripts.diagnose_closed_loop_failures import analyze_record, run_diagnosis, select_failure_indices
from experiments import validation_matrix
from training.reporting import atomic_write_json


class WallChecker:
    def pose_free(self, x, y, yaw):
        return x < 1.0

    def swept_segment_free(self, start, end):
        return self.pose_free(*start) and self.pose_free(*end)


class TestFailureDiagnosis(unittest.TestCase):
    def test_first_transition_uses_true_start(self):
        start = VehicleState(0, 0, 0)
        record = EpisodeRecord()
        plan = Trajectory(np.array([[0, 0, 0], [2, 0, 0]]), dt=0.2)
        record.log(VehicleState(1.1, 0, 0), ControlCmd(1, 0), plan, plan, True)
        windows = [np.array([[0.1, 0, 0], [0.2, 0, 0]])]
        report = analyze_record(start, record, windows, WallChecker(), WallChecker())
        self.assertTrue(report["initial_pose_free"])
        self.assertEqual(report["collision"]["before_state"][0], 0)
        self.assertEqual(report["collision"]["after_state"][0], 1.1)
        self.assertEqual(report["candidate_mechanism"], "safe_control_window_execution_collision")

    def test_far_reference_does_not_mean_control_window_unsafe(self):
        record = EpisodeRecord()
        plan = Trajectory(np.array([[0, 0, 0], [2, 0, 0]]), dt=0.2)
        record.log(VehicleState(0.1, 0, 0), ControlCmd(1, 0), plan, plan, False)
        windows = [np.array([[0.2, 0, 0], [0.3, 0, 0]])]
        report = analyze_record(VehicleState(0, 0, 0), record, windows, WallChecker(), WallChecker())
        self.assertIsNone(report["first_unsafe_control_window"])
        self.assertEqual(report["candidate_mechanism"], "no_collision")

    def test_empty_record(self):
        report = analyze_record(VehicleState(0, 0, 0), EpisodeRecord(), [], WallChecker(), WallChecker())
        self.assertIsNone(report["collision"])
        self.assertEqual(report["steps"], 0)

    def test_failure_selection_keeps_all_failure_types_and_scene_filter(self):
        report = {"episodes": [
            {"success": False, "meta": {"dataset_index": 2, "scene_name": "S8"}},
            {"success": False, "meta": {"dataset_index": 4, "scene_name": "S6"}},
            {"success": True, "meta": {"dataset_index": 5, "scene_name": "S8"}},
        ]}
        self.assertEqual(select_failure_indices(report), [2, 4])
        self.assertEqual(select_failure_indices(report, {"S8"}), [2])

    def test_observation_hooks_restore_after_evaluation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.npz"
            model = root / "model.pt"
            data.write_bytes(b"fixture")
            model.write_bytes(b"fixture")
            baseline = root / "baseline.json"
            atomic_write_json(baseline, {
                "status": "completed", "experiment": {"name": "E5", "safety_mode": "none"},
                "protocol": {"data": str(data), "checkpoint": str(model), "control_seed": 0,
                             "replan_every": 10, "max_steps": 600},
                "episodes": [{"success": False, "meta": {"dataset_index": 0, "scene_name": "S8"}}],
            })
            engine = validation_matrix.ClosedLoopEngine
            source_builder = validation_matrix._build_source
            with patch.object(validation_matrix, "run_validation_experiment", side_effect=RuntimeError("fixture")):
                with self.assertRaisesRegex(RuntimeError, "fixture"):
                    run_diagnosis(baseline, root / "diagnosis")
            self.assertIs(validation_matrix.ClosedLoopEngine, engine)
            self.assertIs(validation_matrix._build_source, source_builder)


if __name__ == "__main__":
    unittest.main()
