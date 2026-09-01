"""L3 单周期 MPC 跟踪复核脚本 smoke 测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.evaluate_single_cycle_tracking import (
    _aggregate,
    _dataset_expert_trajectory,
)


class TestSingleCycleTracking(unittest.TestCase):
    def test_dataset_expert_trajectory_trims_to_mask(self):
        data = {
            "trajs": np.zeros((1, 20, 3), dtype=np.float32),
            "masks": np.ones((1, 20), dtype=np.float32),
            "dt": np.asarray([0.2], dtype=np.float32),
        }
        data["masks"][0, 5:] = 0.0
        trajectory = _dataset_expert_trajectory(data, 0)
        self.assertEqual(trajectory.horizon, 5)
        self.assertAlmostEqual(trajectory.dt, 0.2)

    def test_aggregate_empty(self):
        self.assertEqual(_aggregate([]), {})

    def test_aggregate_computes_metrics(self):
        rows = [
            {"tracking_rms_m": 0.01, "reach_ratio": 1.0, "end_offset_ref_m": 0.1, "collision": False},
            {"tracking_rms_m": 0.03, "reach_ratio": 0.9, "end_offset_ref_m": 0.2, "collision": True},
        ]
        agg = _aggregate(rows)
        self.assertEqual(agg["samples"], 2)
        self.assertAlmostEqual(agg["tracking_rms_mean"], 0.02)
        self.assertAlmostEqual(agg["collision_rate"], 0.5)
        self.assertAlmostEqual(agg["reach_ratio_mean"], 0.95)


class TestScriptImport(unittest.TestCase):
    def test_script_imports(self):
        from scripts.evaluate_single_cycle_tracking import main
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()