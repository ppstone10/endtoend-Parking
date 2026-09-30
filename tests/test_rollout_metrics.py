"""滚动闭环一致性指标测试（E2/E3 的过程级度量）。"""

import unittest

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from metrics.rollout import CycleTracker, analyze_cycle_samples, linear_slope


class _State:
    """最小状态替身：只需要 x/y/yaw 三个属性。"""

    def __init__(self, x: float, y: float, yaw: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw

class TestLinearSlope(unittest.TestCase):
    def test_increasing_series_has_positive_slope(self):
        self.assertAlmostEqual(linear_slope([0.0, 1.0, 2.0, 3.0]), 1.0)

    def test_flat_series_has_zero_slope(self):
        self.assertAlmostEqual(linear_slope([2.0, 2.0, 2.0]), 0.0)

    def test_single_sample_returns_none(self):
        self.assertIsNone(linear_slope([1.0]))


class TestCycleTracker(unittest.TestCase):
    """目标固定在原点：状态 x 即距目标距离（x>0 表示在目标前方）。"""

    def setUp(self):
        self.goal = GoalPose(0.0, 0.0, 0.0)
        self.tracker = CycleTracker()
        self.tracker.begin(self.goal, _State(20.0, 0.0))

    def test_converging_trajectory_has_no_divergence(self):
        trajectory = Trajectory(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), dt=0.2)
        x = 20.0
        steps = 0
        for _ in range(5):
            self.tracker.start_cycle(_State(x, 0.0), trajectory)
            for _ in range(2):
                x -= 1.0  # 向目标前进
                steps += 1
                self.tracker.sample(
                    _State(x, 0.0), path_length=steps * 1.0, time_s=0.1 * steps
                )
        report = analyze_cycle_samples(self.tracker)
        self.assertEqual(report["divergence_events"], 0)
        self.assertAlmostEqual(report["goal_approach_monotonic_ratio"], 1.0)
        self.assertLess(report["drift_slope"], 0.0)
        self.assertAlmostEqual(report["progress_eff_median"], 1.0, delta=1e-6)

    def test_backing_away_is_counted_as_divergence(self):
        trajectory = Trajectory(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), dt=0.2)
        x = 5.0
        steps = 0
        for _ in range(3):
            self.tracker.start_cycle(_State(x, 0.0), trajectory)
            for _ in range(2):
                x += 1.0  # 远离目标
                steps += 1
                self.tracker.sample(
                    _State(x, 0.0), path_length=steps * 1.0, time_s=0.1 * steps
                )
        report = analyze_cycle_samples(self.tracker)
        self.assertEqual(report["divergence_events"], 3)
        self.assertEqual(report["goal_approach_monotonic_ratio"], 0.0)
        self.assertGreater(report["drift_slope"], 0.0)

    def test_stalled_vehicle_has_no_progress_efficiency(self):
        trajectory = Trajectory(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), dt=0.2)
        self.tracker.start_cycle(_State(5.0, 0.0), trajectory)
        for _ in range(3):
            self.tracker.sample(_State(5.0, 0.0), path_length=0.0, time_s=0.0)
        report = analyze_cycle_samples(self.tracker)
        self.assertIsNone(report["progress_eff_median"])
        self.assertEqual(report["divergence_events"], 0)

    def test_heading_divergence_detected_even_when_distance_improves(self):
        """位置一路逼近但航向持续变差——冒烟实测的失败形态必须能被度量出来。"""
        trajectory = Trajectory(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), dt=0.2)
        x = 20.0
        yaw = 0.0
        for _ in range(3):
            self.tracker.start_cycle(_State(x, 0.0, yaw), trajectory)
            for _ in range(2):
                x -= 1.0  # 距离一直下降
                yaw += np.deg2rad(10.0)  # 航向误差一直变大
                self.tracker.sample(_State(x, 0.0, yaw), path_length=1.0, time_s=0.1)
        report = analyze_cycle_samples(self.tracker)
        self.assertEqual(report["divergence_events"], 0)
        self.assertEqual(report["goal_approach_monotonic_ratio"], 1.0)
        self.assertEqual(report["heading_divergence_events"], 3)
        self.assertEqual(report["heading_approach_monotonic_ratio"], 0.0)
        self.assertGreater(report["heading_drift_slope_deg"], 0.0)
        self.assertAlmostEqual(report["yaw_err_end_deg"], 60.0, delta=1e-6)

    def test_distance_to_trajectory_measures_prediction_execution_gap(self):
        trajectory = Trajectory(np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), dt=0.2)
        self.tracker.start_cycle(_State(0.0, 0.3), trajectory)
        self.tracker.sample(_State(0.0, 0.3), path_length=0.0, time_s=0.0)
        report = analyze_cycle_samples(self.tracker)
        self.assertAlmostEqual(report["pred_dist_to_traj_m"], 0.3, delta=1e-6)

    def test_empty_tracker_reports_zero_cycles(self):
        report = analyze_cycle_samples(self.tracker)
        self.assertEqual(report["cycles"], 0)
        self.assertIsNone(report["pred_dist_to_traj_m"])


if __name__ == "__main__":
    unittest.main()
