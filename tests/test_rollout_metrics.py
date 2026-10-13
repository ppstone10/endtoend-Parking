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


class TestForwardFlipRate(unittest.TestCase):
    """行进方向抖动率：2026-10-13 实测确认的失效机制签名。

    成功回合 0–5%、振荡失败回合 30–38%，且在同一 checkpoint 内部即可分开成功与振荡。
    """

    def _run(self, per_cycle_delta: list[float]) -> dict:
        goal = GoalPose(0.0, 0.0, 0.0)
        tracker = CycleTracker()
        tracker.begin(goal, _State(20.0, 0.0))
        trajectory = Trajectory(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), dt=0.2)
        x = 20.0
        path = 0.0
        step = 0
        for delta in per_cycle_delta:
            tracker.start_cycle(_State(x, 0.0), trajectory)
            for _ in range(2):
                x -= delta  # delta<0 即远离目标（倒退）
                path += abs(delta)
                step += 1
                tracker.sample(_State(x, 0.0), path_length=path, time_s=0.1 * step)
        return analyze_cycle_samples(tracker)

    def test_monotone_progress_never_flips(self):
        report = self._run([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(report["forward_flip_rate"], 0.0)
        self.assertEqual(report["backward_cycle_ratio"], 0.0)

    def test_alternating_direction_flips_every_cycle(self):
        report = self._run([1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(report["forward_flip_rate"], 1.0)
        self.assertAlmostEqual(report["backward_cycle_ratio"], 0.5)

    def test_partial_alternative_gives_half(self):
        report = self._run([1.0, 1.0, -1.0, -1.0])
        # 相邻对符号序列 +,+,-,- → 1 次翻转 / 3 对
        self.assertAlmostEqual(report["forward_flip_rate"], 1.0 / 3.0, places=4)

    def test_zero_motion_cycles_are_excluded_from_pairs(self):
        report = self._run([1.0, 0.0, 1.0])
        # 中间周期净前进为 0，两对都不可比 → 无可用对
        self.assertIsNone(report["forward_flip_rate"])


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
