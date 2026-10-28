"""危险尾段不能执行，安全近期参考不应被无条件整段替换。"""
import unittest
import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import FootprintTrajectorySafetyChecker, TerminalChecker
from runtime.safe_prefix import SafePrefixReferenceSource
from sim import ParkingEnvironment, RectangleObstacle


class Source:
    def __init__(self, points, dt=.2):
        self.trajectory = Trajectory(np.asarray(points), dt=dt)
        self.calls = 0
    def begin(self, start, goal):
        self.calls = 0
    def next_trajectory(self, state):
        self.calls += 1
        return self.trajectory, 1.


class PrefixTests(unittest.TestCase):
    def setUp(self):
        env = ParkingEnvironment(40., [RectangleObstacle(4., 5., -1., 1.)])
        self.footprint = RectangleFootprintCollisionChecker(env, vehicle_length=1., vehicle_width=.5,
                                                           collision_margin=.2, resolution=.05)
        self.start = VehicleState(0, 0, 0)
        self.goal = GoalPose(8, -3, 0)
        self.fallback = Source(np.array([[0, -3, 0], [8, -3, 0]], dtype=np.float64))
    def source(self, primary, start=None, goal=None):
        source = SafePrefixReferenceSource(primary, self.fallback, FootprintTrajectorySafetyChecker(self.footprint),
                                            terminal=TerminalChecker(.3, .1), min_duration_s=1.)
        source.begin(start or self.start, goal or self.goal)
        return source

    def test_far_unsafe_tail_returns_exact_safe_prefix_not_expert(self):
        points = np.stack([np.arange(1, 21, dtype=np.float32)*.25, np.zeros(20), np.zeros(20)], axis=1).astype(np.float32)
        primary = Source(points)
        source = self.source(primary)
        result, _ = source.next_trajectory(self.start)
        self.assertLess(len(result.points), len(points))
        self.assertGreaterEqual((len(result.points)-1)*result.dt, 1.)
        np.testing.assert_array_equal(result.points, points[:len(result.points)])
        np.testing.assert_array_equal(primary.trajectory.points, points)
        self.assertEqual(result.points.dtype, points.dtype)
        self.assertEqual(result.dt, primary.trajectory.dt)
        self.assertEqual(self.fallback.calls, 0)
        self.assertTrue(self.footprint.check_trajectory(self.start, result)[0])
        self.assertEqual(source.reference_admission_stats()['safe_prefix'], 1)

    def test_safe_whole_reference_preserves_object(self):
        primary = Source(np.array([[0, -2, 0], [8, -2, 0]], dtype=float))
        result, _ = self.source(primary).next_trajectory(self.start)
        self.assertIs(result, primary.trajectory)

    def test_unsafe_first_bridge_cannot_jump_to_later_safe_points(self):
        primary = Source(np.array([[6, 0, 0], [8, -3, 0]], dtype=float))
        result, _ = self.source(primary).next_trajectory(self.start)
        self.assertIs(result, self.fallback.trajectory)

    def test_short_safe_prefix_falls_back_instead_of_padding_horizon(self):
        primary = Source(np.array([[.25, 0, 0], [4, 0, 0], [8, -3, 0]], dtype=float))
        result, _ = self.source(primary).next_trajectory(self.start)
        self.assertIs(result, self.fallback.trajectory)

    def test_near_safe_reference_without_joint_support_uses_fallback(self):
        state = VehicleState(7.5, -3, 0)
        self.fallback = Source(np.array([[7.8, -3, 0], [8, -3, 0]], dtype=float))
        primary = Source(np.array([[7.8, -3, .5], [8, -3, .5]], dtype=float))
        source = self.source(primary, start=state)
        result, _ = source.next_trajectory(state)
        self.assertIs(result, self.fallback.trajectory)
        self.assertEqual(source.reference_admission_stats()['near_goal_unsupported'], 1)

    def test_joint_support_wraps_yaw_and_uses_strict_original_tolerances(self):
        state = VehicleState(7.5, -3, np.pi)
        goal = GoalPose(8, -3, np.pi-.02)
        primary = Source(np.array([[8, -3, -np.pi+.02]], dtype=float))
        result, _ = self.source(primary, start=state, goal=goal).next_trajectory(state)
        self.assertIs(result, primary.trajectory)
        # 到达状态允许安全保持；不把尚未到达的远端点当作新目标。
        arrived = VehicleState(8, -3, goal.yaw)
        result, _ = self.source(primary, start=arrived, goal=goal).next_trajectory(arrived)
        self.assertIs(result, primary.trajectory)

    def test_mode_cannot_bypass_shared_strict_controller(self):
        from experiments.validation_matrix import ExperimentSpec, run_validation_experiment
        spec = ExperimentSpec('safe_prefix', 'sensor', 'network', 'mpc_vehicle', 'safe_prefix')
        spec.validate()
        with self.assertRaises(ValueError):
            run_validation_experiment(spec, data_path='unused', consistency_hold_limit=1)

    def test_exact_joint_threshold_does_not_count_as_supported(self):
        state = VehicleState(7.5, -3, 0)
        self.fallback = Source(np.array([[8, -3, 0]], dtype=float))
        primary = Source(np.array([[8.25, -3, 0]], dtype=float))
        source = SafePrefixReferenceSource(primary, self.fallback, FootprintTrajectorySafetyChecker(self.footprint),
                                            terminal=TerminalChecker(.25, .125), min_duration_s=1.)
        source.begin(state, self.goal)
        result, _ = source.next_trajectory(state)
        self.assertIs(result, self.fallback.trajectory)

    def test_nonfinite_network_reference_falls_back_and_is_never_emitted(self):
        source = self.source(Source(np.array([[np.nan, 0, 0]], dtype=float)))
        result, _ = source.next_trajectory(self.start)
        self.assertIs(result, self.fallback.trajectory)
        self.assertTrue(np.isfinite(result.points).all())

    def test_all_retention_constraints_are_joint_not_offset_by_total_successes(self):
        from verification.safe_prefix_preflight import summarize_candidate
        def row(i, success):
            return {'dataset_index': i, 'success': success, 'collision': False, 'audit': {
                'actual_contact_steps': 0, 'planning_margin_violations': 0,
                'unsafe_selected_horizons': 0, 'unsafe_emitted_references': 0}}
        baseline = {'rows': {'strict': [row(i, i < 287) for i in range(300)],
                             'native': [row(i, i < 93) for i in range(300)]}}
        historical = {'episodes': [{'meta': {'dataset_index': i}, 'success': i < 205} for i in range(300)]}
        candidate = [row(i, i != 1) for i in range(300)]
        self.assertFalse(summarize_candidate(candidate, baseline, historical, formal=True)['candidate_qualified'])
        candidate[1] = row(1, True)
        accepted = summarize_candidate(candidate, baseline, historical, formal=True)
        self.assertTrue(accepted['candidate_qualified'])
        self.assertFalse(accepted['production_qualified'])
        self.assertFalse(summarize_candidate(candidate[:24], baseline, historical, formal=True)['candidate_qualified'])

    def test_formal_run_requires_frozen_configuration(self):
        from unittest.mock import patch
        from verification.safe_prefix_preflight import run
        with patch('verification.safe_prefix_preflight.validate_baseline', return_value=({}, {})):
            with self.assertRaisesRegex(ValueError, '冻结配置'):
                run('formal', 'unused')


if __name__ == '__main__':
    unittest.main()
