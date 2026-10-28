"""完整参考和实际选中控制必须共享严格余量；停车不是到达。"""
import unittest
from unittest.mock import patch
import numpy as np

from controller import MPCController
from controller.constrained_mpc import ConstrainedMPCController, ControlSafetyError
from interfaces import GoalPose, Trajectory, VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import ClosedLoopEngine, FootprintTrajectorySafetyChecker
from runtime.strict_footprint import StrictFootprintSource
from runtime.sources import SafetyStopError
from sim import DifferentialDriveModel, ParkingEnvironment, RectangleObstacle


class Source:
    def __init__(self, points):
        self.trajectory = Trajectory(np.asarray(points, dtype=float), dt=.1)
    def begin(self, start, goal):
        pass
    def next_trajectory(self, state):
        return self.trajectory, 0.


class NarrowBay:
    def pose_free(self, x, y, yaw):
        return abs(y) + 3 * abs(np.sin(yaw)) + 1.5 * abs(np.cos(yaw)) < 1.6
    def swept_segment_free(self, start, end):
        return all(self.pose_free(*(start + t * (end-start))) for t in np.linspace(0, 1, 30))
    def check_trajectory(self, state, trajectory):
        previous = state.to_array()[:3]
        if not self.pose_free(*previous):
            return False, 'current_pose_collision'
        for point in trajectory.points:
            if not self.swept_segment_free(previous, point):
                return False, 'swept_collision'
            previous = point
        return True, None


class StrictRuntimeTests(unittest.TestCase):
    def test_terminal_capture_uses_original_position_threshold_and_zero_translation(self):
        from controller.terminal_capture_mpc import TerminalCaptureMPCController
        from runtime import TerminalChecker
        checker = RectangleFootprintCollisionChecker(ParkingEnvironment(), vehicle_length=1., vehicle_width=.5,
                                                     collision_margin=.2, resolution=.05)
        mpc = TerminalCaptureMPCController(goal=GoalPose(0, 0, .3), terminal=TerminalChecker(.2, .1),
                                          footprint_checker=checker, horizon=2)
        with patch.object(MPCController, '_solve', return_value=np.zeros(4)):
            selected = mpc._solve(np.zeros(4), np.zeros((2, 3)), VehicleState(.1, 0, 0))
        self.assertTrue(np.all(selected[0::2] == 0))
        self.assertEqual(mpc.safety_audit()['selected'], 'terminal_rotation')
        self.assertEqual(mpc.safety_stats()['terminal_rotation'], 1)
        self.assertEqual(mpc.safety_stats()['checks'], 1)
        self.assertTrue(mpc.safety_audit()['terminal_certificate']['full_rotation_margin_safe'])
        with patch.object(MPCController, '_solve', return_value=np.zeros(4)):
            selected = mpc._solve(np.zeros(4), np.zeros((2, 3)), VehicleState(.2, 0, 0))
        self.assertEqual(mpc.safety_audit()['selected'], 'legacy_mean')
        mpc.reset()
        self.assertEqual(mpc.safety_stats(), {})

    def test_full_turn_must_be_safe_even_when_first_rotation_steps_are_safe(self):
        from controller.terminal_capture_mpc import TerminalCaptureMPCController
        from runtime import TerminalChecker
        mpc = TerminalCaptureMPCController(goal=GoalPose(0, 0, 1.5), terminal=TerminalChecker(.2, .1),
                                          footprint_checker=NarrowBay(), horizon=2)
        with patch.object(MPCController, '_solve', return_value=np.zeros(4)):
            selected = mpc._solve(np.zeros(4), np.zeros((2, 3)), VehicleState(0, 0, 0))
        np.testing.assert_array_equal(selected, np.zeros(4))
        self.assertNotIn('terminal_certificate', mpc.safety_audit())

    def test_low_cost_unsafe_mean_cannot_be_executed(self):
        mpc = ConstrainedMPCController(footprint_checker=NarrowBay(), horizon=2, seed=3)
        with patch.object(MPCController, '_solve', return_value=np.array([0., 1., 0., 1.])):
            chosen = mpc._solve(np.zeros(4), np.zeros((2, 3)), VehicleState(0, 0, 0))
        self.assertTrue(mpc.sequence_safe(chosen, VehicleState(0, 0, 0)))
        self.assertTrue(mpc.safety_audit()['intervened'])

    def test_safe_solution_and_rng_preserved(self):
        native = MPCController(horizon=2, seed=19)
        guarded = ConstrainedMPCController(footprint_checker=NarrowBay(), horizon=2, seed=19)
        ref, state = np.zeros((2, 3)), VehicleState(0, 0, 0)
        np.testing.assert_array_equal(native._solve(np.zeros(4), ref, state),
                                      guarded._solve(np.zeros(4), ref, state))
        self.assertEqual(native._rng.bit_generator.state, guarded._rng.bit_generator.state)

    def test_nonfinite_or_unbounded_sequence_is_never_safe(self):
        mpc = ConstrainedMPCController(footprint_checker=NarrowBay(), horizon=2)
        for values in ([0, np.nan, 0, 0], [3, 0, 0, 0], [0, 2, 0, 0]):
            self.assertFalse(mpc.sequence_safe(np.asarray(values), VehicleState(0, 0, 0)))

    def test_full_horizon_not_just_first_command(self):
        mpc = ConstrainedMPCController(footprint_checker=NarrowBay(), horizon=2)
        self.assertFalse(mpc.sequence_safe(np.array([0., 0., 0., 1.]), VehicleState(0, 0, 0)))

    def test_actual_vehicle_model_is_used(self):
        class DriftModel:
            def step(self, state, command, dt):
                return VehicleState(state.x, state.y + 1., state.yaw)
        mpc = ConstrainedMPCController(footprint_checker=NarrowBay(), vehicle_model=DriftModel(), horizon=2)
        with self.assertRaises(ControlSafetyError):
            mpc._solve(np.zeros(4), np.zeros((2, 3)), VehicleState(0, 0, 0))

    def test_reference_and_fallback_checked_to_the_end(self):
        env = ParkingEnvironment(world_size=20., obstacles=[RectangleObstacle(2.5, 3.5, -.5, .5)])
        checker = RectangleFootprintCollisionChecker(env, vehicle_length=1., vehicle_width=.5,
                                                     collision_margin=.2, resolution=.05)
        unsafe = Source([[.1, 0, 0], [4, 0, 0]])
        safe = Source([[0, -2, 0], [4, -2, 0]])
        source = StrictFootprintSource(unsafe, safe, FootprintTrajectorySafetyChecker(checker))
        state = VehicleState(0, 0, 0)
        source.begin(state, GoalPose(4, -2, 0))
        trajectory, _ = source.next_trajectory(state)
        self.assertIs(trajectory, safe.trajectory)
        self.assertEqual(source.safety_stats()['interventions'], 1)

    def test_current_margin_violation_has_no_legacy_bypass(self):
        checker = FootprintTrajectorySafetyChecker(NarrowBay())
        source = StrictFootprintSource(Source([[0, 0, 0]]), Source([[0, 0, 0]]), checker)
        state = VehicleState(0, 1, 0)
        source.begin(state, GoalPose(5, 0, 0))
        self.assertFalse(source.transition_is_safe(state, state))
        with self.assertRaises(SafetyStopError):
            source.next_trajectory(state)

    def test_control_failure_is_safety_stop_but_other_errors_propagate(self):
        source = Source([[.1, 0, 0], [1, 0, 0]])
        mpc = ConstrainedMPCController(footprint_checker=NarrowBay(), horizon=2)
        engine = ClosedLoopEngine(DifferentialDriveModel(), mpc, source, max_steps=3)
        for error in (ControlSafetyError('no safe command'), RuntimeError('model failure')):
            with patch.object(mpc, 'compute', side_effect=error):
                if type(error) is ControlSafetyError:
                    result = engine.run(VehicleState(0, 0, 0), GoalPose(2, 0, 0))
                    self.assertEqual(result.failure, 'safety_stop')
                    self.assertEqual(result.steps, 0)
                    self.assertFalse(result.success)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'model failure'):
                        engine.run(VehicleState(0, 0, 0), GoalPose(2, 0, 0))

    def test_mode_rejects_gt_and_ideal_diagnostic_combinations(self):
        from experiments.validation_matrix import ExperimentSpec
        for bev, executor in [('gt', 'mpc_vehicle'), ('sensor', 'ideal_path')]:
            with self.assertRaises(ValueError):
                ExperimentSpec('bad', bev, 'network', executor, 'strict_footprint').validate()

    def test_mode_rejects_wrapper_that_can_bypass_full_reference_guard(self):
        from experiments.validation_matrix import ExperimentSpec, run_validation_experiment
        spec = ExperimentSpec('strict', 'sensor', 'network', 'mpc_vehicle', 'strict_footprint')
        with self.assertRaisesRegex(ValueError, '迟滞包装'):
            run_validation_experiment(spec, data_path='unused', consistency_hold_limit=1)

    def test_safety_stop_after_an_executed_prefix_preserves_prefix(self):
        from interfaces import ControlCmd
        source = Source([[.1, 0, 0], [1, 0, 0]])
        mpc = ConstrainedMPCController(footprint_checker=NarrowBay(), horizon=2)
        engine = ClosedLoopEngine(DifferentialDriveModel(), mpc, source, max_steps=3)
        with patch.object(mpc, 'compute', side_effect=[ControlCmd(.1, 0), ControlSafetyError('later')]):
            result = engine.run(VehicleState(0, 0, 0), GoalPose(2, 0, 0))
        self.assertEqual(result.failure, 'safety_stop')
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.record.n_steps, 1)

    def test_replacement_proposal_control_error_is_also_stopped(self):
        from interfaces import ControlCmd
        class ReplacingSource(Source):
            def guard_transition(self, before, after):
                return self.trajectory, 0.
            def transition_is_safe(self, before, after):
                return True
        source = ReplacingSource([[.1, 0, 0], [1, 0, 0]])
        mpc = ConstrainedMPCController(footprint_checker=NarrowBay(), horizon=2)
        engine = ClosedLoopEngine(DifferentialDriveModel(), mpc, source, max_steps=3)
        with patch.object(mpc, 'compute', side_effect=[ControlCmd(.1, 0), ControlSafetyError('replacement')]):
            result = engine.run(VehicleState(0, 0, 0), GoalPose(2, 0, 0))
        self.assertEqual(result.failure, 'safety_stop')
        self.assertEqual(result.steps, 0)

    def test_cli_honors_strict_mode_with_registered_e5(self):
        import tempfile
        from scripts.run_validation_experiment import main
        with tempfile.TemporaryDirectory() as directory, \
                patch('sys.argv', ['entry', '--experiment', 'E5', '--safety-mode', 'strict_footprint', '--output', directory]), \
                patch('scripts.run_validation_experiment.run_validation_experiment', return_value={}) as run, \
                patch('scripts.run_validation_experiment._summary_line', return_value='done'):
            main()
        spec = run.call_args.args[0]
        self.assertEqual((spec.trajectory_source, spec.safety_mode), ('network', 'strict_footprint'))

    def test_formal_qualification_cannot_use_small_set_or_offset_old_loss(self):
        from verification.strict_footprint_study import summarize
        def row(i, success):
            return {'dataset_index': i, 'success': success, 'collision': False, 'audit': {
                'actual_contact_steps': 0, 'planning_margin_violations': 0,
                'unsafe_selected_horizons': 0, 'unsafe_emitted_references': 0}}
        historical = {'episodes': [{'meta': {'dataset_index': i}, 'success': i < 205} for i in range(300)]}
        small = {'native': [row(247, False)], 'strict': [row(247, True)]}
        self.assertFalse(summarize(small, historical, formal=True)['candidate_qualified'])
        rows = {'native': [row(i, i < 205) for i in range(300)],
                'strict': [row(i, i != 0) for i in range(300)]}
        result = summarize(rows, historical, formal=True)
        self.assertEqual(result['strict_successes'], 299)
        self.assertFalse(result['candidate_qualified'])
        self.assertEqual(result['historical_success_lost_in_evaluated_set'], [0])
        rows['strict'][0] = row(0, True)
        self.assertTrue(summarize(rows, historical, formal=True)['candidate_qualified'])
        self.assertFalse(summarize(rows, historical, formal=True)['production_qualified'])


if __name__ == '__main__':
    unittest.main()
