"""RUN-REFINE: batch safety equivalence and isolated controller changes."""
import unittest
import numpy as np

from controller.constrained_mpc import ConstrainedMPCController
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from sim import DifferentialDriveModel, ParkingEnvironment, RectangleObstacle, CircleObstacle, PolygonObstacle
from scripts.batched_footprint_check import BatchedSafety
from scripts.runtime_refinement_mpc import RefinedCEMController
from scripts.feasible_cem_mpc import FeasibleCEMController
from tests.test_feasible_cem_mpc import FreeSpace, FixedNoise
from interfaces import GoalPose
from runtime.termination import TerminalChecker


class BatchSafetyTests(unittest.TestCase):
    def make_checker(self):
        env = ParkingEnvironment(world_size=30., obstacles=[
            RectangleObstacle(2., 4., -2., 2.), CircleObstacle(-4., 2., 1.),
            PolygonObstacle(((-4., -4.), (-1., -4.), (-1., -3.), (-3., -3.), (-3., -1.), (-4., -1.)))])
        return RectangleFootprintCollisionChecker(env, vehicle_length=2., vehicle_width=1.,
                                                 collision_margin=.2, resolution=.1)

    def test_random_poses_and_concave_polygon_match_scalar(self):
        checker = self.make_checker();batch = BatchedSafety(checker, DifferentialDriveModel(), .1, 2., 1.)
        rng = np.random.default_rng(14)
        poses = rng.uniform([-16., -16., -np.pi], [16., 16., np.pi], (1500, 3))
        expected = [checker.pose_free(*p) for p in poses]
        np.testing.assert_array_equal(batch.poses_free(poses), expected)

    def test_edge_contacts_and_world_boundary_match_scalar(self):
        checker = self.make_checker();batch = BatchedSafety(checker, DifferentialDriveModel(), .1, 2., 1.)
        poses = np.array([[.8+e, 0., 0.] for e in [-1e-8, -1e-10, 0., 1e-10, 1e-8]]
                         + [[13.8+e, 0., 0.] for e in [-1e-10, 0., 1e-10]])
        np.testing.assert_array_equal(batch.poses_free(poses), [checker.pose_free(*p) for p in poses])

    def test_full_sequences_and_wrap_match_scalar(self):
        checker = self.make_checker();model = DifferentialDriveModel()
        scalar = ConstrainedMPCController(footprint_checker=checker, vehicle_model=model, horizon=10)
        batch = BatchedSafety(checker, model, .1, 2., 1.)
        rng = np.random.default_rng(21);sequences = rng.uniform(-1., 1., (80, 20))
        for state in [VehicleState(0., 5., 3.12), VehicleState(0., 0., -.1), VehicleState(2., 0., 0.)]:
            np.testing.assert_array_equal(batch.eligible(sequences, state, scalar.sequence_safe),
                                          [scalar.sequence_safe(s, state) for s in sequences])

    def test_map_changes_refresh_cached_obstacles(self):
        checker = self.make_checker();batch = BatchedSafety(checker, DifferentialDriveModel(), .1, 2., 1.)
        p = np.array([[0., 5., 0.]])
        self.assertTrue(batch.poses_free(p)[0])
        checker.env.obstacles.append(RectangleObstacle(-1., 1., 4., 6.))
        self.assertFalse(batch.poses_free(p)[0])

    def test_nonfinite_out_of_limit_and_custom_model_fail_closed(self):
        checker = self.make_checker();model = DifferentialDriveModel()
        scalar = ConstrainedMPCController(footprint_checker=checker, vehicle_model=model, horizon=2)
        batch = BatchedSafety(checker, model, .1, 2., 1.)
        sequences = np.array([[0., 0., 0., 0.], [0., np.nan, 0., 0.], [3., 0., 0., 0.]])
        np.testing.assert_array_equal(batch.eligible(sequences, VehicleState(0., 5., 0.), scalar.sequence_safe), [True, False, False])
        class Drift(DifferentialDriveModel):
            def step(self, state, cmd, dt):
                return VehicleState(3., 0., 0.)
        custom = Drift();scalar.vehicle_model = custom
        batch = BatchedSafety(checker, custom, .1, 2., 1.)
        self.assertFalse(batch.eligible(sequences[:1], VehicleState(0., 5., 0.), scalar.sequence_safe)[0])


class RefinementTests(unittest.TestCase):
    def controller(self, profile='fast', **kwargs):
        return RefinedCEMController(profile=profile, footprint_checker=kwargs.pop('footprint_checker', FreeSpace()), **kwargs)

    def test_fast_matches_legacy_samples_elites_sequence_and_rng(self):
        checker = BatchSafetyTests().make_checker()
        old = FeasibleCEMController(footprint_checker=checker, seed=860)
        new = self.controller(footprint_checker=checker, seed=860)
        state, warm = VehicleState(0., 5., .2), np.zeros(20)
        ref = np.tile([1., 5., .4], (10, 1))
        np.testing.assert_array_equal(old._solve(warm, ref, state), new._solve(warm, ref, state))
        for a, b in zip(old.safety_audit()['generations'], new.safety_audit()['generations']):
            for key in ('feasible_mask', 'elite_indices'):
                self.assertEqual(a[key], b[key])
        self.assertEqual(old._rng.bit_generator.state, new._rng.bit_generator.state)

    def test_single_elite_keeps_distribution_and_budget(self):
        new = self.controller('variance', horizon=1, population=4, iterations=2)
        new.batch_safety.eligible = lambda samples, state, oracle: np.array([True, False, False, False])
        new._solve(np.zeros(2), np.zeros((1, 3)), VehicleState(0., 0., 0.))
        gs = new.safety_audit()['generations']
        self.assertEqual(len(gs), 2)
        for g in gs:
            self.assertTrue(g['sparse_elite_guard'])
            np.testing.assert_array_equal(g['std_before'], [1., .8])
            np.testing.assert_array_equal(g['std_after'], [1., .8])
            np.testing.assert_array_equal(g['mean_after'], [0., 0.])

    def test_variance_floor_with_identical_elites(self):
        new = self.controller('variance', horizon=1, population=2, iterations=5)
        new._rng = FixedNoise([[0., 0.], [0., 0.]])
        new._solve(np.zeros(2), np.zeros((1, 3)), VehicleState(0., 0., 0.))
        for g in new.safety_audit()['generations']:
            self.assertTrue(np.all(np.array(g['std_after']) >= [.1, .08]))
        self.assertEqual(new._rng.calls, 5)

    def test_selection_evaluates_safe_mean_instead_of_always_executing_it(self):
        new = self.controller('selection', horizon=1, population=2, iterations=1)
        new._rng = FixedNoise([[1., 0.], [-1., 0.]])
        selected = new._solve(np.zeros(2), np.array([[.1, 0., 0.]]), VehicleState(0., 0., 0.))
        np.testing.assert_array_equal(selected, [1., 0.])
        self.assertEqual(new.safety_audit()['selected'], 'safe_sample')
        self.assertTrue(new.safety_audit()['final_mean_safe'])

    def test_progress_holds_when_car_is_behind_and_preserves_rotation_phase(self):
        new = self.controller('progress')
        points = np.array([[0., 0., 0.], [.3, 0., 0.], [.3, 0., .4], [.3, 0., .8], [0., 0., .8]])
        self.assertEqual(new._advance_progress(points, VehicleState(0., 0., 0.)), 0)
        self.assertEqual(new._advance_progress(points, VehicleState(0., 0., 0.)), 1)
        for _ in range(20):
            self.assertEqual(new._advance_progress(points, VehicleState(0., 0., 0.)), 1)
        self.assertEqual(new._advance_progress(points, VehicleState(.3, 0., 0.)), 2)
        self.assertEqual(new._advance_progress(points, VehicleState(.3, 0., 0.)), 2)
        self.assertEqual(new._advance_progress(points, VehicleState(.3, 0., .4)), 3)
        self.assertEqual(new._advance_progress(points, VehicleState(.3, 0., .8)), 4)

    def test_progress_duplicate_poses_bounded_and_reverse_is_not_skipped(self):
        new = self.controller('progress')
        points = np.zeros((12, 3));points[8:, 0] = -.3
        self.assertEqual(new._advance_progress(points, VehicleState(0., 0., 0.)), 0)
        self.assertEqual(new._advance_progress(points, VehicleState(0., 0., 0.)), 4)
        self.assertEqual(new._advance_progress(points, VehicleState(0., 0., 0.)), 8)
        self.assertEqual(new._advance_progress(points, VehicleState(0., 0., 0.)), 8)

    def terminal(self):
        return TerminalChecker(tol_pos=.3, tol_yaw=.15)

    def test_terminal_uses_true_goal_preserves_xy_and_full_turn_certificate(self):
        goal = GoalPose(0., 0., -.8)
        new = self.controller('terminal', goal=goal, terminal=self.terminal())
        state = VehicleState(.1, 0., 0.)
        sequence = new._solve(np.zeros(20), np.tile([4., 4., 2.], (10, 1)), state).reshape(-1, 2)
        self.assertEqual(new.safety_audit()['selected'], 'terminal_rotation')
        np.testing.assert_array_equal(sequence[:, 0], np.zeros(10))
        self.assertTrue(np.all(abs(sequence[:, 1]) <= 1.))
        self.assertTrue(new.safety_audit()['terminal_rotation']['full_rotation_margin_safe'])
        self.assertTrue(new.sequence_safe(sequence, state))
        from interfaces import ControlCmd
        for v, w in sequence:
            state = new.vehicle_model.step(state, ControlCmd(float(v), float(w)), .1)
        self.assertTrue(new.terminal.reached(state, goal))

    def test_terminal_denied_full_turn_or_position_boundary_keeps_base_control(self):
        class NoFullTurn(FreeSpace):
            def swept_segment_free(self, start, end):
                return abs(end[2]-start[2]) < .2
        new = self.controller('terminal', goal=GoalPose(0., 0., -1.5), terminal=self.terminal(), footprint_checker=NoFullTurn())
        new._solve(np.zeros(20), np.zeros((10, 3)), VehicleState(.1, 0., 0.))
        self.assertFalse(new.safety_audit()['terminal_rotation']['full_rotation_margin_safe'])
        self.assertNotEqual(new.safety_audit()['selected'], 'terminal_rotation')
        new._solve(np.zeros(20), np.zeros((10, 3)), VehicleState(.3, 0., 0.))
        self.assertFalse(new.safety_audit()['terminal_rotation']['attempted'])

    def test_unknown_profile_and_missing_goal_rejected(self):
        with self.assertRaises(ValueError): self.controller('typo')
        with self.assertRaises(ValueError): self.controller('terminal')


if __name__ == '__main__':
    unittest.main()
