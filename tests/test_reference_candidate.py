import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np

from interfaces import VehicleState, GoalPose
from planner import RectangleFootprintCollisionChecker
from runtime.termination import TerminalChecker
from sim import ParkingEnvironment, RectangleObstacle, DifferentialDriveModel
from verification.planning_margin import PlanningMarginMPC
from verification.reference_candidate import ReferenceCandidateMPC, reference_control_sequence


class ReferenceCandidateTests(unittest.TestCase):
    def controller(self, cls=ReferenceCandidateMPC, wall=False):
        env = ParkingEnvironment(world_size=100,
                                 obstacles=[RectangleObstacle(4.1, 5., -5., 5.)] if wall else [])
        checker = RectangleFootprintCollisionChecker(env, vehicle_length=6., vehicle_width=3.,
                                                     collision_margin=0., resolution=.05)
        controller = cls(contact_checker=checker, dt=.1, horizon=10, seed=296)
        controller.bind_terminal(GoalPose(10, 0, 0), TerminalChecker())
        return controller

    def reference(self):
        return np.column_stack([np.arange(1, 11) * .1, np.zeros(10), np.zeros(10)])

    def zero_solver(self, cost=10.):
        def solve(instance, mean, ref, state):
            instance.last_safety_audit = {'intervened': True, 'selected': 'shifted_warm_start',
                                          'selected_cost': cost, 'selected_horizon_safe': True}
            return np.zeros(20)
        return solve

    def test_nonzero_original_and_rng_unchanged(self):
        old = self.controller(PlanningMarginMPC)
        new = self.controller()
        state = VehicleState(0, 0, 0)
        a = old._solve(np.zeros(20), self.reference(), state)
        b = new._solve(np.zeros(20), self.reference(), state)
        self.assertTrue(np.any(a != 0))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(old._rng.random(), new._rng.random())

    def test_safe_cheaper_reference_candidate_is_used_without_rng(self):
        controller = self.controller()
        rng = deepcopy(controller._rng.bit_generator.state)
        with patch.object(PlanningMarginMPC, '_solve', self.zero_solver()):
            selected = controller._solve(np.zeros(20), self.reference(), VehicleState(0, 0, 0))
        self.assertEqual(controller.last_safety_audit['selected'], 'reference_inverse_candidate')
        self.assertTrue(controller.last_safety_audit['reference_candidate']['accepted'])
        self.assertTrue(controller.sequence_safe(selected, VehicleState(0, 0, 0)))
        self.assertEqual(rng, controller._rng.bit_generator.state)

    def test_unsafe_reference_candidate_does_not_replace_zero(self):
        controller = self.controller(wall=True)
        reference = self.reference() + [1., 0., 0.]
        with patch.object(PlanningMarginMPC, '_solve', self.zero_solver(100.)):
            result = controller._solve(np.zeros(20), reference, VehicleState(.5, 0, 0))
        np.testing.assert_array_equal(result, np.zeros(20))
        self.assertEqual(controller.last_safety_audit['reference_candidate']['reason'], 'unsafe')

    def test_non_improving_and_nonfinite_cost_keep_zero(self):
        for cost in [0., float('nan')]:
            controller = self.controller()
            with patch.object(PlanningMarginMPC, '_solve', self.zero_solver(0.)), \
                    patch.object(controller, '_batch_cost', return_value=np.array([cost])):
                result = controller._solve(np.zeros(20), self.reference(), VehicleState(0, 0, 0))
            np.testing.assert_array_equal(result, np.zeros(20))
            self.assertFalse(controller.last_safety_audit['reference_candidate']['accepted'])

    def test_projection_reverse_limits_and_angle_wrap(self):
        model = DifferentialDriveModel()
        reverse = reference_control_sequence([[-.1, 0., 0.]], VehicleState(0, 0, 0), model, .1, 2., 1.)
        np.testing.assert_allclose(reverse, [-1., 0.])
        wrapped = reference_control_sequence([[0., 0., -np.pi + .02]],
                                             VehicleState(0, 0, np.pi - .02), model, .1, 2., 1.)
        np.testing.assert_allclose(wrapped, [0., .4], atol=1e-12)
        limited = reference_control_sequence([[100., 0., np.pi]], VehicleState(0, 0, 0), model, .1, 2., 1.)
        np.testing.assert_allclose(limited, [2., 1.])

    def test_invalid_input_rejected(self):
        for window in [[], [[np.nan, 0., 0.]], [[0., 0.]]]:
            with self.assertRaises(ValueError):
                reference_control_sequence(window, VehicleState(0, 0, 0), DifferentialDriveModel(), .1, 2., 1.)

    def test_scope_restores_after_exception(self):
        from verification import terminal_capture
        from verification.reference_candidate import reference_candidate_scope
        from experiments.component_capture import VerificationBudget
        from sim import MINING_DRILL_RIG
        original = terminal_capture.TerminalCaptureMPC
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with reference_candidate_scope('unused', MINING_DRILL_RIG, {}, VerificationBudget(1)):
                raise RuntimeError('fixture')
        self.assertIs(terminal_capture.TerminalCaptureMPC, original)


if __name__ == '__main__':
    unittest.main()
