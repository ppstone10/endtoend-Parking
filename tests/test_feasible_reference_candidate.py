import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np

from interfaces import VehicleState, GoalPose
from planner import RectangleFootprintCollisionChecker
from runtime.termination import TerminalChecker
from sim import ParkingEnvironment, RectangleObstacle
from verification.planning_margin import PlanningMarginMPC
from verification.reference_candidate import reference_control_sequence
from verification.feasible_reference_candidate import FeasibleReferenceCandidateMPC, teacher_controller_scope


class FeasibleReferenceTests(unittest.TestCase):
    def controller(self, cls=FeasibleReferenceCandidateMPC, wall=False):
        env = ParkingEnvironment(world_size=100, obstacles=[
            RectangleObstacle(-5., 5., 2.4, 3.)] if wall else [])
        checker = RectangleFootprintCollisionChecker(env, vehicle_length=6., vehicle_width=3.,
                                                     collision_margin=0., resolution=.05)
        controller = cls(contact_checker=checker, dt=.1, horizon=10, seed=296)
        controller.bind_terminal(GoalPose(10, 0, 0), TerminalChecker())
        return controller

    def reference(self):
        return np.column_stack([np.arange(1, 11) * .1, np.zeros(10), np.zeros(10)])

    @staticmethod
    def zero(instance, mean, reference, state):
        instance.last_safety_audit = {'intervened': True, 'selected': 'zero_hold',
                                      'selected_cost': 10., 'selected_horizon_safe': True}
        return np.zeros(20)

    def test_unsafe_window_is_skipped_even_when_projected_controls_are_safe(self):
        controller = self.controller(wall=True)
        state = VehicleState(0, 0, 0)
        reference = self.reference() + [0., 1., 0.]
        candidate = reference_control_sequence(reference, state, controller.vehicle_model, .1, 2., 1.)
        self.assertTrue(controller.sequence_safe(candidate, state))
        rng = deepcopy(controller._rng.bit_generator.state)
        with patch.object(PlanningMarginMPC, '_solve', self.zero), patch(
                'verification.feasible_reference_candidate.reference_control_sequence') as project:
            selected = controller._solve(np.zeros(20), reference, state)
        project.assert_not_called()
        np.testing.assert_array_equal(selected, np.zeros(20))
        self.assertEqual(controller.last_safety_audit['reference_candidate_skipped'], 'unsafe_consumed_window')
        self.assertFalse(controller.last_safety_audit['reference_window_margin_safe'])
        self.assertNotIn('reference_candidate', controller.last_safety_audit)
        self.assertEqual(rng, controller._rng.bit_generator.state)

    def test_legal_window_still_accepts_safe_cheaper_candidate(self):
        controller = self.controller()
        with patch.object(PlanningMarginMPC, '_solve', self.zero):
            result = controller._solve(np.zeros(20), self.reference(), VehicleState(0, 0, 0))
        self.assertTrue(np.any(result != 0))
        self.assertTrue(controller.last_safety_audit['reference_window_margin_safe'])
        self.assertTrue(controller.last_safety_audit['reference_candidate']['accepted'])

    def test_nonzero_parent_preserves_controls_and_rng(self):
        old, new = self.controller(PlanningMarginMPC), self.controller()
        state = VehicleState(0, 0, 0)
        a = old._solve(np.zeros(20), self.reference(), state)
        b = new._solve(np.zeros(20), self.reference(), state)
        self.assertTrue(np.any(a != 0))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(old._rng.bit_generator.state, new._rng.bit_generator.state)
        self.assertNotIn('reference_window_margin_safe', new.last_safety_audit)

    def test_teacher_factory_restored_after_exception(self):
        from verification import reference_teacher
        original = reference_teacher.ReferenceCandidateMPC
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with teacher_controller_scope():
                self.assertIs(reference_teacher.ReferenceCandidateMPC, FeasibleReferenceCandidateMPC)
                raise RuntimeError('fixture')
        self.assertIs(reference_teacher.ReferenceCandidateMPC, original)

    def test_network_factory_restored_after_exception(self):
        from verification import terminal_capture
        from verification.feasible_reference_candidate import feasible_reference_scope
        from experiments.component_capture import VerificationBudget
        from sim import MINING_DRILL_RIG
        original = terminal_capture.TerminalCaptureMPC
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with feasible_reference_scope('unused', MINING_DRILL_RIG, {}, VerificationBudget(1)):
                self.assertIs(terminal_capture.TerminalCaptureMPC, FeasibleReferenceCandidateMPC)
                raise RuntimeError('fixture')
        self.assertIs(terminal_capture.TerminalCaptureMPC, original)


if __name__ == '__main__':
    unittest.main()
