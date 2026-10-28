import unittest
from unittest.mock import patch
import numpy as np

from interfaces import VehicleState, GoalPose
from runtime.termination import TerminalChecker
from verification.safety_candidate import FootprintConstrainedMPC
from verification.terminal_capture import TerminalCaptureMPC


class OpenSpace:
    def pose_free(self, x, y, yaw):
        return True

    def swept_segment_free(self, start, end):
        return True


class TerminalCaptureTests(unittest.TestCase):
    def make(self, checker=None):
        controller = TerminalCaptureMPC(contact_checker=checker or OpenSpace(), seed=37)
        controller.bind_terminal(GoalPose(0, 0, 0), TerminalChecker())
        return controller

    def test_outside_position_keeps_old_output_and_rng(self):
        old = FootprintConstrainedMPC(contact_checker=OpenSpace(), seed=37)
        new = self.make()
        state = VehicleState(1, 0, .5)
        ref = np.zeros((10, 3)); mean = np.zeros(20)
        np.testing.assert_array_equal(old._solve(mean, ref, state), new._solve(mean, ref, state))
        self.assertEqual(old._rng.random(), new._rng.random())

    def test_position_boundary_does_not_relax_tolerance(self):
        new = self.make()
        with patch.object(FootprintConstrainedMPC, '_solve', return_value=np.zeros(20)):
            new.last_safety_audit = {'selected': 'legacy_mean'}
            new._solve(np.zeros(20), np.zeros((10, 3)), VehicleState(.3, 0, .5))
        self.assertEqual(new.last_safety_audit['selected'], 'legacy_mean')

    def test_rotation_clips_and_holds_without_translation(self):
        new = self.make(); state = VehicleState(.1, 0, .35)
        selected = new._solve(np.zeros(20), np.zeros((10, 3)), state).reshape(-1, 2)
        np.testing.assert_array_equal(selected[:, 0], np.zeros(10))
        self.assertLessEqual(np.abs(selected[:, 1]).max(), new.max_omega)
        self.assertAlmostEqual(.35 + selected[:, 1].sum() * new.dt, 0.)
        self.assertTrue(np.all(selected[4:] == 0))
        self.assertEqual(new.last_safety_audit['selected'], 'terminal_rotation')

    def test_full_rotation_must_be_safe_beyond_first_horizon(self):
        class BlockedLater(OpenSpace):
            def swept_segment_free(self, start, end):
                return abs(end[2] - start[2]) < 2.
        new = self.make(BlockedLater())
        with patch.object(FootprintConstrainedMPC, '_solve', return_value=np.zeros(20)):
            new.last_safety_audit = {'selected': 'legacy_mean'}
            selected = new._solve(np.zeros(20), np.zeros((10, 3)), VehicleState(0, 0, 2.5))
        np.testing.assert_array_equal(selected, np.zeros(20))
        self.assertEqual(new.last_safety_audit['selected'], 'legacy_mean')

    def test_wrap_uses_shortest_rotation(self):
        new = self.make(); new.bind_terminal(GoalPose(0, 0, -3.0), TerminalChecker())
        selected = new._solve(np.zeros(20), np.zeros((10, 3)), VehicleState(0, 0, 3.0))
        self.assertGreater(selected[1], 0)
        self.assertAlmostEqual(selected[1::2].sum() * .1, 2 * np.pi - 6.)

    def test_already_jointly_reached_keeps_original_output(self):
        new = self.make()
        with patch.object(FootprintConstrainedMPC, '_solve', return_value=np.zeros(20)):
            new.last_safety_audit = {'selected': 'legacy_mean'}
            new._solve(np.zeros(20), np.zeros((10, 3)), VehicleState(.1, 0, .1))
        self.assertEqual(new.last_safety_audit['selected'], 'legacy_mean')

    def test_missing_or_nonfinite_goal_rejected(self):
        new = TerminalCaptureMPC(contact_checker=OpenSpace())
        with self.assertRaises(ValueError):
            new._solve(np.zeros(20), np.zeros((10, 3)), VehicleState(0, 0, .5))
        with self.assertRaises(ValueError):
            new.bind_terminal(GoalPose(float('nan'), 0, 0), TerminalChecker())

    def test_terminal_factory_is_restored_after_exception(self):
        from verification.terminal_capture import terminal_scope
        from verification import candidate_execution
        from experiments import validation_matrix as matrix
        from experiments.component_capture import VerificationBudget
        from sim import MINING_DRILL_RIG
        engine, controller = matrix.ClosedLoopEngine, candidate_execution.FootprintConstrainedMPC
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with terminal_scope('unused', MINING_DRILL_RIG, {}, VerificationBudget(1)):
                self.assertIs(candidate_execution.FootprintConstrainedMPC, TerminalCaptureMPC)
                raise RuntimeError('fixture')
        self.assertIs(matrix.ClosedLoopEngine, engine)
        self.assertIs(candidate_execution.FootprintConstrainedMPC, controller)

    def test_safe_stopping_without_arrival_cannot_open_formal_gate(self):
        from verification.terminal_study import comparisons
        baseline = [{'dataset_index':168, 'success':False, 'collision':False, 'steps':600,
                     'zero_hold_steps':0, 'planning_margin_violations':419, 'intervened_steps':422}]
        self.assertFalse(comparisons(baseline, baseline, baseline)['pilot_eligible'])


if __name__ == '__main__':
    unittest.main()
