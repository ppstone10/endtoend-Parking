"""FEAS-CEM-001/002: hard eligibility and nonconvex output validation."""
import unittest

import numpy as np

from controller import MPCController
from controller.constrained_mpc import ControlSafetyError
from interfaces import VehicleState
from scripts.feasible_cem_mpc import FeasibleCEMController


class FreeSpace:
    def pose_free(self, x, y, yaw):
        return True

    def swept_segment_free(self, start, end):
        return True


class LeftHalfSpace(FreeSpace):
    def pose_free(self, x, y, yaw):
        return x <= 1e-12

    def swept_segment_free(self, start, end):
        return self.pose_free(*start) and self.pose_free(*end)


class FixedNoise:
    def __init__(self, controls):
        self.controls = np.asarray(controls, dtype=float)
        self.calls = 0

    def normal(self, *, size):
        self.calls += 1
        assert size == self.controls.shape
        return self.controls / np.tile([1., .8], size[1] // 2)


class FeasibleCEMTests(unittest.TestCase):
    def test_free_space_matches_original_sequence_and_rng_exactly(self):
        state = VehicleState(0., 0., .2)
        ref = np.array([[.1, .01, .2], [.2, .02, .2]])
        base = MPCController(horizon=2, population=8, iterations=3, seed=7)
        new = FeasibleCEMController(footprint_checker=FreeSpace(), horizon=2,
                                    population=8, iterations=3, seed=7)
        np.testing.assert_array_equal(base._solve(np.zeros(4), ref, state),
                                      new._solve(np.zeros(4), ref, state))
        self.assertEqual(base._rng.bit_generator.state, new._rng.bit_generator.state)

    def test_unsafe_low_cost_candidates_do_not_enter_elite_fit(self):
        new = FeasibleCEMController(footprint_checker=LeftHalfSpace(), horizon=1,
                                    population=4, iterations=1, elite_frac=.5)
        new._rng = FixedNoise([[1., 0.], [1.5, 0.], [-.2, 0.], [-.4, 0.]])
        chosen = new._solve(np.zeros(2), np.array([[.1, 0., 0.]]), VehicleState(0., 0., 0.))
        np.testing.assert_allclose(chosen, [-.3, 0.], atol=1e-14)
        audit = new.safety_audit()
        self.assertEqual(audit['generations'][0]['feasible_samples'], 2)
        self.assertEqual(audit['generations'][0]['rejected_samples'], 2)
        self.assertEqual(audit['generations'][0]['elite_indices'], [2, 3])

    def test_safe_elites_can_have_unsafe_mean_and_require_fallback(self):
        class NonconvexSpace(FreeSpace):
            def pose_free(self, x, y, yaw):
                return x <= .04 or abs(yaw) >= .05

            def swept_segment_free(self, start, end):
                # Test double isolates nonconvex averaging; production uses real sweeps.
                return self.pose_free(*end)

        new = FeasibleCEMController(footprint_checker=NonconvexSpace(), horizon=1,
                                    population=2, iterations=1)
        new._rng = FixedNoise([[1., 1.], [1., -1.]])
        chosen = new._solve(np.zeros(2), np.array([[.1, 0., 0.]]), VehicleState(0., 0., 0.))
        self.assertFalse(new.safety_audit()['final_mean_safe'])
        self.assertEqual(new.safety_audit()['selected'], 'safe_sample')
        self.assertTrue(new.sequence_safe(chosen, VehicleState(0., 0., 0.)))
        np.testing.assert_array_equal(chosen, [1., 1.])

    def test_empty_feasible_generation_never_fits_unsafe_samples(self):
        new = FeasibleCEMController(footprint_checker=LeftHalfSpace(), horizon=1,
                                    population=2, iterations=2)
        new._rng = FixedNoise([[1., 0.], [1., 0.]])
        chosen = new._solve(np.array([1., 0.]), np.array([[.2, 0., 0.]]), VehicleState(0., 0., 0.))
        np.testing.assert_array_equal(chosen, [0., 0.])
        self.assertEqual(new._rng.calls, 2)
        self.assertTrue(all(g['elite_indices'] == [] for g in new.safety_audit()['generations']))
        self.assertEqual(new.safety_audit()['selected'], 'zero_hold')
        self.assertTrue(new.safety_audit()['is_zero_hold'])

    def test_whole_horizon_checked_even_when_first_step_safe(self):
        class Wall(LeftHalfSpace):
            def pose_free(self, x, y, yaw):
                return x < .15

        new = FeasibleCEMController(footprint_checker=Wall(), horizon=2)
        self.assertFalse(new.sequence_safe(np.array([1., 0., 1., 0.]), VehicleState(0., 0., 0.)))

    def test_invalid_initial_margin_cannot_be_safe_hold(self):
        new = FeasibleCEMController(footprint_checker=LeftHalfSpace(), horizon=1)
        with self.assertRaises(ControlSafetyError):
            new._solve(np.zeros(2), np.zeros((1, 3)), VehicleState(.1, 0., 0.))

    def test_actual_model_with_no_feasible_hold_is_explicit_failure(self):
        class Drift:
            def step(self, state, command, dt):
                return VehicleState(state.x + 1., state.y, state.yaw)

        new = FeasibleCEMController(footprint_checker=LeftHalfSpace(), vehicle_model=Drift(),
                                    horizon=1, population=2, iterations=1)
        with self.assertRaises(ControlSafetyError):
            new._solve(np.zeros(2), np.zeros((1, 3)), VehicleState(0., 0., 0.))

    def test_input_error_and_checker_fault_are_not_safety_stop(self):
        new = FeasibleCEMController(footprint_checker=FreeSpace(), horizon=1)
        with self.assertRaises(ValueError):
            new._solve(np.zeros(2), np.array([[np.nan, 0., 0.]]), VehicleState(0., 0., 0.))
        class Broken(FreeSpace):
            def swept_segment_free(self, start, end):
                raise RuntimeError('checker fault')
        new.footprint_checker = Broken()
        with self.assertRaisesRegex(RuntimeError, 'checker fault'):
            new._solve(np.zeros(2), np.zeros((1, 3)), VehicleState(0., 0., 0.))

    def test_reset_clears_experimental_audit_and_controls(self):
        new = FeasibleCEMController(footprint_checker=FreeSpace(), horizon=1,
                                    population=2, iterations=1)
        new._solve(np.zeros(2), np.zeros((1, 3)), VehicleState(0., 0., 0.))
        self.assertTrue(new.safety_stats())
        new.reset()
        self.assertEqual(new.safety_stats(), {})
        self.assertEqual(new.safety_audit(), {})


if __name__ == '__main__':
    unittest.main()
