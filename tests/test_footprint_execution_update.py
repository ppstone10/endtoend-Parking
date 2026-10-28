"""Safety belongs to the selected full footprint, independently of tracking cost."""
import unittest
from unittest.mock import patch
import numpy as np

from interfaces import VehicleState
from controller import MPCController
from verification.safety_candidate import FootprintConstrainedMPC
from verification.typed_replay import reconstruct_capture_types


class NarrowBay:
    def pose_free(self, x, y, yaw):
        return abs(y) + 3 * abs(np.sin(yaw)) + 1.5 * abs(np.cos(yaw)) < 1.6

    def swept_segment_free(self, start, end):
        return all(self.pose_free(*(start[:3] + t * (end[:3] - start[:3]))) for t in np.linspace(0, 1, 20))


class SafeOutputTests(unittest.TestCase):
    def test_safe_legacy_has_same_sampling_stream_and_numeric_output(self):
        native = MPCController(dt=.1, horizon=2, seed=31)
        constrained = FootprintConstrainedMPC(contact_checker=NarrowBay(), dt=.1, horizon=2, seed=31)
        ref = np.zeros((2, 3)); state = VehicleState(0, 0, 0)
        original = native._solve(np.zeros(4), ref, state)
        chosen = constrained._solve(np.zeros(4), ref, state)
        if constrained.last_safety_audit['legacy_safe']:
            np.testing.assert_array_equal(original, chosen)
        self.assertEqual(native._rng.random(), constrained._rng.random())

    def test_low_cost_safe_sample_is_used_instead_of_unsafe_mean(self):
        mpc = FootprintConstrainedMPC(contact_checker=NarrowBay(), dt=.1, horizon=2)
        def native_search(instance, mean, ref, state):
            instance._batch_cost(np.asarray([[1., 1., 1., 1.], [1., 0., 1., 0.]]), ref, state)
            return np.asarray([1., 1., 1., 1.])
        with patch.object(MPCController, '_solve', native_search):
            chosen = mpc._solve(np.zeros(4), np.asarray([[.1, 0, 0], [.2, 0, 0]]), VehicleState(0, 0, 0))
        np.testing.assert_array_equal(chosen, [1, 0, 1, 0])
        self.assertEqual(mpc.last_safety_audit['selected'], 'safe_sample')

    def test_unsafe_cem_mean_cannot_escape_safe_candidate_check(self):
        mpc = FootprintConstrainedMPC(contact_checker=NarrowBay(), dt=.1, horizon=2, seed=3)
        ref = np.zeros((2, 3))
        with patch.object(MPCController, '_solve', return_value=np.asarray([0, 1, 0, 1.])):
            chosen = mpc._solve(np.zeros(4), ref, VehicleState(0, 0, 0))
        self.assertTrue(mpc.sequence_safe(chosen, VehicleState(0, 0, 0)))
        self.assertFalse(mpc.last_safety_audit['legacy_safe'])

    def test_safe_legacy_output_is_exactly_preserved(self):
        mpc = FootprintConstrainedMPC(contact_checker=NarrowBay(), dt=.1, horizon=2, seed=3)
        legacy = np.asarray([.2, 0, .2, 0])
        with patch.object(MPCController, '_solve', return_value=legacy):
            chosen = mpc._solve(np.zeros(4), np.zeros((2, 3)), VehicleState(0, 0, 0))
        self.assertTrue(np.array_equal(chosen, legacy))
        self.assertEqual(mpc.last_safety_audit['selected'], 'legacy_mean')

    def test_invalid_initial_footprint_is_not_reported_as_safe_hold(self):
        mpc = FootprintConstrainedMPC(contact_checker=NarrowBay(), dt=.1, horizon=2)
        with self.assertRaises(ValueError):
            mpc._solve(np.zeros(4), np.zeros((2, 3)), VehicleState(0, 2, 0))

    def test_typed_replay_keeps_native_scalar_evolution_and_rejects_drift(self):
        from sim import DifferentialDriveModel
        from interfaces import ControlCmd
        start = VehicleState(0., 0., 0.)
        after = DifferentialDriveModel().step(start, ControlCmd(.1, .2), .1)
        types = reconstruct_capture_types(start.to_array(), np.asarray([start.to_array()]),
                                          np.asarray([after.to_array()]), [[.1, .2]], [0],
                                          DifferentialDriveModel(), .1)
        self.assertEqual(types[0], ['float', 'float', 'float'])
        wrong = after.to_array(); wrong[0] += .01
        with self.assertRaises(ValueError):
            reconstruct_capture_types(start.to_array(), np.asarray([start.to_array()]),
                                      np.asarray([wrong]), [[.1, .2]], [0], DifferentialDriveModel(), .1)

    def test_safe_stop_after_losing_old_success_cannot_claim_pilot_benefit(self):
        from verification.candidate_execution import compare_rows
        native = [{'dataset_index': 1, 'success': True, 'collision': False}]
        candidate = [{'dataset_index': 1, 'success': False, 'collision': False,
                      'zero_hold_steps': 600, 'planning_margin_violations': 0, 'intervened_steps': 600}]
        result = compare_rows(native, candidate)
        self.assertFalse(result['pilot_benefit'])
        self.assertEqual(result['old_success_lost'], [1])

    def test_candidate_scope_restores_factory_on_exception(self):
        from verification.candidate_execution import candidate_scope
        from experiments import validation_matrix as matrix
        from experiments.component_capture import VerificationBudget
        from sim import MINING_DRILL_RIG
        original = matrix.ClosedLoopEngine
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with candidate_scope('control', 'unused', MINING_DRILL_RIG, {}, VerificationBudget(1)):
                self.assertIsNot(matrix.ClosedLoopEngine, original)
                raise RuntimeError('fixture')
        self.assertIs(matrix.ClosedLoopEngine, original)

    def test_perception_restore_keeps_python_and_numpy_types_distinct(self):
        from verification.typed_replay import restore_pose
        native = restore_pose([1., 2., .1], ['float', 'float64', 'float'])
        self.assertIs(type(native[0]), float)
        self.assertIs(type(native[1]), np.float64)
        with self.assertRaises(ValueError):
            restore_pose([1., 2., .1], ['float', 'unknown', 'float'])

    def test_tampered_pilot_cannot_open_formal_gate(self):
        from pathlib import Path
        import tempfile, json, hashlib
        from verification.candidate_study import validated_pilot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / 'report.json'
            pilot.write_text('{}', encoding='utf-8')
            digest = hashlib.sha256(pilot.read_bytes()).hexdigest()
            (root / 'artifacts.json').write_text(json.dumps({'report.json': digest}), encoding='utf-8')
            pilot.write_text('{"pilot_benefit":true}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '漂移'):
                validated_pilot(pilot, root / 'unused-audit.json')


if __name__ == '__main__':
    unittest.main()
