"""隔离求解器候选，避免实验性结果污染默认控制和验收。"""

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from controller import MPCController
from interfaces import VehicleState
from scripts.probe_mpc_solver import IncumbentMPC, CorrelatedMPC, isolated_controller, validate_probe_indices, run_probe
from experiments import validation_matrix


class TestSolverCandidates(unittest.TestCase):
    def test_incumbent_preserves_zero_cost_warm_start(self):
        mpc = IncumbentMPC(seed=8)
        state = VehicleState(0., 0., 0.)
        reference = np.zeros((mpc.horizon, 3))
        warm = np.zeros(2 * mpc.horizon)
        np.testing.assert_array_equal(mpc._solve(warm, reference, state), warm)

    def test_incumbent_no_worse_than_warm_start(self):
        mpc = IncumbentMPC(seed=17)
        state = VehicleState(0., 0., 0.)
        reference = np.column_stack((np.linspace(.1, 1., mpc.horizon), np.zeros((mpc.horizon, 2))))
        warm = np.tile([.4, .1], mpc.horizon)
        result = mpc._solve(warm, reference, state)
        costs = mpc._batch_cost(np.stack([warm, result]), reference, state)
        self.assertLessEqual(costs[1], costs[0] + 1e-12)

    def test_candidates_are_deterministic_and_bounded(self):
        for cls in (IncumbentMPC, CorrelatedMPC):
            first, second = cls(seed=44), cls(seed=44)
            reference = np.column_stack((np.linspace(.1, 1., first.horizon), np.zeros((first.horizon, 2))))
            args = (np.zeros(2 * first.horizon), reference, VehicleState(0., 0., 0.))
            result = first._solve(*args)
            np.testing.assert_array_equal(result, second._solve(*args))
            self.assertLessEqual(np.abs(result[::2]).max(), first.max_v)
            self.assertLessEqual(np.abs(result[1::2]).max(), first.max_omega)

    def test_correlated_noise_has_stationary_scale(self):
        mpc = CorrelatedMPC(seed=42)
        noise = mpc._sample_noise(20000)
        self.assertEqual(noise.shape, (20000, 2 * mpc.horizon))
        sequence = noise.reshape(20000, mpc.horizon, 2)
        np.testing.assert_allclose(sequence.std(axis=0), 1., atol=.025)
        self.assertAlmostEqual(np.corrcoef(sequence[:, 0, 0], sequence[:, 1, 0])[0, 1], .8, delta=.02)

    def test_factory_restored_on_exception(self):
        original = validation_matrix.MPCController
        original_engine = validation_matrix.ClosedLoopEngine
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with isolated_controller('incumbent'):
                self.assertIs(validation_matrix.MPCController, IncumbentMPC)
                raise RuntimeError('probe')
        self.assertIs(validation_matrix.MPCController, original)
        self.assertIs(validation_matrix.ClosedLoopEngine, original_engine)

    def test_baseline_factory_is_unmodified(self):
        with isolated_controller('baseline'):
            self.assertIs(validation_matrix.MPCController, MPCController)

    def test_invalid_indices_rejected(self):
        report = {'protocol': {'evaluated_indices': [1, 2]}}
        for indices in ([], [1, 1], [3]):
            with self.assertRaises(ValueError):
                validate_probe_indices(report, indices)
        self.assertEqual(validate_probe_indices(report, [2, 1]), [2, 1])

    def test_changed_control_period_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / 'baseline.json'
            output = Path(directory) / 'probe'
            baseline.write_text(json.dumps({'status': 'completed', 'experiment': {'name': 'E5', 'safety_mode': 'none'},
                                            'overall': {'reconstruct_failures': 0}, 'protocol': {'mpc_dt': .2}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '控制周期'):
                run_probe(baseline, output, candidate='incumbent')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
