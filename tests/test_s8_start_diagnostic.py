import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controller import MPCController
from experiments.s8_start_diagnostic import predict_sequences, summarize_iteration, swept_flags
from interfaces import ControlCmd, VehicleState
from sim import DifferentialDriveModel


class StartDiagnosticTests(unittest.TestCase):
    def test_predictions_match_real_vehicle_every_step_including_limits(self):
        mpc = MPCController(horizon=3, max_v=1., max_omega=.5)
        model = DifferentialDriveModel(max_v=1., max_omega=.5)
        state = VehicleState(.2, -.3, 3.13, .7, -.2)
        commands = np.array([[2., 1., -.4, -.7, .2, .3], [-2., -.1, .1, .8, 0., 0.]])
        predictions = predict_sequences(mpc, state, commands)
        for index, sequence in enumerate(commands):
            current = state
            for step in range(mpc.horizon):
                current = model.step(current, ControlCmd(*sequence[2*step:2*step+2]), mpc.dt)
                np.testing.assert_array_equal(predictions[index, step], current.to_array()[:3])
        for invalid in (np.zeros(6), np.zeros((1, 5)), np.full((1, 6), np.nan)):
            with self.assertRaises(ValueError): predict_sequences(mpc, state, invalid)

    def test_risk_audit_includes_initial_connection_and_does_not_change_ranking(self):
        calls = []
        checker = SimpleNamespace(swept_segment_free=lambda a, b: bool(a[0] <= .5 and b[0] <= .5))
        points = np.array([[[.2, 0., 0.], [.8, 0., 0.]], [[.1, 0., 0.], [.2, 0., 0.]]])
        flags = swept_flags(np.zeros(3), points, checker, lambda: calls.append(1))
        np.testing.assert_array_equal(flags, [[False, True], [False, False]])
        self.assertEqual(len(calls), 4)
        samples = np.zeros((2, 4)); costs = np.array([1., 2.]); before = costs.copy()
        row = summarize_iteration(samples, costs, flags, .5)
        self.assertEqual(row['first_safe_count'], 2); self.assertEqual(row['horizon_safe_count'], 1)
        self.assertFalse(row['minimum_cost_horizon_safe']); self.assertEqual(row['best_horizon_safe_index'], 1)
        np.testing.assert_array_equal(costs, before)
        unsafe = swept_flags(np.array([1., 0., 0.]), points[:1], checker, lambda: None)
        self.assertTrue(unsafe[0, 0])

    def test_precheck_never_constructs_or_executes_controller(self):
        from scripts import diagnose_s8_start_tracking as cli
        with patch.object(cli, 'prepare_config', return_value={}), patch.object(cli, 'MPCController') as controller:
            cli.precheck(); controller.assert_not_called()

    def test_failed_identity_is_preserved_and_output_cannot_be_reused(self):
        from scripts import diagnose_s8_start_tracking as cli
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'review'
            with patch.object(cli, 'prepare_config', side_effect=TimeoutError('identity budget')), patch.object(cli, 'MPCController') as controller:
                with self.assertRaises(TimeoutError): cli.run(output)
                controller.assert_not_called()
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'budget_exhausted')
            self.assertEqual(report['training_updates'], 0); self.assertFalse(report['search_performed'])
            with self.assertRaises(FileExistsError): cli.run(output)


if __name__ == '__main__': unittest.main()
