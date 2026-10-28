from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from interfaces import Trajectory, VehicleState
from scripts import validate_s8_start_pulse as cli


class StartPulseTests(unittest.TestCase):
    def test_only_single_frozen_safe_response_is_accepted(self):
        response = dict(amplitude=.01, reference_internal_safe=True,
                        event_window_connected_safe=True, actual_step_safe=True)
        prefix = dict(index=2351, strategy='candidate', control_seed=2, event=11,
                      event_diagnostic=dict(responses=[response]))
        self.assertEqual(cli.select_candidate(dict(prefixes=[prefix])), cli.EXPECTED)
        for prefixes in ([], [prefix, prefix]):
            with self.assertRaises(ValueError): cli.select_candidate(dict(prefixes=prefixes))
        unsafe = deepcopy(prefix); unsafe['event_diagnostic']['responses'][0]['actual_step_safe'] = False
        with self.assertRaises(ValueError): cli.select_candidate(dict(prefixes=[unsafe]))
        other = dict(prefix, control_seed=1)
        with self.assertRaises(ValueError): cli.select_candidate(dict(prefixes=[other]))

    def test_reference_boundaries_and_total_yaw_are_not_relaxed(self):
        points = np.column_stack((np.arange(4), np.zeros((4, 2)))).astype(np.float32)
        base = Trajectory(points, .2)
        case = dict(start=VehicleState(0., 0., 0.), planner=SimpleNamespace(
            _collision_checker=SimpleNamespace(check_trajectory=lambda *_: (True, None))))
        changed = points.copy(); changed[1:3, 2] = .01
        row = cli.validate_reference(case, base, base, Trajectory(changed, .2))
        self.assertTrue(row['original_start_and_reference_safe'])
        for column, value in ((0, .01), (2, .11)):
            bad = points.copy(); bad[1, column] += value
            with self.assertRaises(ValueError): cli.validate_reference(case, base, base, Trajectory(bad, .2))

    def test_failed_control2_does_not_start_other_streams_or_export(self):
        points = np.zeros((2, 3)); states = np.zeros((12, 5)); commands = np.zeros((12, 2)); flags = np.zeros(12, bool)
        stored = dict(reference=points, dt=np.asarray(.2), states=states, commands=commands, margin_violations=flags)
        archive = MagicMock(); archive.__enter__.return_value = stored
        checker = SimpleNamespace(swept_segment_free=lambda *_: True)
        case = dict(planner=SimpleNamespace(_collision_checker=checker))
        engine = SimpleNamespace(margin_violations=flags, mpc=SimpleNamespace(execution_trace=np.zeros((12, 2))))
        summary = dict(steps=12, margin_violation_steps=2, reached_with_margin=False)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'validation'
            with patch.object(cli, 'prepare_config', return_value={}), patch.object(cli, 'verify_config'), \
                 patch.object(cli.search, 'load_sources', return_value=({}, [(dict(index=2351), case, None, None, None)])), \
                 patch.object(cli, 'validate_reference', return_value={}), patch.object(cli.np, 'load', return_value=archive), \
                 patch.object(cli, 'run_case', return_value=(None, engine, [])) as execute, \
                 patch.object(cli, 'record_arrays', return_value=(states, commands)), patch.object(cli, 'save_record'), \
                 patch.object(cli.search, 'result_row', return_value=summary), patch.object(cli.search, 'check_first_window') as window:
                report = cli.run(output)
                self.assertEqual([call.kwargs['control_seed'] for call in execute.call_args_list], [2, 2])
                window.assert_not_called()
            self.assertEqual(report['status'], 'completed'); self.assertEqual(report['executed_episodes'], 2)
            self.assertEqual(report['decision'], 'local_safe_step_failed_full_start_transfer')
            self.assertFalse(report['source_qualified']); self.assertFalse(report['teacher_bank_created'])
            with self.assertRaises(FileExistsError): cli.run(output)

    def test_precheck_executes_no_rollout(self):
        with patch.object(cli, 'prepare_config', return_value={}), patch.object(cli, 'run_case') as execute:
            cli.precheck(); execute.assert_not_called()


if __name__ == '__main__': unittest.main()
