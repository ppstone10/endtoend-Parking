import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from experiments.s8_reference_timing import double_duration_reference
from interfaces import Trajectory


class TimingTests(unittest.TestCase):
    def test_original_knots_endpoints_dt_and_double_duration(self):
        original=Trajectory(np.array([[0,0,.1],[1,2,.3],[2,3,.5]]),.2)
        result=double_duration_reference(original)
        self.assertTrue(np.array_equal(result.points[::2],original.points))
        self.assertEqual(result.dt,.2)
        self.assertEqual((len(result.points)-1)*result.dt,2*(len(original.points)-1)*original.dt)
        self.assertTrue(np.allclose(result.points[1,:2],[.5,1]))

    def test_wrapped_midpoint_and_no_truncation(self):
        original=Trajectory(np.array([[0,0,np.deg2rad(179)],[1,0,np.deg2rad(-179)]]),.2)
        mid=double_duration_reference(original).points[1,2]
        self.assertAlmostEqual(abs(float(mid)),np.pi,places=5)
        with self.assertRaises(ValueError):double_duration_reference(Trajectory(np.zeros((161,3)),.2))
        with self.assertRaises(ValueError):double_duration_reference(Trajectory(np.array([[0,0,0],[1,0,np.pi]]),.2))
        with self.assertRaises(ValueError):double_duration_reference(Trajectory(np.zeros((4,3)),.4))

    def test_precheck_and_budget_failure_never_execute_reference(self):
        from scripts import preflight_s8_entry_timing as cli
        with patch.object(cli,'prepare_config',return_value=({},{})),patch.object(cli,'run_case') as run:
            cli.precheck();run.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'new'
            with patch.object(cli,'prepare_config',side_effect=TimeoutError('budget')),patch.object(cli,'run_case') as run:
                with self.assertRaises(TimeoutError):cli.run(target)
                run.assert_not_called()
            report=json.loads((target/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertEqual(report['training_updates'],0);self.assertFalse(report['allowed_to_prepare_training_plan'])
            with self.assertRaises(FileExistsError):cli.run(target)


if __name__=='__main__':unittest.main()
