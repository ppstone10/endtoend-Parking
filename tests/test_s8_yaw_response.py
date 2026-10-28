import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controller import MPCController
from experiments.s8_yaw_response import yaw_window_basis,probe_command
from interfaces import Trajectory,VehicleState


class YawResponseTests(unittest.TestCase):
    def test_window_basis_preserves_endpoints_and_outside_support(self):
        basis=yaw_window_basis(30,59,10,10)
        self.assertEqual(basis.shape,(30,));self.assertEqual(basis[0],0);self.assertEqual(basis[-1],0)
        self.assertTrue((basis>=0).all() and (basis<=1).all())
        phase=np.arange(30)*2
        self.assertTrue((basis[np.abs(phase-14.5)>=6.5]==0).all())
        self.assertGreater(basis.max(),.9)
        for args in ((1,1,0,10),(30,59,-1,10),(30,59,59,10),(30,59,0,0)):
            with self.assertRaises(ValueError):yaw_window_basis(*args)

    def test_probe_preserves_live_mpc_rng_progress_reference_and_commands(self):
        trajectory=Trajectory(np.column_stack((np.arange(30)*.05,np.zeros(30),np.zeros(30))),.2)
        mpc=MPCController(seed=2351);state=VehicleState(0,0,0)
        mpc.compute(trajectory,state)
        before_u=mpc._last_u.copy();before_rng=json.dumps(mpc._rng.bit_generator.state,sort_keys=True)
        before_idx=mpc._progress_idx;before_points=trajectory.points.copy()
        zero=probe_command(mpc,trajectory,state,0.)
        plus=probe_command(mpc,trajectory,state,.01)
        minus=probe_command(mpc,trajectory,state,-.01)
        self.assertTrue(np.array_equal(mpc._last_u,before_u));self.assertEqual(mpc._progress_idx,before_idx)
        self.assertEqual(json.dumps(mpc._rng.bit_generator.state,sort_keys=True),before_rng)
        self.assertTrue(np.array_equal(trajectory.points,before_points))
        self.assertEqual(plus['progress_index'],zero['progress_index']);self.assertEqual(minus['progress_index'],zero['progress_index'])
        self.assertTrue(np.array_equal(plus['trajectory'].points[:,:2],before_points[:,:2]))
        self.assertTrue(np.array_equal(plus['trajectory'].points[[0,-1]],before_points[[0,-1]]))
        self.assertTrue(np.array_equal(plus['trajectory'].points[plus['basis']==0.],before_points[plus['basis']==0.]))
        actual=mpc.compute(trajectory,state)
        self.assertEqual((zero['command'].v,zero['command'].omega),(actual.v,actual.omega))
        with self.assertRaises(ValueError):probe_command(mpc,trajectory,state,.02)
        with self.assertRaises(ValueError):probe_command(MPCController(),trajectory,state,0.)

    def test_precheck_budget_and_output_guard_never_rollout(self):
        from scripts import diagnose_s8_yaw_response as cli
        with patch.object(cli,'prepare_config',return_value={}),patch.object(cli,'run_case') as execute:
            cli.precheck();execute.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'new'
            with patch.object(cli,'prepare_config',side_effect=TimeoutError('budget')),patch.object(cli,'run_case') as execute:
                with self.assertRaises(TimeoutError):cli.run(output)
                execute.assert_not_called()
            row=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(row['status'],'budget_exhausted');self.assertEqual(row['training_updates'],0)
            self.assertFalse(row['teacher_labels_created']);self.assertFalse(row['automatic_deployment'])
            with self.assertRaises(FileExistsError):cli.run(output)

    def test_full_branch_requires_the_single_frozen_local_witness(self):
        from scripts import validate_s8_yaw_pulse as cli
        response=dict(amplitude=-.01,reference_internal_safe=True,original_start_connected_safe=True,
            event_window_connected_safe=True,actual_step_safe=True)
        event=dict(seed=20261021,source_index=2351,query_step=40,teacher_update=1,event=197,responses=[response])
        self.assertEqual(cli.select_candidate({'events':[event]}),cli.EXPECTED)
        with self.assertRaises(ValueError):cli.select_candidate({'events':[]})
        with self.assertRaises(ValueError):cli.select_candidate({'events':[event,event]})
        unsafe=dict(event,responses=[dict(response,reference_internal_safe=False)])
        with self.assertRaises(ValueError):cli.select_candidate({'events':[unsafe]})
        with patch.object(cli,'prepare_config',return_value={}),patch.object(cli,'run_case') as execute:
            cli.precheck();execute.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'new'
            with patch.object(cli,'prepare_config',side_effect=TimeoutError('budget')),patch.object(cli,'run_case') as execute:
                with self.assertRaises(TimeoutError):cli.run(output)
                execute.assert_not_called()
            report=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted');self.assertFalse(report['teacher_labels_created'])
            with self.assertRaises(FileExistsError):cli.run(output)


if __name__=='__main__':unittest.main()
