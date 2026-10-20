from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from controller import MPCController
from experiments.footprint_yaw_mpc import FootprintYawMPC
from interfaces import VehicleState,Trajectory
from interfaces import GoalPose
from experiments.recovery_compatibility import run_compatibility,summarize_compatibility
from experiments.recovery_execution import ContextCollector,verify_network_suffix
from runtime import NetworkSource
from tests.test_recovery_execution_probe import FreeChecker,NoisePipeline,TinyModel,TinyPlanner,engine_parts
from scripts import validate_recovery_compatibility as cli
from scripts.probe_reference_events import fixed_cases


class FootprintYawTests(unittest.TestCase):
    def test_orientation_term_equals_four_corner_residual_minus_translation(self):
        mpc=FootprintYawMPC(vehicle_length=6,vehicle_width=3,horizon=2)
        a,b=np.array([1.,2.,.3]),np.array([1.2,1.8,-.4])
        corners=np.array([[3,1.5],[3,-1.5],[-3,-1.5],[-3,1.5]])
        def placed(p):
            rot=np.array([[np.cos(p[2]),-np.sin(p[2])],[np.sin(p[2]),np.cos(p[2])]])
            return corners@rot.T+p[:2]
        expected=np.mean(np.sum((placed(a)-placed(b))**2,axis=1))-np.sum((a[:2]-b[:2])**2)
        self.assertAlmostEqual(float(mpc.orientation_residual(a[2]-b[2])),expected)
        self.assertAlmostEqual(float(mpc.orientation_residual(a[2]-b[2]+2*np.pi)),expected)

    def test_gain_zero_is_original_cost_commands_and_rng(self):
        base=MPCController(horizon=3,population=8,iterations=2,seed=4)
        trial=FootprintYawMPC.from_controller(base,vehicle_length=6,vehicle_width=3,gain=0.)
        points=Trajectory(np.array([[0,0,0],[.1,.1,.2],[.2,.2,.3]]),.1)
        for _ in range(3):
            state=VehicleState(0,0,0)
            self.assertEqual(base.compute(points,state),trial.compute(points,state))
        self.assertEqual(base._rng.bit_generator.state,trial._rng.bit_generator.state)

    def test_copy_preserves_progress_identity_without_mutating_original(self):
        base=MPCController(horizon=3,population=8,iterations=1,seed=4)
        trajectory=Trajectory(np.zeros((4,3)),.1)
        base.compute(trajectory,VehicleState(0,0,0));warm=base._last_u.copy();rng=deepcopy(base._rng.bit_generator.state)
        trial=FootprintYawMPC.from_controller(base,vehicle_length=6,vehicle_width=3)
        self.assertIsNot(trial._progress_traj,base._progress_traj)
        self.assertEqual(trial._progress_idx,base._progress_idx)
        trial.compute(trial._progress_traj,VehicleState(0,0,0))
        np.testing.assert_array_equal(base._last_u,warm);self.assertEqual(base._rng.bit_generator.state,rng)
        self.assertEqual(trial._progress_idx,base._progress_idx+1)

    def test_geometry_validation_and_nonnegative_addition(self):
        for length in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):FootprintYawMPC(vehicle_length=length,vehicle_width=3)
        trial=FootprintYawMPC(vehicle_length=6,vehicle_width=3,horizon=3)
        base=MPCController(horizon=3);samples=np.zeros((2,6));ref=np.array([[0,0,.2]]*3);state=VehicleState(0,0,0)
        np.testing.assert_array_less(base._batch_cost(samples,ref,state),trial._batch_cost(samples,ref,state))

    def test_added_cost_stage_and_terminal_and_positive_gain_rng(self):
        base=MPCController(horizon=3,population=8,iterations=2,seed=7)
        trial=FootprintYawMPC.from_controller(base,vehicle_length=6,vehicle_width=3)
        samples=np.array([[0,.1,0,.2,0,.3],[0,-.1,0,-.2,0,-.3]])
        ref=np.array([[0,0,.2]]*3);state=VehicleState(0,0,0)
        yaw=np.cumsum(samples[:,1::2],axis=1)*.1
        residual=trial.orientation_residual(yaw-ref[:,2])
        expected=base.w_xy*residual.sum(axis=1)+base.w_terminal*residual[:,-1]
        np.testing.assert_allclose(trial._batch_cost(samples,ref,state)-base._batch_cost(samples,ref,state),expected)
        trajectory=Trajectory(ref,.1);base.compute(trajectory,state);trial.compute(trajectory,state)
        self.assertEqual(base._rng.bit_generator.state,trial._rng.bit_generator.state)

    def test_continuous_expert_replans_and_network_matches_full_suffix(self):
        kwargs,mpc=engine_parts()
        engine=ContextCollector(mpc=mpc,source=NetworkSource(NoisePipeline(),TinyModel()),planning_checker=FreeChecker(),max_steps=30,replan_every=10,**kwargs)
        goal=GoalPose(100,0,0);original=engine.run(VehicleState(0,0,0),goal);context=engine.contexts[1]
        args=dict(vehicle_length=6,vehicle_width=3,max_steps=20,replan_every=10)
        normal,_,_,_=run_compatibility(context,'network','original',TinyModel(),TinyPlanner(),goal,kwargs,**args)
        self.assertTrue(verify_network_suffix(original,normal,10)['matched'])
        result,_,audit,details=run_compatibility(context,'expert_continuous','original',TinyModel(),TinyPlanner(),goal,kwargs,**args)
        self.assertEqual(details['expert_applied'],2);self.assertEqual(len(details['teacher_references']),2)
        self.assertEqual(len(audit),result.steps)
        self.assertTrue(all(row['orientation_cost_increment']==0 for row in audit))
        with self.assertRaises(ValueError):run_compatibility(context,'network','footprint_yaw',TinyModel(),TinyPlanner(),goal,kwargs,**args)

    def test_identity_and_output_reuse(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.write_text('frozen')
            config={'kind':'continuous_recovery_and_footprint_yaw','code':{},'runtime':{},'footprint_gain':1.,
                    'progress_floor_ratio':.8,'output':str(root/'results'),'inputs':{'x':{'path':str(source),'sha256':cli.file_digest(source)}}}
            path=root/'config.json';path.write_text(json.dumps(config));sha=cli.file_digest(path)
            with patch.object(cli,'execution_identity',return_value={}),patch.object(cli,'runtime_identity',return_value={}):
                self.assertEqual(cli.load_config(path,sha),config);(root/'results').mkdir()
                with self.assertRaises(FileExistsError):cli.run(path,sha)
                source.write_text('changed')
                with self.assertRaises(ValueError):cli.load_config(path,sha)

    def test_smoke_never_claims_training_or_student_retention(self):
        report=summarize_compatibility([],smoke=True)
        self.assertEqual(report['decision'],'bounded_preflight_only')
        self.assertFalse(report['automatic_training']);self.assertTrue(report['student_success_retention_not_evaluated'])

    def test_stopped_cost_candidate_is_not_in_formal_scope(self):
        cases=fixed_cases()
        for case in cases:case['scene_name']='S4_dump_area' if case['index'] in [1298,1455,1559,1441,1334,1680] else 'S8_weigh_station'
        selected=cli.select_formal_cases(cases)
        self.assertEqual([c['index'] for c in selected],[1298,1455,1559,1441,1334,1680])
        with self.assertRaises(ValueError):cli.select_formal_cases(cases[1:])

    def test_teacher_planning_failure_is_not_dropped_or_fake_prefix_pass(self):
        archive={'expert_once_states':np.ones((10,5)),'expert_once_commands':np.ones((10,2))}
        states,commands=np.empty((0,5)),np.empty((0,2))
        result=cli.verify_teacher_prefix(states,commands,archive,'initial: no feasible trajectory')
        self.assertIsNone(result['matched']);self.assertEqual(result['compared_steps'],0)
        with self.assertRaises(ValueError):cli.verify_teacher_prefix(states,commands,archive,None)


if __name__=='__main__':unittest.main()
