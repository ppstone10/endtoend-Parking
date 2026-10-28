import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.s8_reference_search import shape_reference,execution_rank,search_reference_parameters,YawLimitError
from experiments.s8_clearance_surrogate import signed_rectangle_gap,swept_penetration_summary
from interfaces import Trajectory
from planner.collision import RectangleFootprintCollisionChecker
from sim import ParkingEnvironment
from sim.obstacles import RectangleObstacle,CircleObstacle


class ReferenceSearchTests(unittest.TestCase):
    def test_shape_zero_exact_and_nonzero_preserves_xy_endpoints_time(self):
        base=Trajectory(np.column_stack((np.linspace(0,1,31),np.zeros(31),np.full(31,3.13))),.2)
        zero=shape_reference(base,base,np.zeros(7))
        changed=shape_reference(base,base,np.full(7,.02))
        self.assertTrue(np.array_equal(zero.points,base.points))
        self.assertTrue(np.array_equal(changed.points[:,:2],base.points[:,:2]))
        self.assertTrue(np.array_equal(changed.points[[0,-1]],base.points[[0,-1]]))
        self.assertEqual(changed.dt,.2);self.assertEqual(changed.horizon,31)
        delta=np.arctan2(np.sin(changed.points[:,2]-base.points[:,2]),np.cos(changed.points[:,2]-base.points[:,2]))
        self.assertLessEqual(abs(delta).max(),.020001)
        with self.assertRaises(ValueError):shape_reference(base,base,np.zeros(6))
        with self.assertRaises(ValueError):shape_reference(base,base,np.full(7,.031))
        old=Trajectory(np.column_stack((base.points[:,:2],np.full(31,3.04))),.2)
        with self.assertRaises(YawLimitError):shape_reference(base,old,np.full(7,.03))

    def test_rank_never_trades_safety_for_terminal_error(self):
        summary=dict(success=True,collision=False,reached_with_margin=False,margin_violation_steps=2,
            final_pos_err=.1,final_yaw_err=.1,truncated_preflight=False)
        proxy=dict(max_penetration=.001,sum_sq_penetration=.000001)
        unsafe=execution_rank(summary,proxy,np.zeros(7))
        strict=execution_rank(dict(summary,reached_with_margin=True,margin_violation_steps=0,final_pos_err=.19),
            dict(max_penetration=0.,sum_sq_penetration=0.),np.full(7,.03))
        collision=execution_rank(dict(summary,collision=True,success=False,final_pos_err=0.),proxy,np.zeros(7))
        self.assertLess(strict,unsafe);self.assertLess(unsafe,collision)
        with self.assertRaises(ValueError):execution_rank(dict(summary,truncated_preflight=True),proxy,np.zeros(7))

    def test_de_budget_seed_bounds_progress_and_first_strict_stop(self):
        budget=SimpleNamespace(check=lambda:None)
        def execute():
            calls=[]
            def objective(x):
                calls.append(x.copy());return (1,0,float(np.square(x-.017).sum()),0.,0.,0.,float(np.linalg.norm(x)))
            result=search_reference_parameters(objective,budget,123,dimension=2,population_size=8,generations=2)
            return result,calls
        first,calls=execute();second,again=execute()
        self.assertEqual(len(calls),24);self.assertTrue(np.array_equal(calls,again))
        self.assertTrue(np.array_equal(calls[0],np.zeros(2)));self.assertLessEqual(np.abs(calls).max(),.03)
        self.assertLessEqual(first['score'],(1,0,2*.017**2,0.,0.,0.,0.))
        self.assertTrue(np.array_equal(first['parameters'],second['parameters']))
        only=[]
        def strict(x):only.append(x);return (0,0,0.,0.,0.,0.,0.)
        found=search_reference_parameters(strict,budget,123,dimension=2,population_size=8,generations=2)
        self.assertEqual(len(only),1);self.assertEqual(found['decision'],'strict_found')
        with self.assertRaises(TimeoutError):
            search_reference_parameters(strict,SimpleNamespace(check=lambda:(_ for _ in ()).throw(TimeoutError('budget'))),123)

    def test_generation_checkpoints_are_copies_and_retain_reproducibility(self):
        budget=SimpleNamespace(check=lambda:None);checkpoints=[]
        objective=lambda x:(1,0,float(np.square(x-.01).sum()),0.,0.,0.,float(np.linalg.norm(x)))
        def checkpoint(generation,population,scores,rng):
            checkpoints.append((generation,population.shape,scores.shape,rng['bit_generator']))
            population[:]=100.;scores[:]=100.;rng['state']['state']=0
        first=search_reference_parameters(objective,budget,17,dimension=2,population_size=8,generations=1,on_generation=checkpoint)
        second=search_reference_parameters(objective,budget,17,dimension=2,population_size=8,generations=1)
        self.assertEqual([c[0] for c in checkpoints],[0,1])
        self.assertEqual(checkpoints[0][1:3],((8,2),(8,7)))
        self.assertTrue(np.array_equal(first['parameters'],second['parameters']));self.assertEqual(first['score'],second['score'])


class ClearanceProxyTests(unittest.TestCase):
    def checker(self):
        return RectangleFootprintCollisionChecker(ParkingEnvironment(20.,[RectangleObstacle(1,3,-1,1)]),
            vehicle_length=2.,vehicle_width=1.,collision_margin=.2,resolution=.1)

    def test_proxy_sign_matches_original_checker_on_rotated_rectangles_and_world(self):
        checker=self.checker();rng=np.random.default_rng(42)
        poses=np.column_stack((rng.uniform(-10,10,100),rng.uniform(-10,10,100),rng.uniform(-np.pi,np.pi,100)))
        gap=signed_rectangle_gap(poses,checker)
        self.assertEqual(gap.shape,(100,))
        self.assertTrue(np.array_equal(gap>0,np.array([checker.pose_free(*p) for p in poses])))
        checker.env.obstacles.append(CircleObstacle(0,0,1))
        with self.assertRaises(ValueError):signed_rectangle_gap(poses,checker)

    def test_swept_proxy_uses_same_sampling_and_retains_hard_flags(self):
        checker=self.checker();start=np.array([-2.,0.,0.]);states=np.array([[4.,0.,0.]])
        flags=np.array([not checker.swept_segment_free(start,states[0])])
        self.assertTrue(flags[0])
        result=swept_penetration_summary(start,states,checker,flags)
        self.assertGreater(result['max_penetration'],0.);self.assertGreater(result['sampled_points'],2)
        self.assertEqual(swept_penetration_summary(start,states,checker,np.array([False]))['sampled_points'],0)
        with self.assertRaises(ValueError):swept_penetration_summary(start,states,checker,np.array([],dtype=bool))


class SearchCLITests(unittest.TestCase):
    def test_first_window_compares_runtime_local_goal_not_dataset_global_goal(self):
        from scripts import search_s8_start_references as cli
        from interfaces import GoalPose,VehicleState
        from runtime.sources import NetworkSource
        case=dict(index=0,start=VehicleState(1,2,.2),goal=GoalPose(3,4,.7))
        source=NetworkSource(None,None);source.begin(case['start'],case['goal'])
        canonical=source._to_local_goal(case['start']).astype(np.float32)
        frame=dict(mask=np.ones(320),pose=case['start'].to_array()[:3],bev=np.zeros((3,4,4)),
            state=np.zeros(2,dtype=np.float32),goal=canonical+np.float32(1e-6))
        data=dict(bevs=np.array([frame['bev']]),goals=np.array([[3,4,.7]]))
        with patch.object(cli,'prepare_supervision_frames',return_value=[{'role':'recovery','goal':frame['goal'].copy()}]), \
                patch.object(cli,'audit_source_windows',return_value=[{'clean':True}]) as audit:
            _,window=cli.check_first_window(case,data,[frame],np.zeros((2,5)),SimpleNamespace(check=lambda:None))
            self.assertTrue(window['clean'])
            self.assertTrue(np.array_equal(np.asarray(audit.call_args.args[0][0]['goal']),canonical))

    def test_precheck_and_budget_failure_never_search_or_rollout(self):
        from scripts import search_s8_start_references as cli
        with patch.object(cli,'checked_config',return_value={}),patch.object(cli,'run_case') as execute, \
                patch.object(cli,'search_reference_parameters') as search:
            cli.precheck(Path('unused'));execute.assert_not_called();search.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'new'
            with patch.object(cli,'checked_config',side_effect=TimeoutError('budget')), \
                    patch.object(cli,'run_case') as execute,patch.object(cli,'search_reference_parameters') as search:
                with self.assertRaises(TimeoutError):cli.run('unused',output,smoke=True)
                execute.assert_not_called();search.assert_not_called()
            report=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted');self.assertFalse(report['search_performed'])
            self.assertFalse(report['teacher_bank_created']);self.assertFalse(report['qualification_passed'])
            with self.assertRaises(FileExistsError):cli.run('unused',output,smoke=True)

    def test_candidate_verification_pairs_fixed_seeds_and_rejects_unclean_window(self):
        from scripts import search_s8_start_references as cli
        from interfaces import VehicleState
        sample={'points':np.zeros((320,3),dtype=np.float32)}
        frames=[dict(length=2)];base=Trajectory(np.zeros((2,3)),.2)
        case=dict(start=VehicleState(0,0,0));engine=SimpleNamespace(margin_violations=[])
        strict=dict(reached_with_margin=True)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(cli,'check_first_window',return_value=(sample,{'clean':False})), \
                    patch.object(cli,'run_case',return_value=(None,engine,[])) as execute, \
                    patch.object(cli,'save_record'),patch.object(cli,'result_row',return_value=strict):
                checked=cli.confirm_candidate(case,base,base,frames,np.zeros((2,5)),{},
                    SimpleNamespace(check=lambda:None),Path(folder))
                self.assertFalse(checked['qualified'])
                self.assertEqual([c.kwargs.get('control_seed',0) for c in execute.call_args_list],[0,1,1,2,2])

    def test_prepare_never_overwrites_frozen_configuration(self):
        from scripts import search_s8_start_references as cli
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            with patch.object(cli,'prepare_config',return_value={'kind':'fixture'}):
                cli.prepare(path)
                with self.assertRaises(FileExistsError):cli.prepare(path)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'kind':'fixture'})

    def test_known_start_witness_is_replayed_once_without_de_search(self):
        from scripts import search_s8_start_references as cli
        calls=[]
        def evaluate(x):
            calls.append(x.copy());return ((1,2,0.,0.,0.,0.,0.) if len(calls)==1 else (0,0,0.,0.,0.,0.,0.))
        with patch.object(cli,'search_reference_parameters') as search:
            result=cli.search_source(evaluate,{'index':2351,'fixed_witness':-.001},SimpleNamespace(check=lambda:None),None)
            search.assert_not_called()
        self.assertEqual(result['evaluations'],2);self.assertEqual(result['decision'],'strict_found')
        self.assertTrue(np.array_equal(calls[0],np.zeros(7)));self.assertTrue(np.array_equal(calls[1],np.full(7,-.001)))


if __name__=='__main__':unittest.main()
