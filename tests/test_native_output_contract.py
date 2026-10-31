"""Contract tests for native hard stopping and complete geometric gradients."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from planner import RectangleFootprintCollisionChecker
from sim import ParkingEnvironment,RectangleObstacle,CircleObstacle,PolygonObstacle
from scripts.native_output_geometry import NativeGeometry
from scripts.native_output_objectives import stop_window_loss,native_pose_loss
from scripts import native_output_objectives as objectives
from scripts import native_output_fit as fitting
from scripts import train_native_output_contract as driver
from model.prediction import TrajectoryPrediction


def geometry(obstacles=()):
    checker=RectangleFootprintCollisionChecker(ParkingEnvironment(world_size=40,obstacles=list(obstacles)),
        vehicle_length=6,vehicle_width=3,collision_margin=.2,resolution=.1)
    return NativeGeometry(checker)


class NativeStopTests(unittest.TestCase):
    def test_subthreshold_049_cannot_escape_hard_stop_training(self):
        logits=torch.full((1,320),torch.logit(torch.tensor(.49)).item(),requires_grad=True)
        loss=stop_window_loss(logits,[1]);loss.backward()
        self.assertGreater(float(loss.detach()),0);self.assertLess(float(logits.grad[0,1]),0)
        self.assertGreater(float(logits.grad[0,0]),0)

    def test_first_step_and_deferred_window(self):
        logits=torch.zeros(2,4,requires_grad=True)
        loss=stop_window_loss(logits,[0,None]);loss.backward()
        self.assertLess(float(logits.grad[0,0]),0);self.assertFalse(logits.grad[1].any())
        zero=stop_window_loss(logits,[None,None]);self.assertEqual(float(zero.detach()),0)

    def test_margin_and_native_endpoint_coordinate_gradient(self):
        logits=torch.tensor([[-1.,1.,-1.]])
        self.assertEqual(float(stop_window_loss(logits,[1])),0)
        points=torch.zeros(1,3,3,requires_grad=True)
        loss=native_pose_loss(points,logits,torch.tensor([[1.,0.,0.]]),torch.tensor([[.3,.2]]))
        loss.backward();self.assertNotEqual(float(points.grad[0,1,0]),0)
        self.assertFalse(points.grad[0,0].any());self.assertFalse(points.grad[0,2].any())

    def test_shape_errors_are_rejected(self):
        with self.assertRaises(ValueError):stop_window_loss(torch.zeros(1,3),[3])
        with self.assertRaises(ValueError):stop_window_loss(torch.zeros(1,3),[])


class NativeGeometryTests(unittest.TestCase):
    def test_safe_path_zero_and_risky_first_bridge_has_gradient(self):
        geo=geometry([RectangleObstacle(5,6,-3,3)])
        safe=torch.tensor([[0.,0.,0.],[1.,0.,0.]],requires_grad=True)
        self.assertEqual(float(geo.loss(safe,np.zeros(3)).detach()),0)
        bad=torch.tensor([[4.,0.,0.]],requires_grad=True)
        loss=geo.loss(bad,np.zeros(3));self.assertGreater(float(loss.detach()),0)
        loss.backward();self.assertTrue(torch.isfinite(bad.grad).all());self.assertGreater(float(bad.grad.abs().sum()),0)
        self.assertEqual(geo.safe_prefix(bad,np.zeros(3)).tolist(),[False])

    def test_full_emitted_tail_not_teacher_length_is_checked(self):
        geo=geometry([RectangleObstacle(5,6,-3,3)])
        points=torch.tensor([[0.,0.,0.],[1.,0.,0.],[4.,0.,0.]])
        self.assertEqual(float(geo.loss(points[:2],np.zeros(3))),0)
        self.assertGreater(float(geo.loss(points,np.zeros(3))),0)
        self.assertEqual(geo.safe_prefix(points,np.zeros(3)).tolist(),[True,True,False])

    def test_circle_polygon_rotation_and_boundary_match_scalar(self):
        obstacles=[CircleObstacle(5,0,1),PolygonObstacle(((-8.,4.),(-4.,4.),(-4.,8.),(-8.,8.)))]
        geo=geometry(obstacles);rng=np.random.default_rng(19)
        poses=np.column_stack((rng.uniform(-19,19,(60,2)),rng.uniform(-np.pi,np.pi,60)))
        risk=geo.pose_penetration(torch.tensor(poses,dtype=torch.float64)).detach().numpy()
        expected=np.array([not geo.checker.pose_free(*p) for p in poses])
        np.testing.assert_array_equal(risk>0,expected)

    def test_ignore_nonforbidden_and_reject_nonconvex(self):
        self.assertEqual(float(geometry([RectangleObstacle(-1,1,-1,1,forbidden=False)]).loss(torch.zeros(1,3),np.zeros(3))),0)
        with self.assertRaises(ValueError):geometry([PolygonObstacle(((0,0),(3,0),(1,1),(3,3),(0,3)))])
        with self.assertRaises(ValueError):geometry([PolygonObstacle(((0,0),(1,0),(2,0)))])

    def test_no_silent_sampling_cap_for_oversized_segment(self):
        with self.assertRaisesRegex(ValueError,'resource limit'):
            geometry().loss(torch.tensor([[1e20,0.,0.]]),np.zeros(3))

    def test_window_requires_safe_complete_prefix(self):
        geo=geometry([RectangleObstacle(5,6,-3,3)])
        points=torch.tensor([[4.,0.,0.],[0.,0.,0.]])
        self.assertIsNone(geo.terminal_index(points,np.zeros(3),np.zeros(3),np.array([.3,.2])))
        self.assertEqual(geometry().terminal_index(points,np.zeros(3),np.zeros(3),np.array([.3,.2])),1)


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__();self.decoder=torch.nn.Module()
        self.decoder.point_head=torch.nn.Linear(1,3,bias=False)
        self.decoder.stop_head=torch.nn.Linear(1,1,bias=False)
        torch.nn.init.zeros_(self.decoder.point_head.weight)
        torch.nn.init.constant_(self.decoder.stop_head.weight,-.04)

    def forward_with_stop(self,bev,goal,state,**kwargs):
        features=torch.ones(len(bev),4,1)
        return TrajectoryPrediction(self.decoder.point_head(features),self.decoder.stop_head(features).squeeze(-1))


def tiny_sample():
    return dict(source_index=0,frame_index=0,pose=np.zeros(3),bev=torch.zeros(1,4,4),goal=torch.tensor([8.,0.,0.]),
        state=torch.zeros(2),points=torch.tensor([[0.,0.,0.],[3.,0.,0.],[6.,0.,0.],[8.,0.,0.]]),
        mask=torch.ones(4),tolerances=torch.tensor([.3,.2]))


class NativeStudyTests(unittest.TestCase):
    def test_no_window_sample_keeps_coordinate_training(self):
        model=TinyNetwork();sample=tiny_sample();cases={0:dict(planner=type('Planner',(),{'_collision_checker':geometry().checker})())}
        loss,record=objectives.objective(model,[sample],cases,'A',1,1);loss.backward()
        self.assertEqual(record['stop_deferred'],1);self.assertEqual(record['window_indices'],[None])
        self.assertGreater(float(model.decoder.point_head.weight.grad.abs().sum()),0)
        self.assertEqual(float(model.decoder.stop_head.weight.grad.abs().sum()),0)

    def test_geometry_mask_is_native_horizon_and_B_retains_old_objective(self):
        model=TinyNetwork();sample=tiny_sample();sample['mask']=torch.tensor([1.,0.,0.,0.])
        class Geometry:
            def __init__(self,checker):pass
            def loss(inner,points,pose):
                self.assertEqual(len(points),4)
                return points.sum()*0+2
        with patch.object(objectives,'NativeGeometry',Geometry):
            loss,record=objectives.objective(model,[sample],{0:dict(planner=type('Planner',(),{'_collision_checker':None})())},'B',1,1)
        self.assertAlmostEqual(float(loss.detach()),record['legacy_joint_loss']+2)

    def test_dose_and_artifact_denominator(self):
        self.assertEqual(len(fitting.expected_files()),82)
        self.assertEqual(driver.PROTOCOL['new_updates'],1200)
        self.assertEqual(driver.PROTOCOL['seeds'],[20261028,20261029])
        self.assertEqual(driver.PROTOCOL['checkpoints'],[0,50,100,200])
        self.assertEqual(driver.PROTOCOL['candidate'],'AB')

    def test_A_or_B_cannot_rescue_failed_AB_or_missing_seed(self):
        arms=[dict(seed=s,arm=a,final=dict(passed=True)) for s in (20261028,20261029) for a in ('A','B','AB')]
        self.assertTrue(fitting.decision(arms)['passed']);arms[-1]['final']['passed']=False
        self.assertFalse(fitting.decision(arms)['passed'])
        with self.assertRaises(ValueError):fitting.decision(arms[:-1])

    def test_authorization_before_fit_output_or_optimizer(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'fit'
            with patch.object(driver,'precheck',side_effect=ValueError('authorization')),patch.object(fitting,'execute') as execute,patch('torch.optim.Adam') as optimizer:
                with self.assertRaises(ValueError):driver.fit('prepared',output)
                execute.assert_not_called();optimizer.assert_not_called();self.assertFalse(output.exists())

    def test_prepare_overwrite_and_source_drift_rejected(self):
        with TemporaryDirectory() as directory:
            with patch.object(driver,'frozen_config',return_value=dict(source='one')):
                target=Path(directory)/'prepared';driver.prepare(target)
                with self.assertRaises(FileExistsError):driver.prepare(target)
                self.assertEqual(driver.precheck(target),dict(source='one'))
            with patch.object(driver,'frozen_config',return_value=dict(source='two')):
                with self.assertRaises(ValueError):driver.precheck(target)

    def test_incomplete_fit_preserved_and_never_resumed(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'fit';config=dict(config_sha256='hash',source={})
            with patch.object(fitting,'baseline',return_value=[]),patch.object(fitting,'load_model_checkpoint',side_effect=ValueError('broken')):
                config['checkpoint']=dict(path='missing')
                with self.assertRaises(ValueError):fitting.execute(config,[],None,None,output,None,lambda:None)
                import json
                report=json.loads((output/'report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['status'],'incomplete');self.assertFalse(report['training_performed'])
                with self.assertRaises(FileExistsError):fitting.execute(config,[],None,None,output,None,lambda:None)


if __name__=='__main__':unittest.main()
