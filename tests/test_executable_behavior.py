from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from controller import MPCController
from experiments.executable_behavior import PreflightBudget,RecordingMPC,RecordedFixedExpert,prepare_supervision_frames
from experiments.behavior_preservation import fit_bounded_copy
from interfaces import Trajectory,VehicleState,GoalPose


class ExecutableBehaviorTests(unittest.TestCase):
    def test_recording_preserves_control_and_rng(self):
        budget=PreflightBudget(180)
        old=MPCController(dt=.1,horizon=10,seed=4)
        actual=RecordingMPC(dt=.1,horizon=10,seed=4,budget=budget)
        trajectory=Trajectory(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),dt=.2)
        frame={'kind':'network','offset':0,'mask':np.zeros(320)};actual.active_frame=frame
        state=VehicleState(0.,0.,0.)
        for _ in range(3):
            expected=old.compute(trajectory,state);cmd=actual.compute(trajectory,state)
            self.assertEqual((expected.v,expected.omega),(cmd.v,cmd.omega))
            np.testing.assert_array_equal(old._last_u,actual._last_u)
        self.assertEqual(old._rng.bit_generator.state,actual._rng.bit_generator.state)
        self.assertTrue(frame['mask'].any())

    def test_expert_labels_align_to_actual_progress_and_do_not_truncate(self):
        actual=RecordingMPC(dt=.1,horizon=10,seed=4,budget=PreflightBudget())
        trajectory=Trajectory(np.array([[i/10,0.,0.] for i in range(50)]),dt=.2)
        frame={'kind':'expert','offset':None,'pose':(2.,0.,0.),'mask':np.zeros(320)};actual.active_frame=frame
        actual.compute(trajectory,VehicleState(2.,0.,0.))
        self.assertEqual(frame['offset'],20)
        self.assertEqual(frame['length'],30)
        np.testing.assert_allclose(frame['points'][0],[0.,0.,0.],atol=1e-6)
        self.assertTrue(frame['mask'][0])
        self.assertTrue(frame['mask'][1])

    def test_deadline_and_invalid_budget(self):
        with patch('experiments.executable_behavior.time.perf_counter',side_effect=[0.,181.]):
            budget=PreflightBudget()
            with self.assertRaises(TimeoutError):budget.check()
        with self.assertRaises(ValueError):PreflightBudget(181)

    def test_prepare_expert_samples_and_empty_consumption(self):
        frame={'kind':'expert','bev':np.zeros((5,2,2),dtype=np.float32),'goal':np.zeros(3,dtype=np.float32),
               'state':np.zeros(2,dtype=np.float32),'points':np.zeros((320,3),dtype=np.float32),
               'length':4,'mask':np.ones(320,dtype=np.float32)}
        samples=prepare_supervision_frames([frame],None,'recovery',1)
        self.assertEqual(samples[0]['stop'][:5].tolist(),[0.,0.,0.,1.,1.])
        self.assertEqual(samples[0]['source_index'],1)
        frame['mask'][:]=0
        with self.assertRaises(ValueError):prepare_supervision_frames([frame],None,'recovery',1)

    def test_existing_output_rejected_before_loading_inputs(self):
        from scripts.preflight_behavior_repair import run
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileExistsError):run(Path(name))

    def test_network_samples_reproduce_native_stop_and_reject_mismatch(self):
        from types import SimpleNamespace
        class Teacher:
            stop_threshold=.26
            def forward_with_stop(self,bev,goal,state):
                z=torch.full((1,320),-10.);z[:,5:]=1.
                return SimpleNamespace(points=torch.zeros(1,320,3),stop_logits=z)
        frame={'kind':'network','bev':np.zeros((5,2,2),dtype=np.float32),'goal':np.zeros(3,dtype=np.float32),
               'state':np.zeros(2,dtype=np.float32),'length':6,'mask':np.ones(320,dtype=np.float32)}
        samples=prepare_supervision_frames([frame],Teacher(),'anchor',5)
        self.assertEqual(samples[0]['points'].shape,(320,3))
        self.assertEqual(samples[0]['stop'].shape,(320,))
        frame['length']=5
        with self.assertRaises(ValueError):prepare_supervision_frames([frame],Teacher(),'anchor',5)

    def test_short_fit_copies_parent_and_uses_exact_fixed_budget(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(2,4)
            def forward_with_stop(self,bev,goal,state):
                from types import SimpleNamespace
                value=self.linear(state)
                return SimpleNamespace(points=value[:,:3,None].transpose(1,2),stop_logits=value[:,3:])
        parent=Tiny();before=deepcopy(parent.state_dict())
        sample={'bev':torch.zeros(1),'goal':torch.zeros(3),'state':torch.ones(2),
                'points':torch.ones(1,3),'stop':torch.ones(1),'mask':torch.ones(1),'role':'recovery'}
        called=[];candidate,history=fit_bounded_copy(parent,[sample],check_budget=lambda:called.append(1))
        self.assertEqual(len(history),12);self.assertEqual(len(called),12)
        for key in before:torch.testing.assert_close(before[key],parent.state_dict()[key],atol=0,rtol=0)
        self.assertTrue(any(not torch.equal(before[k],candidate.state_dict()[k]) for k in before))
        with self.assertRaises(ValueError):fit_bounded_copy(parent,[sample],steps=13)


if __name__=='__main__':unittest.main()
