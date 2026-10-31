"""EMIT-DIAG: cache semantics, paired controls and bounded attribution."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from interfaces import BEVTensor, GoalPose, VehicleState
from model.variants import MineParkingNetV1
from scripts.emitted_execution_capture import CapturedNetworkSource, NativePredictionRecorder, paired_prefix
from scripts.diagnose_emitted_execution import classify_triplet, metric_summary


class Pipeline:
    def __init__(self): self.calls=0
    def set_target_goals(self,goals): pass
    def capture_bev(self,*pose):
        self.calls+=1
        return BEVTensor(np.zeros((5,16,16),np.float32),1.,(8.,8.,8.,8.),list('abcde'))


class Budget:
    def check(self): pass


class EmittedExecutionTests(unittest.TestCase):
    def test_fixed_reference_does_not_move_or_predict_again(self):
        torch.manual_seed(3)
        model=MineParkingNetV1(hidden_dim=8,max_horizon=4,dt=.2)
        source=CapturedNetworkSource(Pipeline(),model,Budget(),fixed=True,step=lambda:0)
        start=VehicleState(1.,2.,.1,0.,0.);goal=GoalPose(3.,2.,.2)
        source.begin(start,goal)
        first,_=source.next_trajectory(start)
        moved,_=source.next_trajectory(VehicleState(2.,3.,.4,.5,.1))
        np.testing.assert_array_equal(first.points,moved.points)
        self.assertEqual(source.sensor_pipeline.calls,1)
        self.assertEqual(len(source.plans),1)
        self.assertEqual(len(source.raw),1)
        self.assertEqual(source.raw[0]['points'].shape,(4,3))
        self.assertEqual(source.raw[0]['logits'].shape,(4,))
        self.assertFalse(model.decoder._forward_hooks)

    def test_rolling_first_prediction_matches_fixed_and_recaptures(self):
        torch.manual_seed(2);model=MineParkingNetV1(hidden_dim=8,max_horizon=4,dt=.2)
        state=VehicleState(1.,2.,.1,0.,0.);goal=GoalPose(3.,2.,.2)
        sources=[CapturedNetworkSource(Pipeline(),deepcopy(model),Budget(),fixed=f,step=lambda:0) for f in (True,False)]
        predictions=[]
        for source in sources:
            source.begin(state,goal);predictions.append(source.next_trajectory(state)[0])
        np.testing.assert_array_equal(predictions[0].points,predictions[1].points)
        sources[1].next_trajectory(VehicleState(2.,3.,.4,.5,.1))
        self.assertEqual(len(sources[1].plans),2)
        self.assertEqual(sources[1].sensor_pipeline.calls,2)

    def test_shared_prefix_must_match_and_does_not_include_replanning_step(self):
        states=np.arange(75,dtype=float).reshape(15,5);commands=np.ones((15,2))
        a=dict(states=states.copy(),commands=commands.copy())
        b=deepcopy(a);b['states'][10:]+=1
        self.assertEqual(paired_prefix(a,b)['shared_control_steps'],10)
        b['states'][9,0]+=1
        with self.assertRaises(ValueError):paired_prefix(a,b)

    def test_collision_is_not_margin_and_classification_uses_arrival(self):
        def row(success=True,collision=False,margin=0):
            return dict(success=success,collision=collision,margin_violation_steps=margin)
        teacher=row(margin=6)
        self.assertEqual(classify_triplet(teacher,row(),row()),'all_three_reached')
        self.assertEqual(classify_triplet(teacher,row(),row(False)),'replanning_associated_failure')
        self.assertEqual(classify_triplet(teacher,row(False),row()),'replanning_recovers_fixed_failure')
        self.assertEqual(classify_triplet(teacher,row(False),row(False)),'network_fixed_reference_execution_failure')
        self.assertEqual(classify_triplet(row(False,True),row(),row()),'teacher_execution_task_failure')
        summary=metric_summary([teacher,row()])
        self.assertEqual(summary['successes'],2)
        self.assertEqual(summary['collisions'],0)
        self.assertEqual(summary['margin_affected_tasks'],1)

    def test_prediction_observer_preserves_public_output_and_weights(self):
        torch.manual_seed(8);model=MineParkingNetV1(hidden_dim=8,max_horizon=4,dt=.2)
        before={k:v.clone() for k,v in model.state_dict().items()}
        bev=Pipeline().capture_bev();goal=GoalPose(2.,1.,.2);state=VehicleState(0.,0.,0.,0.,0.)
        baseline=model.predict(bev,goal,state)
        recorder=NativePredictionRecorder(model,Budget())
        observed=recorder.predict(bev,goal,state)
        np.testing.assert_array_equal(baseline.points,observed.points)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        with patch.object(model,'predict',side_effect=ValueError('failed inference')):
            with self.assertRaises(ValueError):recorder.predict(bev,goal,state)
        self.assertFalse(model.decoder._forward_hooks)

    def test_incomplete_prefix_and_empty_denominator_rejected(self):
        with self.assertRaises(ValueError):metric_summary([])
        empty=dict(states=np.empty((0,5)),commands=np.empty((0,2)))
        with self.assertRaises(ValueError):paired_prefix(empty,empty)
        short=dict(states=np.zeros((3,5)),commands=np.zeros((3,2)))
        self.assertEqual(paired_prefix(short,short)['shared_control_steps'],3)


if __name__=='__main__': unittest.main()
