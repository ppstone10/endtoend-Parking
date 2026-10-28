from copy import deepcopy
import unittest

import numpy as np
import torch

from experiments.behavior_preservation import consumed_point_indices,coverage_gate,probe_gate,behavior_loss


class BehaviorPreservationTests(unittest.TestCase):
    def test_actual_window_can_reach_later_than_twenty(self):
        actual=consumed_point_indices(100,199,62,10)
        np.testing.assert_array_equal(actual,np.arange(31,37))
        np.testing.assert_array_equal(consumed_point_indices(1,1,0,10),[0])
        with self.assertRaises(ValueError):consumed_point_indices(4,7,7,10)

    def test_coverage_needs_anchor_and_both_scenes_strict(self):
        rows=[{'dataset_index':i,'scene_name':scene,'supervision':role,'teacher_strict':strict}
              for i,(scene,role,strict) in enumerate([('S3','anchor',False),('S4_dump_area','recovery',True),('S8_weigh_station','recovery',False)])]
        self.assertFalse(coverage_gate(rows)['passed'])
        rows[-1]['teacher_strict']=True
        self.assertTrue(coverage_gate(rows)['passed'])
        with self.assertRaises(ValueError):coverage_gate(rows+[rows[0]])

    @staticmethod
    def row(i,success,collision):
        return {'dataset_index':i,'task_id':f'id-{i}','scene_name':'S4','success':success,'collision':collision}

    def test_gate_requires_success_and_collision_and_no_regression(self):
        a=[self.row(0,True,False),self.row(1,False,True)]
        b=[self.row(0,True,False),self.row(1,True,False)]
        self.assertTrue(probe_gate(a,b)['passed'])
        b[1]['success']=False
        self.assertFalse(probe_gate(a,b)['passed'])
        b[1]['success']=True;b[0]['success']=False
        self.assertFalse(probe_gate(a,b)['passed'])
        with self.assertRaises(ValueError):probe_gate(a,b[:1])

    def test_role_weighting_is_not_sample_count_weighting(self):
        p=torch.zeros(3,2,3);target=p.clone();target[0,:,0]=3
        q=torch.full((3,2),.5);mask=torch.ones(3,2)
        loss=behavior_loss(p,torch.zeros(3,2),target,q,mask,['recovery','anchor','anchor'])
        self.assertAlmostEqual(float(loss),4.5)

    def test_teacher_match_zero_gradient_wrap_and_input_readonly(self):
        p=torch.randn(2,3,3,dtype=torch.double,requires_grad=True);z=torch.randn(2,3,dtype=torch.double,requires_grad=True)
        target=p.detach().clone();q=z.detach().sigmoid();mask=torch.ones(2,3,dtype=torch.double)
        saved=target.clone();loss=behavior_loss(p,z,target,q,mask,['anchor','improve']);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),0.,places=12)
        torch.testing.assert_close(p.grad,torch.zeros_like(p));torch.testing.assert_close(z.grad,torch.zeros_like(z))
        torch.testing.assert_close(target,saved,atol=0,rtol=0)
        target[:,:,2]+=2*torch.pi
        self.assertAlmostEqual(float(behavior_loss(p,z,target,q,mask,['anchor','improve']).detach()),0.,places=12)

    def test_hard_stop_targets_and_empty_window_rejected(self):
        p=torch.zeros(1,2,3);z=torch.zeros(1,2);q=torch.tensor([[0.,1.]])
        loss=behavior_loss(p,z,p,q,torch.ones(1,2),['recovery'])
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):behavior_loss(p,z,p,q,torch.zeros(1,2),['recovery'])


if __name__=='__main__':unittest.main()
