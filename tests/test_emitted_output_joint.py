"""EMIT-JOINT-001: numerical, gradient and native threshold contract."""
import math
import unittest

import numpy as np
import torch

from experiments.emitted_output_joint import (
    first_stop_distribution, native_lengths, emitted_pose_loss, paired_objective,
)
from model.network import variable_loss_fn, endpoint_alignment_loss, terminal_pose_loss
from model.variants import MineParkingNetV1
from interfaces import BEVTensor, GoalPose, VehicleState


class EmittedOutputJointTests(unittest.TestCase):
    def test_mass_includes_no_stop_and_extreme_logits(self):
        logits = torch.tensor([[-1000., -1000., -1000.], [1000., -1000., 1000.],
                               [-1., 0., 1.]], requires_grad=True)
        q = first_stop_distribution(logits)
        torch.testing.assert_close(q.sum(1), torch.ones(3))
        torch.testing.assert_close(q[:2], torch.tensor([[0., 0., 1.], [1., 0., 0.]]))
        (q * torch.arange(3)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(native_lengths(logits).tolist(), [3, 1, 2])

    def test_last_logit_cannot_change_fallback_endpoint(self):
        a = torch.tensor([[-2., -3., -1000.]])
        b = a.clone(); b[:, -1] = 1000
        torch.testing.assert_close(first_stop_distribution(a), first_stop_distribution(b))
        self.assertEqual(first_stop_distribution(torch.zeros(2, 1)).tolist(), [[1.], [1.]])

    def test_both_heads_get_correct_direction(self):
        points = torch.tensor([[[2., 0., .4], [0., 0., 0.]]], requires_grad=True)
        logits = torch.tensor([[0., 0.]], requires_grad=True)
        loss = emitted_pose_loss(points, logits, torch.zeros(1, 3), torch.ones(1, 2))
        loss.backward()
        self.assertGreater(float(points.grad[0, 0, 0]), 0.)
        self.assertGreater(float(points.grad[0, 0, 2]), 0.)
        self.assertGreater(float(logits.grad[0, 0]), 0.)  # gradient descent delays bad early stop
        self.assertEqual(float(logits.grad[0, 1]), 0.)
        self.assertTrue(torch.isfinite(points.grad).all())
        reverse = points.detach().flip(1).requires_grad_()
        other = logits.detach().clone().requires_grad_()
        emitted_pose_loss(reverse, other, torch.zeros(1, 3), torch.ones(1, 2)).backward()
        self.assertLess(float(other.grad[0, 0]), 0.)  # good early point should stop

    def test_wrapped_yaw_and_per_sample_tolerance(self):
        points = torch.tensor([[[0., 0., -math.pi + .01]]], dtype=torch.float64)
        goal = torch.tensor([[0., 0., math.pi - .01]], dtype=torch.float64)
        loss = emitted_pose_loss(points, torch.zeros(1, 1), goal, torch.tensor([[.3, .1]]))
        self.assertAlmostEqual(float(loss), .04, places=8)

    def test_soft_improvement_does_not_imply_native_pass(self):
        # At logit 0 the hard rule stops at bad point 0; the proxy still weights point 1.
        p = torch.tensor([[[1., 0., 0.], [0., 0., 0.]]])
        logits = torch.zeros(1, 2)
        proxy = emitted_pose_loss(p, logits, torch.zeros(1, 3), torch.ones(1, 2))
        self.assertEqual(native_lengths(logits).item(), 1)
        self.assertEqual(float(proxy), .5)
        self.assertGreater(float(p[0, native_lengths(logits).item()-1, 0]), .3)

    def test_default_off_is_exact_old_objective(self):
        torch.manual_seed(2)
        p = torch.randn(2, 4, 3, requires_grad=True)
        logits = torch.randn(2, 4, requires_grad=True)
        target = torch.randn_like(p); mask = torch.tensor([[1.,1.,0.,0.], [1.,1.,1.,1.]])
        expected = (variable_loss_fn(p, logits, target, mask, stop_weight=.2, balance_stop=False,
                                    stop_target_mode='cumulative')
                    + .5 * endpoint_alignment_loss(p, target, mask, tail_points=8)
                    + terminal_pose_loss(p, target, mask, yaw_weight=10.))
        actual = paired_objective(p, logits, target, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(torch.autograd.grad(actual, (p,logits), retain_graph=True),
                        torch.autograd.grad(expected, (p,logits))):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_native_lengths_matches_predict_at_zero_and_no_crossing(self):
        model = MineParkingNetV1(max_horizon=4, hidden_dim=8)
        for bias in [0., -100.]:
            with torch.no_grad():
                model.decoder.stop_head.weight.zero_(); model.decoder.stop_head.bias.fill_(bias)
                out = model.forward_with_stop(torch.zeros(1,5,16,16), torch.zeros(1,3), torch.zeros(1,2))
            native = model.predict(BEVTensor(np.zeros((5,16,16),np.float32), resolution=1.,
                                             extent=(8.,8.,8.,8.), channels=list('abcde')),
                                   GoalPose(0,0,0), VehicleState(0,0,0,0,0))
            self.assertEqual(native.horizon, int(native_lengths(out.stop_logits)[0]))
            np.testing.assert_array_equal(native.points, out.points[0,:native.horizon].numpy())

    def test_reject_invalid_inputs(self):
        for logits in [torch.zeros(0, 3), torch.zeros(1, 0), torch.tensor([[float('nan')]])]:
            with self.assertRaises(ValueError): first_stop_distribution(logits)
        for temperature in [0, -1, float('inf')]:
            with self.assertRaises(ValueError): first_stop_distribution(torch.zeros(1,3), temperature=temperature)
        with self.assertRaises(ValueError):
            emitted_pose_loss(torch.zeros(1,3,3), torch.zeros(1,3), torch.zeros(1,3), torch.zeros(1,2))


if __name__ == '__main__':
    unittest.main()
