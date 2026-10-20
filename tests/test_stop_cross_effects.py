import math
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.stop_cross_effects import cross_directional_derivatives, first_stop_preservation_bound
from experiments.stop_objective_audit import audit_head_gradients, OBJECTIVES


class StopCrossEffectsTests(unittest.TestCase):
    def test_cross_dot_is_full_direction_not_group_direction(self):
        refs = {name: torch.tensor([1., 2.], dtype=torch.double)
                for name in ('original_prefix', 'original_tail', 'threshold_prefix', 'threshold_event')}
        full = {'chosen': torch.tensor([-3., 1.], dtype=torch.double)}
        result = cross_directional_derivatives(refs, full)
        self.assertEqual(result['chosen']['threshold_event'], 1.)

    def test_prefix_boundary_and_inside_outside(self):
        z = torch.tensor([[-2., -1., 1., 2.]], dtype=torch.double)
        velocity = torch.tensor([[1., 2., 0., 100.]], dtype=torch.double)
        result = first_stop_preservation_bound(z, velocity, torch.tensor([2]), threshold=.5)
        self.assertEqual(result['positive_step_supremum'], .5)
        self.assertEqual(result['limiting_point_index'], 1)
        self.assertEqual(result['limiting_constraint'], 'prefix_crossing')
        self.assertEqual(int((z + .49 * velocity >= 0).long().argmax(1)), 2)
        self.assertEqual(int((z + .51 * velocity >= 0).long().argmax(1)), 1)

    def test_event_boundary_zero_velocity_and_length_one(self):
        z = torch.tensor([[1., 2.]], dtype=torch.double)
        v = torch.tensor([[-2., 0.]], dtype=torch.double)
        result = first_stop_preservation_bound(z, v, torch.tensor([0]), threshold=.5)
        self.assertEqual(result['positive_step_supremum'], .5)
        self.assertEqual(result['limiting_constraint'], 'event_lost')
        infinite = first_stop_preservation_bound(z, torch.zeros_like(z), torch.tensor([0]), threshold=.5)
        self.assertIsNone(infinite['positive_step_supremum'])
        self.assertTrue(infinite['unbounded'])

    def test_wrong_first_stop_and_nonfinite_rejected(self):
        z = torch.tensor([[-1., 1.]])
        for stop in (torch.tensor([0]), torch.tensor([2]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                first_stop_preservation_bound(z, torch.zeros_like(z), stop, threshold=.5)
        with self.assertRaises(ValueError):
            first_stop_preservation_bound(z, torch.full_like(z, float('inf')), torch.tensor([1]), threshold=.5)
        with self.assertRaises(ValueError):
            first_stop_preservation_bound(torch.ones(1, 2), torch.zeros(1, 2), torch.tensor([1]), threshold=.5)

    def test_original_threshold_and_boundary_zero(self):
        cutoff = math.log(.26 / .74)
        z = torch.tensor([[cutoff - 1., cutoff]], dtype=torch.double)
        result = first_stop_preservation_bound(z, torch.tensor([[0., -1.]], dtype=torch.double),
                                               torch.tensor([1]), threshold=.26)
        self.assertEqual(result['positive_step_supremum'], 0.)

    def test_cli_rejects_existing_output(self):
        from scripts.audit_stop_cross_effects import run
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(Path(directory))

    def test_gradient_export_and_finite_difference(self):
        torch.manual_seed(17)
        x = torch.randn(4, 5, 3, dtype=torch.double)
        w = torch.randn(1, 3, dtype=torch.double); b = torch.randn(1, dtype=torch.double)
        lengths = torch.tensor([2, 3, 4, 5])
        saved = (w.clone(), b.clone())
        metrics, grads = audit_head_gradients(x, lengths, w, b, return_gradients=True)
        _, group = audit_head_gradients(x[:2], lengths[:2], w, b, return_gradients=True)
        cross = cross_directional_derivatives(group, {mode: grads[mode] for mode in OBJECTIVES})
        from experiments.stop_objective_audit import objective_terms
        for mode in OBJECTIVES:
            d = -grads[mode]; epsilon = 1e-6
            z = (x[:2] @ w.T).squeeze(-1) + b
            v = x[:2] @ d[:-1] + d[-1]
            before = objective_terms(z, lengths[:2], .26)['threshold_event'].mean()
            after = objective_terms(z + epsilon*v, lengths[:2], .26)['threshold_event'].mean()
            self.assertAlmostEqual(float((after-before)/epsilon), cross[mode]['threshold_event'], places=6)
            self.assertAlmostEqual(metrics[mode]['gradient_norm'], float(grads[mode].norm()), places=12)
        self.assertTrue(torch.equal(saved[0], w) and torch.equal(saved[1], b))
        self.assertTrue(all(not value.requires_grad for value in grads.values()))


if __name__ == '__main__':
    unittest.main()
