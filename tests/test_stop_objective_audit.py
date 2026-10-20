import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.stop_objective_audit import OBJECTIVES, audit_head_gradients, stop_objective


class StopObjectiveTests(unittest.TestCase):
    def test_original_equivalence(self):
        z = torch.tensor([[1., -2., 3.], [0., 2., -1.]])
        lengths = torch.tensor([2, 3])
        labels = (torch.arange(3)[None] >= lengths[:, None] - 1).float()
        torch.testing.assert_close(stop_objective(z, lengths, objective='cumulative'),
                                   .2 * F.binary_cross_entropy_with_logits(z, labels))

    def test_formula_and_tail_zero_gradient(self):
        a = math.log(.26 / .74)
        z = torch.tensor([[0., 2., 1., 8.]], dtype=torch.double, requires_grad=True)
        lengths = torch.tensor([3])
        worst = stop_objective(z, lengths, objective='worst_prefix')
        expected = .1 * (F.softplus(z[0, 1] - a) + F.softplus(a - z[0, 2]))
        torch.testing.assert_close(worst, expected)
        for mode in ('prefix_event', 'worst_prefix'):
            grad, = torch.autograd.grad(stop_objective(z, lengths, objective=mode), z)
            self.assertEqual(float(grad[0, 3]), 0.)
            self.assertLess(float(grad[0, 2]), 0.)

    def test_length_one_horizon_and_extreme_logits(self):
        for length in (1, 320):
            for mode in OBJECTIVES:
                z = torch.full((1, 320), 1e4, requires_grad=True)
                loss = stop_objective(z, torch.tensor([length]), objective=mode)
                grad, = torch.autograd.grad(loss, z)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(grad).all())
                if length == 1 and mode != 'cumulative':
                    self.assertTrue(torch.equal(grad[:, 1:], torch.zeros_like(grad[:, 1:])))

    def test_max_ties_share_gradient(self):
        z = torch.zeros(1, 3, requires_grad=True)
        loss = stop_objective(z, torch.tensor([3]), objective='worst_prefix')
        grad, = torch.autograd.grad(loss, z)
        self.assertEqual(float(grad[0, 0]), float(grad[0, 1]))
        self.assertGreater(float(grad[0, 0]), 0.)

    def test_chunking_and_read_only(self):
        torch.manual_seed(3)
        x = torch.randn(5, 6, 4); w = torch.randn(1, 4); b = torch.randn(1)
        saved = [v.clone() for v in (x, w, b)]
        a = audit_head_gradients(x, torch.tensor([1, 2, 3, 5, 6]), w, b, chunk_size=5)
        c = audit_head_gradients(x, torch.tensor([1, 2, 3, 5, 6]), w, b, chunk_size=2)
        for mode in OBJECTIVES:
            self.assertAlmostEqual(a[mode]['loss'], c[mode]['loss'], places=12)
            for key in a[mode]['directional_derivatives']:
                self.assertAlmostEqual(a[mode]['directional_derivatives'][key], c[mode]['directional_derivatives'][key], places=12)
        self.assertTrue(all(torch.equal(v, s) for v, s in zip((x, w, b), saved)))
        self.assertTrue(all(v.grad is None for v in (x, w, b)))

    def test_invalid_inputs(self):
        for lengths in (torch.tensor([0]), torch.tensor([4]), torch.tensor([1.5]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                stop_objective(torch.zeros(1, 3), lengths, objective='prefix_event')
        for threshold in (0., 1., float('nan')):
            with self.assertRaises(ValueError):
                stop_objective(torch.zeros(1, 3), torch.tensor([2]), objective='worst_prefix', threshold=threshold)
        with self.assertRaises(ValueError):
            stop_objective(torch.zeros(1, 3), torch.tensor([2]), objective='unknown')
        for z in (torch.zeros(0, 3), torch.tensor([[float('inf')]])):
            with self.assertRaises(ValueError):
                stop_objective(z, torch.ones(len(z), dtype=torch.long), objective='prefix_event')

    def test_cli_rejects_drift_and_existing_output(self):
        from scripts import audit_stop_objectives as cli
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                cli.run(root)
            (root / 'config.json').write_text('{}')
            with patch.object(cli, 'PRIOR', root):
                with self.assertRaisesRegex(ValueError, '身份漂移'):
                    cli.load_verified_inputs()


if __name__ == '__main__':
    unittest.main()
