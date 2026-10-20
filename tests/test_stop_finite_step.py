import unittest

import torch

from experiments.stop_finite_step import quarter_bound_head, cached_float32_logits, finite_loss_changes


class StopFiniteStepTests(unittest.TestCase):
    def test_single_update_dtype_and_parent_unchanged(self):
        head = torch.nn.Linear(2, 1); before = {k: v.clone() for k, v in head.state_dict().items()}
        direction = torch.tensor([1., 2., 3.], dtype=torch.double); direction /= direction.norm()
        candidate, eta = quarter_bound_head(head, direction, .01)
        self.assertEqual(eta, .0025)
        self.assertTrue(all(torch.equal(v, head.state_dict()[k]) for k, v in before.items()))
        self.assertEqual(candidate.weight.dtype, torch.float32)
        torch.testing.assert_close(candidate.weight, (before['weight'].double() + eta*direction[:-1][None]).float(), atol=0, rtol=0)

    def test_cached_stepwise_shape_and_loss_delta(self):
        head = torch.nn.Linear(2, 1); x = torch.randn(9, 4, 2)
        logits = cached_float32_logits(head, x)
        self.assertEqual(logits.shape, (9, 4))
        torch.testing.assert_close(logits[:8, 0], head(x[:8, 0].contiguous()).squeeze(1))
        report = finite_loss_changes(logits, logits, torch.ones(9, dtype=torch.long)*3,
                                     {'all': list(range(9))}, threshold=.26)
        self.assertEqual(report['float64_reduction']['all']['threshold_prefix']['delta'], 0.)

    def test_invalid_bound_and_direction(self):
        head = torch.nn.Linear(2, 1)
        for bound in (0., float('inf'), -1.):
            with self.assertRaises(ValueError):
                quarter_bound_head(head, torch.ones(3), bound)
        with self.assertRaises(ValueError):
            quarter_bound_head(head, torch.ones(3), .01)

    def test_cli_refuses_existing_output(self):
        from pathlib import Path
        import tempfile
        from scripts.probe_stop_finite_step import run
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(Path(directory))


if __name__ == '__main__':
    unittest.main()
