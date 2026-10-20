import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.stop_endpoint_intervals import affine_cache_logits, first_stop_intervals, earliest_selected_interval, event_record


class StopEndpointIntervalTests(unittest.TestCase):
    def test_open_prefix_and_closed_event_boundary(self):
        z = torch.tensor([[1., -2.]], dtype=torch.double)
        v = torch.tensor([[-1., 0.]], dtype=torch.double)
        intervals = first_stop_intervals(z, v, threshold=.5)
        self.assertTrue(intervals['upper_closed'][0, 0])
        self.assertEqual(float(intervals['upper'][0, 0]), 1.)
        self.assertFalse(intervals['lower_closed'][0, 1])
        self.assertEqual(float(intervals['lower'][0, 1]), 1.)

    def test_fallback_and_horizon_one(self):
        z = torch.tensor([[-2., -3., -4.]])
        intervals = first_stop_intervals(z, torch.zeros_like(z), threshold=.5)
        self.assertEqual(intervals['nonempty'].tolist(), [[False, False, True]])
        one = first_stop_intervals(torch.tensor([[-3.]]), torch.ones(1, 1), threshold=.5)
        self.assertTrue(one['nonempty'][0, 0])
        self.assertTrue(torch.isinf(one['upper'][0, 0]))

    def test_simultaneous_root_does_not_create_spurious_event(self):
        z = torch.tensor([[-1., -1., -2.]], dtype=torch.double)
        v = torch.tensor([[1., 1., 0.]], dtype=torch.double)
        intervals = first_stop_intervals(z, v, threshold=.5)
        self.assertFalse(intervals['nonempty'][0, 1])
        self.assertEqual(float(intervals['lower'][0, 0]), 1.)
        self.assertTrue(intervals['lower_closed'][0, 0])

    def test_endpoint_may_move_before_first_bad_event(self):
        z = torch.tensor([[1., .5, -.1]], dtype=torch.double)
        v = torch.tensor([[-1., 0., 1.]], dtype=torch.double)
        intervals = first_stop_intervals(z, v, threshold=.5)
        good = torch.tensor([[True, True, False]])
        bad = earliest_selected_interval(intervals, ~good)
        self.assertIsNone(event_record(intervals, bad, 0))

    def test_random_small_arrangements_match_direct_evaluation(self):
        rng = np.random.default_rng(19)
        z = rng.normal(size=(12, 6)); v = rng.normal(size=(12, 6)); v[0, :2] = 0
        intervals = first_stop_intervals(torch.from_numpy(z), torch.from_numpy(v), threshold=.5)
        for row in range(len(z)):
            roots = sorted({0., *(float(-z[row, j]/v[row, j]) for j in range(6) if v[row,j] and -z[row,j]/v[row,j]>0)})
            samples = [(a+b)/2 for a,b in zip(roots, roots[1:])] + [roots[-1]+1.]
            for eta in samples:
                hits = np.flatnonzero(z[row]+eta*v[row]>=0)
                actual = int(hits[0]) if len(hits) else 5
                included = []
                for j in range(6):
                    lo, hi = float(intervals['lower'][row,j]), float(intervals['upper'][row,j])
                    if intervals['nonempty'][row,j] and lo < eta < hi:
                        included.append(j)
                self.assertEqual(included, [actual])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            first_stop_intervals(torch.empty(0, 2), torch.empty(0, 2), threshold=.5)
        with self.assertRaises(ValueError):
            first_stop_intervals(torch.ones(1, 2), torch.ones(1, 2), threshold=1.)

    def test_affine_partial_final_chunk_stops_at_declared_count(self):
        x=torch.randn(11,4,3);w=torch.randn(1,3);b=torch.randn(1);d=torch.randn(4,dtype=torch.double)
        base,velocity=affine_cache_logits(x,w,b,d,count=9,chunk_size=4)
        self.assertEqual(base.shape,(9,4))
        torch.testing.assert_close(base,(x[:9].double()@w.double().T).squeeze(-1)+b.double())
        torch.testing.assert_close(velocity,x[:9].double()@d[:-1]+d[-1])

    def test_zero_only_singleton_and_input_preservation(self):
        z=torch.tensor([[0.,-2.]],dtype=torch.double)
        v=torch.tensor([[-1.,0.]],dtype=torch.double)
        saved_z=z.clone(); saved_v=v.clone()
        intervals=first_stop_intervals(z,v,threshold=.5)
        self.assertTrue(intervals['nonempty'][0,0])
        self.assertEqual(float(intervals['lower'][0,0]),0.)
        self.assertEqual(float(intervals['upper'][0,0]),0.)
        self.assertTrue(intervals['lower_closed'][0,0])
        self.assertTrue(intervals['upper_closed'][0,0])
        self.assertFalse(intervals['lower_closed'][0,1])
        torch.testing.assert_close(z,saved_z,rtol=0,atol=0)
        torch.testing.assert_close(v,saved_v,rtol=0,atol=0)

    def test_cli_refuses_existing_output_before_loading_cache(self):
        from scripts.audit_stop_endpoint_intervals import run
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileExistsError):
                run(Path(name))


if __name__ == '__main__':
    unittest.main()
