import unittest
from pathlib import Path
import tempfile

import torch

from experiments.stop_direction_feasibility import (
    clip_direction_simplex, parent_protection_matrices, positive_halfspace_certificate, select_verified_direction,
)


class StopDirectionFeasibilityTests(unittest.TestCase):
    def test_feasible_polygon_and_witness(self):
        matrix = torch.tensor([[1., -1., 0.]], dtype=torch.double)
        directions = torch.tensor([[-1., 0.], [0., -1.], [-.5, -.5]], dtype=torch.double)
        gradient = torch.tensor([1., 1.], dtype=torch.double)
        result = clip_direction_simplex(matrix, torch.zeros(1), directions @ gradient)
        witness = select_verified_direction(result, matrix, directions, gradient)
        self.assertEqual(witness['status'], 'verified_nonzero_witness')
        self.assertLess(witness['unit_objective_slope'], 0)
        self.assertLessEqual(witness['max_normalized_constraint_violation'], 1e-9)

    def test_empty_budget_and_invalid(self):
        a = torch.ones(1, 3)
        self.assertEqual(clip_direction_simplex(a, torch.zeros(1), torch.zeros(3))['status'],
                         'empty_family_polygon_float64')
        self.assertEqual(clip_direction_simplex(a, torch.zeros(1), torch.zeros(3), max_cuts=0)['status'],
                         'budget_exhausted')
        with self.assertRaises(ValueError):
            clip_direction_simplex(a, torch.zeros(1), torch.zeros(3), max_cuts=-1)

    def test_degenerate_line_point_and_cancellation(self):
        a = torch.tensor([[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.]], dtype=torch.double)
        result = clip_direction_simplex(a, torch.zeros(3), torch.zeros(3))
        self.assertEqual(result['status'], 'polygon_complete')
        directions = torch.tensor([[1.], [-1.], [2.]], dtype=torch.double)
        witness = select_verified_direction(result, a, directions, torch.ones(1, dtype=torch.double))
        self.assertEqual(witness['status'], 'no_verified_witness')

    def test_goal_compatible_protection_relaxes_only_good_prefix(self):
        x = torch.tensor([[[1.], [2.], [3.]]], dtype=torch.double)
        z = torch.tensor([[-2., -1., 1.]], dtype=torch.double)
        good = torch.tensor([[False, True, True]])
        directions = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]], dtype=torch.double)
        strict, compatible = parent_protection_matrices(x, z, torch.tensor([2]), good, directions, threshold=.5)
        self.assertEqual(strict.shape, (3, 3))
        self.assertEqual(compatible.shape, (2, 3))
        torch.testing.assert_close(compatible, strict[[0, 2]])
        with self.assertRaises(ValueError):
            parent_protection_matrices(x, z, torch.tensor([2]), torch.zeros_like(good), directions, threshold=.5)

    def test_deterministic_input_read_only_and_reject_overwrite(self):
        a = torch.tensor([[1., -1., 0.], [-1., 0., 1.]], dtype=torch.double)
        saved = a.clone()
        first = clip_direction_simplex(a, torch.zeros(2), torch.tensor([1., 2., 3.]))
        self.assertEqual(first, clip_direction_simplex(a, torch.zeros(2), torch.tensor([1., 2., 3.])))
        self.assertTrue(torch.equal(a, saved))
        from scripts.audit_stop_direction_feasibility import run
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(Path(directory))

    def test_infeasibility_certificate_and_independent_recompute(self):
        a = torch.tensor([[2., -1., 2.], [-1., 2., -1.]], dtype=torch.double)
        result = positive_halfspace_certificate(a)
        self.assertEqual(result['status'], 'verified_family_infeasibility_certificate')
        coefficients = torch.tensor(result['nonnegative_coefficients'], dtype=torch.double)
        combined = coefficients @ a[result['active_offsets']]
        self.assertTrue((combined > 1e-9).all())
        self.assertAlmostEqual(float(combined.min()), result['positive_gap'])
        self.assertEqual(positive_halfspace_certificate(a, max_subsets=1)['status'], 'certificate_budget_exhausted')
        self.assertEqual(positive_halfspace_certificate(-torch.ones(1, 3))['status'], 'no_certificate_found')


if __name__ == '__main__':
    unittest.main()
