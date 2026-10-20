import unittest

import torch

from experiments.stop_fullspace_feasibility import fullspace_parent_constraints, protected_common_descent


class StopFullspaceTests(unittest.TestCase):
    def test_free_and_protected_witness(self):
        g = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.double)
        p = torch.tensor([[0., 1.]], dtype=torch.double)
        result = protected_common_descent(g, p)
        self.assertEqual(result['status'], 'verified_common_descent')
        direction = torch.tensor(result['witness']['direction'], dtype=torch.double)
        self.assertTrue((g @ direction < -1e-8).all())
        self.assertLessEqual(float((p @ direction).max()), 1e-9)

    def test_conflict_only_certifies_unit_margin_bound(self):
        g = torch.tensor([[1., 0.]], dtype=torch.double)
        p = torch.tensor([[-1., 0.]], dtype=torch.double)
        result = protected_common_descent(g, p)
        self.assertEqual(result['status'], 'certified_small_unit_margin_upper_bound')
        self.assertIsNone(result['witness'])
        self.assertLessEqual(result['dual']['unit_margin_upper_bound'], 1e-10)

    def test_budget_is_not_infeasible_and_dual_reconstructs(self):
        g = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]], dtype=torch.double)
        p = torch.empty(0, 2, dtype=torch.double)
        result = protected_common_descent(g, p, max_iterations=0)
        self.assertEqual(result['status'], 'iteration_budget_exhausted')
        coefficients = torch.tensor(result['dual']['group_coefficients'], dtype=torch.double)
        vector = coefficients @ g
        torch.testing.assert_close(vector, torch.tensor(result['dual']['vector'], dtype=torch.double))
        self.assertAlmostEqual(float(vector.norm()), result['dual']['unit_margin_upper_bound'])

    def test_dual_sparse_reconstruction_and_read_only(self):
        torch.manual_seed(4)
        g = torch.randn(4, 3, dtype=torch.double)
        p = torch.randn(9, 3, dtype=torch.double)
        saved = (g.clone(), p.clone())
        result = protected_common_descent(g, p, max_iterations=10)
        dual = result['dual']
        lam = torch.tensor(dual['group_coefficients'], dtype=torch.double)
        mu = torch.tensor(dual['protection_coefficients'], dtype=torch.double)
        self.assertTrue((lam >= 0).all() and (mu >= 0).all())
        self.assertAlmostEqual(float(lam.sum()), 1.)
        vector = lam @ g + (mu @ p[dual['active_protection_indices']] if len(mu) else 0.)
        torch.testing.assert_close(vector, torch.tensor(dual['vector'], dtype=torch.double), atol=1e-12, rtol=0)
        self.assertTrue(torch.equal(saved[0], g) and torch.equal(saved[1], p))

    def test_full_rank_linear_witness_and_cli_guards(self):
        from pathlib import Path
        import tempfile
        from scripts.audit_stop_fullspace import run
        g = torch.tensor([[1., 0.], [-1., .01]], dtype=torch.double)
        result = protected_common_descent(g, torch.empty(0, 2), max_iterations=0)
        self.assertEqual(result['status'], 'verified_common_descent')
        self.assertEqual(result['witness']['source'], 'least_squares_candidate_fully_verified')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(Path(directory))

    def test_frozen_config_guards(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from scripts import audit_stop_fullspace as cli
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / 'task'
            with patch.object(cli, 'identity', return_value={'fixed': 'same'}):
                cli.prepare(task)
                config_path = task / 'config.json'
                self.assertEqual(cli.check_config(config_path)['budgets']['max_iterations'], 4096)
                with self.assertRaises(FileExistsError):
                    cli.prepare(task)
                with patch.object(cli, 'identity', return_value={'fixed': 'changed'}):
                    with self.assertRaises(ValueError):
                        cli.check_config(config_path)
                payload = json.loads(config_path.read_text(encoding='utf-8'))
                payload['budgets']['max_iterations'] = 1
                config_path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    cli.check_config(config_path)

    def test_fullspace_protection_shape_and_invalid(self):
        x = torch.tensor([[[1.], [2.], [3.]]], dtype=torch.double)
        z = torch.tensor([[-2., -1., 1.]], dtype=torch.double)
        good = torch.tensor([[False, True, True]])
        strict, compatible = fullspace_parent_constraints(x, z, torch.tensor([2]), good, threshold=.5)
        self.assertEqual(strict.shape, (3, 2))
        torch.testing.assert_close(compatible, strict[[0, 2]])
        torch.testing.assert_close(strict.norm(dim=1), torch.ones(3, dtype=torch.double))
        with self.assertRaises(ValueError):
            protected_common_descent(torch.ones(1, 2), torch.ones(2, 3))
        with self.assertRaises(ValueError):
            protected_common_descent(torch.ones(1, 2), torch.ones(2, 2), max_seconds=0)


if __name__ == '__main__':
    unittest.main()
