"""真实输出截断、严格车位容差和缺失证据回归。"""
import unittest
import numpy as np

from metrics.prediction_analysis import analyze_prediction_errors


def case(tolerances=True):
    prediction = np.array([[[0., 0., 0.], [1., 0., 0.], [3., 0., np.pi]]])
    target = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]])
    mask = np.array([[1., 1., 0.]])
    meta = [{'task_id': 'example', 'dataset': {'selected_goal': {'tol_pos': .3, 'tol_yaw': .1}}}]
    if not tolerances:
        meta = [{'task_id': 'example'}]
    return prediction, target, mask, meta


class TestEmittedTerminal(unittest.TestCase):
    def test_good_expert_endpoint_does_not_hide_bad_native_endpoint(self):
        p, t, m, meta = case()
        report, rows = analyze_prediction_errors(p, t, m, meta,
            stop_logits=np.array([[-20., -20., 20.]]), local_goals=np.array([[1., 0., 0.]]))
        overall = report['overall']
        self.assertEqual(overall['fde_m'], 0.)
        self.assertEqual(overall['emitted_endpoint_position_error_mean_m'], 2.)
        self.assertEqual(overall['emitted_endpoint_joint_passed'], 0)
        self.assertEqual(overall['emitted_any_joint_passed'], 1)
        self.assertEqual(overall['emitted_suffix_points'], 1)
        self.assertFalse(rows[0]['emitted_endpoint_joint_passed'])

    def test_missing_tolerance_is_not_a_pass_or_guessed_default(self):
        p, t, m, meta = case(False)
        report, _ = analyze_prediction_errors(p, t, m, meta,
            stop_logits=np.array([[-20., 20., -20.]]), local_goals=np.array([[1., 0., 0.]]))
        self.assertEqual(report['overall']['terminal_status'], 'not_observed')
        self.assertEqual(report['overall']['emitted_endpoint_joint_evaluated'], 0)
        self.assertIsNone(report['overall']['emitted_endpoint_joint_rate'])
        self.assertEqual(report['overall']['terminal_tolerance_missing_samples'], 1)

    def test_missing_local_goal_preserves_old_metrics_but_no_terminal_claim(self):
        p, t, m, meta = case()
        report, _ = analyze_prediction_errors(p, t, m, meta)
        self.assertEqual(report['overall']['fde_m'], 0.)
        self.assertEqual(report['overall']['terminal_status'], 'not_observed')

    def test_strict_boundary_and_stop_cutoff(self):
        p, t, m, meta = case()
        p[0, 0] = [.3, 0., 0.]
        p[0, 1] = [0., 0., 0.]
        report, _ = analyze_prediction_errors(p, t, m, meta,
            stop_logits=np.array([[20., -20., -20.]]), local_goals=np.zeros((1, 3)))
        self.assertEqual(report['overall']['emitted_endpoint_joint_passed'], 0)
        self.assertEqual(report['overall']['emitted_any_joint_passed'], 0)

    def test_no_stop_uses_full_horizon_and_yaw_wraps(self):
        p, t, m, meta = case()
        p[0, -1] = [1., 0., 2*np.pi-.01]
        report, rows = analyze_prediction_errors(p, t, m, meta,
            stop_logits=np.full((1, 3), -20.), local_goals=np.array([[1., 0., 0.]]))
        self.assertEqual(rows[0]['predicted_length'], 3)
        self.assertFalse(rows[0]['stop_found'])
        self.assertEqual(report['overall']['emitted_endpoint_joint_passed'], 1)
        self.assertAlmostEqual(rows[0]['emitted_endpoint_yaw_error_rad'], .01)

    def test_invalid_goals_and_tolerances_rejected(self):
        p, t, m, meta = case()
        for goal in [np.zeros((2, 3)), np.array([[np.nan, 0., 0.]])]:
            with self.assertRaises(ValueError):
                analyze_prediction_errors(p, t, m, meta, local_goals=goal)
        meta[0]['dataset']['selected_goal']['tol_pos'] = 0.
        with self.assertRaises(ValueError):
            analyze_prediction_errors(p, t, m, meta, local_goals=np.zeros((1, 3)))


if __name__ == '__main__':
    unittest.main()
