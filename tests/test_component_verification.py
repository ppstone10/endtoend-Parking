"""Production verification must expose failures without manufacturing a link diagnosis."""
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from experiments.component_verification import acceptance, locate_failure, pose_errors, validate_baseline


def baseline_fixture():
    return {'status': 'completed', 'experiment': {'name': 'E5', 'bev_source': 'sensor',
            'trajectory_source': 'network', 'executor': 'mpc_vehicle', 'safety_mode': 'none'},
            'protocol': {'evaluated_indices': [0], 'selected_indices': [0],
                         'max_steps': 600, 'replan_every': 10, 'control_seed': 0, 'mpc_dt': .1},
            'episodes': [{'meta': {'dataset_index': 0}}], 'overall': {}}


class ComponentVerificationTests(unittest.TestCase):
    def test_archived_projection_reaches_network_replay_without_relabeling(self):
        import json
        from types import SimpleNamespace
        from experiments import component_analysis as analysis
        from interfaces import BEVTensor
        from interfaces.bev import assert_camera_projection_matches

        class ReplayObserved(Exception):
            pass

        for projection in ('per_view_v2', 'legacy_overlay_v1', None):
            with self.subTest(projection=projection), tempfile.TemporaryDirectory() as directory:
                data = np.zeros((5, 2, 2), dtype=np.float32)
                bev = BEVTensor(data, 1., (1., 1., 1., 1.), list('abcde'), projection)
                metadata = bev.to_metadata()
                source = Path(directory) / 'episode-0000'
                source.with_suffix('.json').write_text(json.dumps({
                    'baseline_match': True, 'npz_sha256': 'verified', 'boundary_errors': [],
                    'frames': [{}], 'inputs': [{'bev_meta': metadata, 'target_channel': 'image',
                        'height_mode': 'semantic', 'capture_to_predict_equal': True,
                        'capture_pose': [0., 0., 0.], 'state': [0.] * 5, 'goal': [0.] * 3}],
                }), encoding='utf-8')
                np.savez(source.with_suffix('.npz'), plan_issue_steps=[0],
                    plan_issue_states=np.zeros((1, 5)), states_before=np.zeros((1, 5)),
                    states_after=np.zeros((1, 5)), initial_state=np.zeros(5),
                    issue_bevs=data[None], goal=np.zeros(3),
                    local_prediction_0000=np.zeros((1, 3)), global_plan_0000=np.zeros((1, 3)),
                    plan_dt_0000=.2)

                def predict(replayed_bev, *args):
                    expected = projection or 'legacy_overlay_v1'
                    assert_camera_projection_matches(replayed_bev.camera_projection, expected)
                    self.assertEqual(replayed_bev.to_metadata(), metadata)
                    np.testing.assert_array_equal(replayed_bev.data, data)
                    raise ReplayObserved()

                pipeline = SimpleNamespace(set_target_goals=lambda goals: None,
                    capture_bev=lambda *pose: bev, bev_config=None,
                    lidar_sensor=SimpleNamespace(max_range=10.))
                restored = SimpleNamespace(task=SimpleNamespace(scene=SimpleNamespace(env=None)),
                                           goal=SimpleNamespace(x=0., y=0., yaw=0.),
                                           tol_pos=.3, tol_yaw=.1)
                from sim import MINING_DRILL_RIG
                loaded = SimpleNamespace(model=SimpleNamespace(predict=predict),
                    model_config={'bev_channels': 5, 'target_channel': 'image', 'height_mode': 'semantic'})
                fidelity = SimpleNamespace(to_dict=lambda: {'height_mae': 0., 'density_mae': 0.})
                with patch.object(analysis.matrix, 'reconstruct_dataset_task', return_value=restored), \
                        patch.object(analysis.matrix, '_build_pipeline', return_value=(SimpleNamespace(_collision_checker=None), pipeline)), \
                        patch('scripts.diagnose_closed_loop_failures._sha256', return_value='verified'), \
                        patch.object(analysis, 'rasterize_lidar_truth_occupancy'), \
                        patch.object(analysis, 'rasterize_ground_truth_target'), \
                        patch.object(analysis, 'compute_bev_fidelity_metrics', return_value=fidelity):
                    with self.assertRaises(ReplayObserved):
                        analysis.analyze_episode(directory, 0,
                            {'protocol': {'control_seed': 0, 'replan_every': 10}},
                            {'task_meta': [{}]}, {'seed': 0}, MINING_DRILL_RIG,
                            loaded, SimpleNamespace(check=lambda: None))

    def test_local_checks_do_not_prove_link_fault(self):
        result = locate_failure(success=False, collision=False, boundary_errors=[],
                                reference_unsafe=False, connection_unsafe=False,
                                execution_unsafe=False)
        self.assertEqual(result, 'unresolved_nonarrival')

    def test_collision_with_safe_reference_fails_control_layer(self):
        result = locate_failure(success=False, collision=True, boundary_errors=[],
                                reference_unsafe=False, connection_unsafe=False,
                                execution_unsafe=True)
        self.assertEqual(result, 'execution_left_safe_reference')

    def test_measured_boundary_error_is_explicit(self):
        self.assertEqual(locate_failure(success=False, collision=False,
                                       boundary_errors=['local_global'], reference_unsafe=False,
                                       connection_unsafe=False, execution_unsafe=False),
                         'boundary_mismatch')

    def test_small_center_error_does_not_hide_heading_error(self):
        position, yaw = pose_errors([0.01, 0, 0.1], [0, 0, 0])
        self.assertLess(position, 0.02)
        self.assertGreater(yaw, 0.09)

    def test_smoke_and_missing_checks_cannot_qualify(self):
        rows = [{'success': True, 'collision': False}] * 3
        self.assertFalse(acceptance(rows, full_coverage=False, checks_ok=True)['production_qualified'])
        rows *= 100
        self.assertFalse(acceptance(rows, full_coverage=True, checks_ok=False)['production_qualified'])

    def test_joint_thresholds_and_exact_denominator(self):
        rows = ([{'success': True, 'collision': False}] * 210 +
                [{'success': False, 'collision': True}] * 30 +
                [{'success': False, 'collision': False}] * 60)
        self.assertTrue(acceptance(rows, full_coverage=True, checks_ok=True)['production_qualified'])
        self.assertFalse(acceptance(rows[:-1], full_coverage=True, checks_ok=True)['production_qualified'])

    def test_diagnostic_substitutions_and_incomplete_denominator_rejected(self):
        baseline = baseline_fixture()
        self.assertEqual(list(validate_baseline(baseline, [0])), [0])
        for field, value in [('executor', 'ideal_path'), ('bev_source', 'gt'), ('safety_mode', 'expert_fallback')]:
            altered = deepcopy(baseline)
            altered['experiment'][field] = value
            with self.assertRaises(ValueError):
                validate_baseline(altered, [0])
        for indices in ([], [0, 0], [1]):
            with self.assertRaises(ValueError):
                validate_baseline(baseline, indices)
        altered = deepcopy(baseline)
        altered['protocol']['selected_indices'] = [0, 1]
        with self.assertRaises(ValueError):
            validate_baseline(altered, [0])
        altered['protocol']['selected_indices'] = [0, 0]
        with self.assertRaises(ValueError):
            validate_baseline(altered, [0])

    def test_capture_factories_restore_after_native_error(self):
        from experiments import validation_matrix as matrix
        from experiments.component_capture import capture_native_e5, VerificationBudget
        original_engine, original_builder = matrix.ClosedLoopEngine, matrix._build_source
        baseline = baseline_fixture()
        baseline['protocol'].update(data='unused', checkpoint='unused')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            with patch.object(matrix, 'run_validation_experiment', side_effect=RuntimeError('native-error')):
                with self.assertRaisesRegex(RuntimeError, 'native-error'):
                    capture_native_e5(baseline, output, [0], VerificationBudget(1))
            self.assertIs(matrix.ClosedLoopEngine, original_engine)
            self.assertIs(matrix._build_source, original_builder)
            with self.assertRaises(FileExistsError):
                capture_native_e5(baseline, output, [0], VerificationBudget(1))

    def test_budget_stops_without_autocontinuation(self):
        from experiments.component_capture import VerificationBudget
        with patch('experiments.component_capture.time.perf_counter', side_effect=[0, 2]):
            budget = VerificationBudget(1)
            with self.assertRaises(TimeoutError):
                budget.check()

    def test_production_entrypoints_and_explicit_legacy_mode(self):
        import scripts.evaluate_bev_fidelity as l1
        import scripts.evaluate_visited_state_trajectory as l2
        import scripts.evaluate_single_cycle_tracking as l3
        for script, stage in [(l1, 'L1'), (l2, 'L2'), (l3, 'L3')]:
            with patch('sys.argv', ['entry', '--phase', 'inspect', '--output', 'unused']), \
                    patch('scripts.verify_closed_loop_components.main') as production:
                script.main()
                production.assert_called_once_with(stage=stage)
            with patch('sys.argv', ['entry', '--legacy-diagnostic']), patch.object(script, 'legacy_main') as legacy:
                script.main()
                legacy.assert_called_once_with()

    def test_transform_preserves_yaw_and_float32_handoff(self):
        from experiments.component_analysis import global_prediction, local_goal
        local = np.asarray([[1, 0, -.1], [2, 1, .2]], dtype=np.float32)
        state = [3, 4, np.pi / 2, .1, .2]
        result = global_prediction(local, state)
        np.testing.assert_allclose(result[:, :2], [[3, 5], [2, 6]], atol=1e-6)
        self.assertEqual(result.dtype, local.dtype)
        np.testing.assert_allclose(local_goal([3, 5, np.pi / 2], state), [1, 0, 0], atol=1e-12)

    def test_transform_uses_float32_rounding_tolerance_but_rejects_axis_error(self):
        from experiments.component_analysis import global_prediction, transform_matches
        local = np.asarray([[1, 0, -.1], [2, 1, .2]], dtype=np.float32)
        state = [3, 4, np.pi / 2, .1, .2]
        world = global_prediction(local, state)
        rounded = np.nextafter(world, np.float32(np.inf))
        self.assertTrue(transform_matches(local, state, rounded))
        wrong = world.copy()
        wrong[:, :2] = wrong[:, 1::-1]
        self.assertFalse(transform_matches(local, state, wrong))

    def test_incomplete_report_is_not_inspectable(self):
        from scripts.verify_closed_loop_components import inspect_completed
        from training.reporting import atomic_write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            atomic_write_json(path / 'report.json', {'schema': 'production_component_verification_v1', 'status': 'incomplete'})
            with self.assertRaises(ValueError):
                inspect_completed(path)

    def test_existing_output_is_not_overwritten(self):
        from scripts.verify_closed_loop_components import run
        with tempfile.TemporaryDirectory() as directory, patch('scripts.verify_closed_loop_components.precheck', return_value=({}, {})):
            with self.assertRaises(FileExistsError):
                run('unused', directory)

    def test_capture_risks_include_full_yaw_footprint_not_center_only(self):
        from scripts.diagnose_reference_execution import window_risks

        class NarrowBay:
            def pose_free(self, x, y, yaw):
                # 6x3矩形旋转后横向外廓大于bay半宽即危险；中心可保持不变。
                return 3 * abs(np.sin(yaw)) + 1.5 * abs(np.cos(yaw)) < 1.6

            def swept_segment_free(self, a, b):
                return all(self.pose_free(*(a[:3] + t * (b[:3] - a[:3]))) for t in np.linspace(0, 1, 30))

        before = np.zeros(5)
        after = np.asarray([.01, 0, .1, 0, 0])
        window = np.asarray([[.01, 0, 0], [.02, 0, 0]])
        risks = window_risks(before, window, after, np.asarray([after]), NarrowBay())
        self.assertIsNone(risks['reference_internal_blocked_index'])
        self.assertTrue(risks['connection_safe'])
        self.assertFalse(risks['actual_transition_safe'])

    def test_reference_demands_wrap_yaw_and_expose_untrackable_speed(self):
        from experiments.component_verification import trajectory_demands
        points = [[0, 0, np.pi - .01], [.1, 0, -np.pi + .01], [1, 0, -np.pi + .02]]
        demand = trajectory_demands(points, .1, max_v=2, max_omega=1)
        self.assertEqual(demand['segments_over_vehicle_speed'], 1)
        self.assertEqual(demand['segments_over_vehicle_yaw_rate'], 0)

    def test_artifact_tampering_is_rejected_before_inspection(self):
        from scripts.verify_closed_loop_components import inspect_completed
        from experiments.component_verification import SCHEMA
        from scripts.diagnose_closed_loop_failures import _sha256
        from training.reporting import atomic_write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {'schema': SCHEMA, 'status': 'completed', 'coverage': {'requested_indices': []}}
            atomic_write_json(root / 'report.json', report)
            names = ['config.json', 'capture/replayed-report.json', 'summary.md']
            names += [f'layers/{name}.json' for name in ('L1', 'L2', 'L3', 'L1_L2', 'L2_L3')]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}', encoding='utf-8')
            artifacts = {name.replace('/', '\\'): _sha256(root / name) for name in names + ['report.json']}
            atomic_write_json(root / 'artifacts.json', artifacts)
            with patch('scripts.verify_closed_loop_components.precheck', side_effect=RuntimeError('artifact-check-passed')):
                with self.assertRaisesRegex(RuntimeError, 'artifact-check-passed'):
                    inspect_completed(root)
            (root / 'summary.md').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '漂移'):
                inspect_completed(root)


if __name__ == '__main__':
    unittest.main()
