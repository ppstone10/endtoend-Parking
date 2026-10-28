import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from interfaces import VehicleState, GoalPose, Trajectory, BEVTensor
from runtime import NetworkSource
from experiments.component_capture import VerificationBudget
from verification.candidate_layer_quality import LAYERS, episode_verdict, aggregate_verdicts
from verification.candidate_layer_capture import capture_inputs_scope
from verification.candidate_layer_study import verify_manifest
from verification.perception_calibration import fit_profile, evaluate_profile, calibration_selection


def facts(**changes):
    base = dict(inputs_complete=True, physical_unsafe_references=0, margin_unsafe_references=0,
                near_goal_queries=1, near_goal_unsupported=0, unsafe_actual_steps=0,
                unsafe_horizons=0, success=True, steps=10)
    return {**base, **changes}


def sample(seed, value=.8, target=True):
    return {'seed': seed, 'group': f'S1|low|target={int(target)}', 'target_observed': target,
            'metrics': dict(occupancy_iou=value, occupancy_recall=value,
                            target_iou=value if target else None, target_hit_rate=value if target else None)}


class LayerQualityTests(unittest.TestCase):
    def verdict(self, **changes):
        return episode_verdict(facts(**changes), {name: [] for name in LAYERS}, ['passed'])

    def test_reference_risk_fails_despite_exact_replay(self):
        layers = self.verdict(margin_unsafe_references=1)
        self.assertEqual(layers['L2']['contract_status'], 'passed')
        self.assertEqual(layers['L2']['quality_status'], 'failed')
        self.assertFalse(layers['L2']['module_qualified'])
        self.assertEqual(layers['L1_L2']['quality_status'], 'failed')

    def test_safe_stopping_does_not_pass_coupled_task(self):
        layers = self.verdict(success=False)
        self.assertTrue(layers['L3']['covered_scope_qualified'])
        self.assertEqual(layers['L2_L3']['quality_failures'], ['original_task_not_reached'])

    def test_terminal_missing_and_not_observed_are_distinct(self):
        self.assertEqual(self.verdict(near_goal_unsupported=1)['L2']['quality_status'], 'failed')
        self.assertEqual(self.verdict(near_goal_queries=0)['L2']['quality_status'], 'blocked')

    def test_no_profile_or_absolute_gate_cannot_create_perception_qualification(self):
        errors = {name: [] for name in LAYERS}
        blocked = episode_verdict(facts(), errors, ['blocked'])
        self.assertEqual(blocked['L1']['quality_status'], 'blocked')
        passed = self.verdict()
        self.assertTrue(passed['L1']['covered_scope_qualified'])
        self.assertFalse(passed['L1']['module_qualified'])
        self.assertEqual(passed['L1']['absolute_quality_status'], 'not_established')

    def test_missing_evidence_and_contract_errors_block_qualification(self):
        for name, row in self.verdict(inputs_complete=False).items():
            self.assertFalse(row['module_qualified'], name)
        errors = {name: [] for name in LAYERS};errors['L3'] = [{'error': 'control_replay'}]
        self.assertFalse(episode_verdict(facts(), errors, ['passed'])['L3']['module_qualified'])

    def test_aggregate_preserves_failed_and_blocked_cases(self):
        rows = [{'dataset_index': 1, 'layers': self.verdict()},
                {'dataset_index': 2, 'layers': self.verdict(near_goal_queries=0)},
                {'dataset_index': 3, 'layers': self.verdict(margin_unsafe_references=1)}]
        result = aggregate_verdicts(rows)['L2']
        self.assertEqual(result['quality_status'], 'failed')
        self.assertEqual(result['failed_indices'], [3])
        self.assertEqual(result['blocked_indices'], [2])
        with self.assertRaises(ValueError):
            aggregate_verdicts([])


class PerceptionReferenceTests(unittest.TestCase):
    def test_fit_and_independent_holdout_and_unsupported_group(self):
        rows = [sample(seed) for seed in range(45)]
        profile = fit_profile(rows)
        group = profile['groups']['S1|low|target=1']
        self.assertEqual(group['status'], 'ready')
        self.assertFalse(set(group['fit_seeds']) & set(group['holdout_seeds']))
        self.assertEqual(evaluate_profile(profile, sample(99)), 'passed')
        self.assertEqual(evaluate_profile(profile, sample(99, .1)), 'failed')
        self.assertEqual(evaluate_profile(profile, sample(99, target=False)), 'blocked')

    def test_preflight_insufficient_and_empty_truth_cannot_pass(self):
        rows = [sample(seed) for seed in range(45)]
        self.assertEqual(evaluate_profile(fit_profile(rows, preflight=True), rows[0]), 'blocked')
        self.assertEqual(evaluate_profile(fit_profile(rows[:2]), rows[0]), 'blocked')
        profile = fit_profile(rows);observed = sample(99);observed['metrics']['occupancy_iou'] = None
        self.assertEqual(evaluate_profile(profile, observed), 'blocked')

    def test_holdout_failure_does_not_refit_bounds(self):
        rows = [sample(seed, .1 if seed % 3 == 0 else .8) for seed in range(45)]
        group = fit_profile(rows)['groups']['S1|low|target=1']
        self.assertEqual(group['status'], 'failed')
        self.assertEqual(group['bounds']['occupancy_iou'], .8)

    def test_duplicate_seed_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            fit_profile([sample(0), sample(0)])
        with self.assertRaises(ValueError):
            fit_profile([sample(0, float('nan'))])

    def test_training_selection_excludes_val_recovery_and_duplicate_seed(self):
        def meta(seed, **changes):
            return dict(seed=seed, task_id=str(seed), scene_name='S1', difficulty={'noise_level': 'low'}, **changes)
        rows = [meta(0), meta(1), meta(1), meta(2, recovery=True), meta(3)]
        self.assertEqual(calibration_selection(rows, [meta(0)]), [1, 4])


class InputCaptureTests(unittest.TestCase):
    def test_exception_restores_engine_capture_predict_and_source(self):
        from experiments import validation_matrix as matrix
        class Pipe:
            def set_target_goals(self, goals):
                pass
            def capture_bev(self, x, y, yaw):
                return BEVTensor(np.ones((1, 2, 2), dtype=np.float32), 1., (1., 1., 1., 1.), ['occupancy'])
        pipeline = Pipe()
        model = SimpleNamespace(predict=lambda bev, goal, state: Trajectory(np.array([[0., 0., 0.]]), .2))
        primary = NetworkSource(pipeline, model)
        class Source:
            step = 0
            raw_references = {}
            def __init__(self):
                self.primary = primary
            def next_trajectory(self, state):
                trajectory, elapsed = primary.next_trajectory(state)
                self.raw_references['raw_0000'] = trajectory.points
                return trajectory, elapsed
        source = Source()
        class Engine:
            def __init__(self):
                self.source = source
            def run(self, start, goal):
                primary.begin(start, goal)
                self.source.next_trajectory(start)
                raise RuntimeError('fixture')
        old_capture, old_predict, old_next = pipeline.capture_bev, model.predict, source.next_trajectory
        with patch.object(matrix, 'ClosedLoopEngine', Engine), patch(
                'verification.candidate_layer_capture.pipeline_semantics', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with capture_inputs_scope('unused', VerificationBudget(1)):
                    matrix.ClosedLoopEngine().run(VehicleState(0., 0., 0.), GoalPose(0., 0., 0.))
            self.assertIs(matrix.ClosedLoopEngine, Engine)
        self.assertEqual(pipeline.capture_bev, old_capture)
        self.assertIs(model.predict, old_predict)
        self.assertEqual(source.next_trajectory, old_next)

    def test_manifest_missing_and_path_escape_rejected(self):
        from training.reporting import atomic_write_json
        with TemporaryDirectory() as directory:
            path = Path(directory);atomic_write_json(path/'artifacts.json', {})
            with self.assertRaises(ValueError):
                verify_manifest(path, ['report.json'])
            atomic_write_json(path/'artifacts.json', {'../outside.json': 'bad'})
            with self.assertRaises(ValueError):
                verify_manifest(path, [])

    def test_manifest_rejects_changed_file(self):
        from training.reporting import atomic_write_json
        from scripts.diagnose_closed_loop_failures import _sha256
        with TemporaryDirectory() as directory:
            path = Path(directory);(path/'report.json').write_text('{}', encoding='utf-8')
            atomic_write_json(path/'artifacts.json', {'report.json': _sha256(path/'report.json')})
            (path/'report.json').write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_manifest(path, ['report.json'])

    def test_profile_bounds_recomputed_after_manifest_verification(self):
        from verification.perception_calibration import load_profile
        from training.reporting import atomic_write_json
        from scripts.diagnose_closed_loop_failures import _sha256
        with TemporaryDirectory() as directory:
            path = Path(directory);config = {'code': {'fixture': 'hash'}, 'inputs': {}}
            atomic_write_json(path/'config.json', config)
            observations = [sample(seed) for seed in range(45)]
            profile = fit_profile(observations)
            profile.update(observations=observations, code=config['code'], inputs={},
                           config_sha256=_sha256(path/'config.json'))
            report = dict(status='completed', config_sha256=profile['config_sha256'], tasks=45, ready_groups=1)
            atomic_write_json(path/'report.json', report)
            def save():
                atomic_write_json(path/'profile.json', profile)
                atomic_write_json(path/'artifacts.json', {name: _sha256(path/name)
                    for name in ('config.json', 'report.json', 'profile.json')})
            save();self.assertEqual(load_profile(path, config)['groups'], profile['groups'])
            profile['groups']['S1|low|target=1']['bounds']['occupancy_iou'] = .1
            save()
            with self.assertRaisesRegex(ValueError, '复现'):
                load_profile(path, config)


if __name__ == '__main__':
    unittest.main()
