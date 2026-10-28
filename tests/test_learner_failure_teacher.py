from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.learner_failure_teacher import failure_contexts, readiness, validated_query
from interfaces import VehicleState, GoalPose, Trajectory


class FailureTeacherTests(unittest.TestCase):
    def test_contexts_stay_before_first_loss_and_keep_original_time(self):
        contexts = [SimpleNamespace(step=s) for s in range(0, 600, 10)]
        margins = [False] * 90 + [True] + [False] * 509
        self.assertEqual([c.step for c in failure_contexts(contexts, margins)], [10, 50, 90])
        self.assertEqual([c.step for c in failure_contexts(contexts, [False] * 600)], [10, 200, 400])
        self.assertEqual(failure_contexts(contexts, [True] + [False] * 599), [])
        with self.assertRaises(ValueError):
            failure_contexts([SimpleNamespace(step=15)], [False] * 600)

    def test_gate_requires_all_sources_latest_query_and_new_input(self):
        rows = [dict(seed=s, dataset_index=i, queries=[
            dict(step=10, decision='strict_query_accepted', input_sha256=f'{s}-{i}-10'),
            dict(step=80, decision='strict_query_accepted', input_sha256=f'{s}-{i}-80')])
            for s, i in ((20261020, 1196), (20261021, 1196), (20261020, 2149))]
        new = {q['input_sha256'] for r in rows for q in r['queries']}
        self.assertTrue(readiness(rows, new)['passed'])
        self.assertFalse(readiness(rows[:-1], new)['passed'])
        self.assertFalse(readiness(rows, new - {'20261020-2149-80'})['passed'])
        rows[0]['queries'][-1]['decision'] = 'teacher_rejected'
        self.assertFalse(readiness(rows, new)['passed'])

    def test_precheck_has_no_rollout_and_existing_output_is_untouched(self):
        from scripts.preflight_learner_failure_teacher import precheck, run
        with patch('scripts.preflight_learner_failure_teacher.prepare_config', return_value={}), \
             patch('scripts.preflight_learner_failure_teacher.collect') as collect:
            precheck(); collect.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_learner_failure_teacher.prepare_config') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_teacher_first_window_uses_actual_continuation_not_learner_suffix(self):
        checker = SimpleNamespace(pose_free=lambda *args: True,
            swept_segment_free=lambda a, b: a[0] <= 1 and b[0] <= 1)
        case = dict(index=1196, start=VehicleState(0, 0, 0), goal=GoalPose(0, 0, 0),
                    planner=SimpleNamespace(_collision_checker=checker),
                    engine_kwargs={'collision_checker': checker})
        frame = dict(kind='expert', pose=(0., 0., 0.), bev=np.zeros((5, 2, 2), np.float32),
            goal=np.zeros(3, np.float32), state=np.zeros(2, np.float32),
            points=np.zeros((320, 3), np.float32), length=10,
            mask=np.r_[np.ones(10, np.float32), np.zeros(310, np.float32)])
        parent = SimpleNamespace(stop_threshold=.26, forward_with_stop=lambda *args:
            SimpleNamespace(stop_logits=torch.zeros(1, 320)))
        actual = np.zeros((20, 5)); actual[0, 0] = 2
        sample, audit = validated_query(frame, [frame], case, 10, 'C240', parent,
            SimpleNamespace(check=lambda: None), np.zeros((30, 5)), actual)
        self.assertEqual(sample['role'], 'recovery'); self.assertEqual(sample['frame_index'], 1)
        self.assertEqual(audit['start_step'], 10); self.assertEqual(audit['execution_margin_violations'], 2)
        self.assertEqual(audit['reference_margin_violations'], 0); self.assertFalse(audit['clean'])
        with self.assertRaises(ValueError):
            validated_query(frame, [frame], case, 0, 'C240', parent,
                SimpleNamespace(check=lambda: None), np.zeros((30, 5)), actual)

    def test_refinement_requires_safe_connection_and_precheck_never_plans(self):
        from scripts.preflight_failure_teacher_refinement import safe_connected_reference, precheck, run
        checker = SimpleNamespace(pose_free=lambda *args: True,
            swept_segment_free=lambda a, b: a[0] >= 0 and b[0] >= 0)
        reference = Trajectory(np.array([[0., 0., 0.], [1., 0., 0.]]), .2)
        self.assertFalse(safe_connected_reference(reference, checker, VehicleState(-1, 0, 0)))
        self.assertTrue(safe_connected_reference(reference, checker, VehicleState(0, 0, 0)))
        with patch('scripts.preflight_failure_teacher_refinement.prepare_config', return_value={}), \
             patch('scripts.preflight_failure_teacher_refinement.collect') as collect:
            precheck(); collect.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_failure_teacher_refinement.prepare_config') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_budget_exhaustion_is_incomplete_and_never_retried(self):
        import json
        from scripts.preflight_learner_failure_teacher import run
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_learner_failure_teacher.prepare_config', return_value={}), \
                 patch('scripts.preflight_learner_failure_teacher.collect', side_effect=TimeoutError('limit')) as collect:
                with self.assertRaises(TimeoutError): run(Path(folder) / 'new')
                collect.assert_called_once()
            report = json.loads((Path(folder) / 'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'budget_exhausted')
            self.assertFalse(report['training_performed'])


if __name__ == '__main__': unittest.main()
