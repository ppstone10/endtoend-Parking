from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.effective_recovery_data import input_id, label_id
from experiments.s6_recovery_learning import build_recovery_schedules, batch_identity, recovery_qualification


def sample(value, role='recovery', step=10):
    return dict(bev=torch.full((5, 2, 2), float(value)), goal=torch.zeros(3), state=torch.zeros(2),
        points=torch.zeros(320, 3), stop=(torch.arange(320) >= 19).float(),
        mask=(torch.arange(320) < 10).float(), role=role, source_index=2149, frame_index=step//10,
        query_step=step, query_policy='C240-20261020', origin='learner_query')


class SelfTeacher:
    stop_threshold = .26

    def forward_with_stop(self, *args):
        return SimpleNamespace(points=torch.full((1, 320, 3), .5), stop_logits=torch.full((1, 320), -9.))


def fixture():
    base = [[sample(-1-j*4-k, 'anchor' if k < 2 else 'recovery') for k in range(4)] for j in range(24)]
    queries = [sample(s, step=s) for s in (400, 10, 200)]
    return base, queries


class S6RecoveryTests(unittest.TestCase):
    def test_only_three_recovery_targets_differ_on_shared_inputs(self):
        base, queries = fixture(); before = deepcopy(base)
        plans, audit = build_recovery_schedules(base, queries, SelfTeacher())
        self.assertEqual(audit['changed_slots'], [[0, 2], [8, 2], [16, 2]])
        self.assertEqual(audit['query_steps'], [10, 200, 400])
        for j, (original, control, strict) in enumerate(zip(before, plans['self_targets'], plans['strict_targets'])):
            for k, (a, b, c) in enumerate(zip(original, control, strict)):
                self.assertEqual(input_id(b), input_id(c)); self.assertTrue(torch.equal(b['mask'], c['mask']))
                if [j, k] not in audit['changed_slots']:
                    self.assertEqual(batch_identity([a]), batch_identity([b]))
                    self.assertEqual(label_id(a), label_id(c))
                else:
                    self.assertNotEqual(label_id(b), label_id(c))
        self.assertEqual(batch_identity(base[0]), batch_identity(before[0]))

    def test_bad_origin_duplicate_inputs_wrong_slot_or_short_stop_fail(self):
        base, queries = fixture()
        variants = [queries[:-1], [queries[0]]*3, [dict(q, origin='teacher_continuation') for q in queries]]
        for bad in variants:
            with self.assertRaises(ValueError): build_recovery_schedules(base, bad, SelfTeacher())
        base[0][2]['role'] = 'anchor'
        with self.assertRaises(ValueError): build_recovery_schedules(base, queries, SelfTeacher())
        base, queries = fixture(); teacher = SelfTeacher()
        teacher.forward_with_stop = lambda *args: SimpleNamespace(points=torch.zeros(1, 320, 3),
                                                                   stop_logits=torch.ones(1, 320))
        with self.assertRaises(ValueError): build_recovery_schedules(base, queries, teacher)

    def test_batch_identity_detects_target_changes_not_just_input(self):
        base, _ = fixture(); changed = deepcopy(base[0]); changed[2]['points'][0, 0] = 1
        self.assertEqual(input_id(base[0][2]), input_id(changed[2]))
        self.assertNotEqual(batch_identity(base[0]), batch_identity(changed))

    def test_precheck_no_fit_or_rollout_existing_output_not_overwritten(self):
        from scripts.preflight_s6_recovery_learning import precheck, run
        with patch('scripts.preflight_s6_recovery_learning.prepare_context', return_value={}), \
             patch('scripts.preflight_s6_recovery_learning.fit_recovery_point') as fit:
            precheck(); fit.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_recovery_learning.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_closed_loop_gate_needs_net_gain_strict_s6_and_no_lost_successes(self):
        seeds = (20261020, 20261021)

        def row(index, success, scene='S6_scene'):
            return dict(dataset_index=index, task_id=f'task-{index}', scene_name=scene,
                        success=success, collision=not success, reached_with_margin=success)

        control = [row(1, True, 'S4_scene'), row(2, True, 'S8_scene'), row(3, False)]
        better = [row(1, True, 'S4_scene'), row(2, True, 'S8_scene'), row(3, True)]
        parents = {s: control for s in seeds}; candidates = {s: better for s in seeds}
        baseline_fit = {s:[row(2149, s == seeds[1]), row(2256, True)] for s in seeds}
        good_fit = {s:[row(2149, True), row(2256, True)] for s in seeds}
        with patch('experiments.s6_recovery_learning.consumption_qualification', return_value={'passed': True}):
            call = lambda models, fit: recovery_qualification(control, parents, parents, parents,
                                                              parents, models, baseline_fit, fit)
            self.assertTrue(call(candidates, good_fit)['passed'])
            self.assertFalse(call(parents, good_fit)['passed'])
            bad_fit = deepcopy(good_fit); bad_fit[seeds[0]][0]['reached_with_margin'] = False
            self.assertFalse(call(candidates, bad_fit)['passed'])
            bad = deepcopy(candidates); bad[seeds[0]][0]['success'] = False
            self.assertFalse(call(bad, good_fit)['passed'])
            self.assertFalse(call({seeds[0]: better}, good_fit)['passed'])
        with patch('experiments.s6_recovery_learning.consumption_qualification', return_value={'passed': False}):
            self.assertFalse(recovery_qualification(control, parents, parents, parents, parents,
                                                    candidates, baseline_fit, good_fit)['passed'])

    def test_gradient_callback_rejects_late_target_drift(self):
        from scripts.preflight_s6_recovery_learning import fit_recovery_point
        base, queries = fixture(); plans, _ = build_recovery_schedules(base, queries, SelfTeacher())
        schedule = plans['strict_targets']; seed = 20261020
        context = dict(parent=None, plans={'strict_targets': {seed: schedule}},
            config=dict(parent_checkpoints={str(seed): 'not_loaded'},
                        planned_batch_identities={'strict_targets': {seed: [batch_identity(b) for b in schedule]}}))

        def mutate_before_gradient(model, optimizer, actual, **kwargs):
            actual[0][2]['points'] = actual[0][2]['points'] + 1
            kwargs['check_budget']()
            self.fail('漂移目标不可进入梯度')

        with patch('scripts.preflight_s6_recovery_learning.load_candidate', return_value=(None, None)), \
             patch('scripts.preflight_s6_recovery_learning.fit_continuation', side_effect=mutate_before_gradient):
            with self.assertRaises(ValueError):
                fit_recovery_point(context, 'strict_targets', seed, SimpleNamespace(check=lambda: None), Path('unused'))

    def test_budget_exhaustion_does_not_fit_or_retry(self):
        import json
        from scripts.preflight_s6_recovery_learning import run
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_recovery_learning.prepare_context', side_effect=TimeoutError('limit')) as prepare, \
                 patch('scripts.preflight_s6_recovery_learning.fit_recovery_point') as fit:
                with self.assertRaises(TimeoutError): run(Path(folder)/'new')
                prepare.assert_called_once(); fit.assert_not_called()
            report = json.loads((Path(folder)/'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'budget_exhausted')
            self.assertFalse(report['allowed_to_prepare_training_plan'])


if __name__ == '__main__': unittest.main()
