from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_sampling import TRAINING_SEEDS
from experiments.retention_teacher import select_teacher_pairs, build_teacher_schedules, teacher_qualification
from tests.test_retention_sampling import sample, row
from tests.test_retention_quality import quality_fixture


class R0(torch.nn.Module):
    stop_threshold = .26
    def forward_with_stop(self, bev, goal, state):
        return SimpleNamespace(points=torch.zeros(len(state), 1, 3), stop_logits=torch.ones(len(state), 1) * 3)


def paired_fixture():
    pairs = {}
    for i in range(4):
        pairs[i] = []
        for f in (40, 41, 42):
            c = sample(i, f); r = dict(c, points=torch.zeros_like(c['points']))
            pairs[i].append(dict(c240=c, r0=r))
    return pairs


class RetentionTeacherTests(unittest.TestCase):
    def test_common_queries_are_time_stratified_unique_and_only_targets_differ(self):
        samples = [sample(0, i) for i in range(1, 7)]
        quality = [dict(frame_index=s['frame_index'], clean=True, input_sha256=input_id(s),
                        label_sha256=label_id(s)) for s in samples]
        pairs, audit = select_teacher_pairs(samples, quality, R0())
        self.assertTrue(audit['passed']); self.assertEqual(audit['selected_frames'], [2, 4, 6])
        for pair in pairs:
            self.assertEqual(input_id(pair['r0']), input_id(pair['c240']))
            self.assertTrue(torch.equal(pair['r0']['mask'], pair['c240']['mask']))
            self.assertNotEqual(label_id(pair['r0']), label_id(pair['c240']))

    def test_unsafe_insufficient_or_unsupported_reference_cannot_supply_labels(self):
        samples = [sample(0, i) for i in range(3)]
        quality = [dict(frame_index=s['frame_index'], clean=False, input_sha256=input_id(s),
                        label_sha256=label_id(s)) for s in samples]
        self.assertFalse(select_teacher_pairs(samples, quality, R0())[1]['passed'])
        for s, q in zip(samples, quality):
            q['clean'] = True; s['points'] = torch.ones(3, 3); s['mask'] = torch.tensor([0., 0., 1.])
            q['label_sha256'] = label_id(s)
        pairs, audit = select_teacher_pairs(samples, quality, R0())
        self.assertFalse(audit['passed']); self.assertEqual(pairs, [])
        self.assertEqual(audit['unsupported_R0'], 3)

    def test_audit_identity_drift_fails_before_teacher_prediction(self):
        s = sample(); q = dict(frame_index=0, clean=True, input_sha256='drift', label_sha256=label_id(s))
        with self.assertRaises(ValueError):
            select_teacher_pairs([s], [q], R0())

    def test_same_queries_masks_recovery_and_other_anchors_are_preserved(self):
        for seed in TRAINING_SEEDS:
            base, _, _ = quality_fixture(seed)
            for batch in base:
                for slot in (0, 1):
                    if batch[slot]['source_index'] in range(4) and batch[slot]['frame_index'] == 0:
                        batch[slot] = sample(10 + batch[slot]['source_index'], 0)
            plans, audit = build_teacher_schedules(base, paired_fixture(), key_sources=(0, 1, 2, 3))
            self.assertTrue(audit['passed']); self.assertEqual(audit['query_slots'], 12)
            self.assertEqual(audit['common_unique_inputs'], 12)
            for original, r, c in zip(base, plans['R0_targets'], plans['C240_targets']):
                for a, b, d in zip(original, r, c):
                    self.assertEqual(input_id(b), input_id(d)); self.assertTrue(torch.equal(b['mask'], d['mask']))
                    if a['role'] == 'recovery' or a['source_index'] not in range(4):
                        self.assertEqual(input_id(a), input_id(b)); self.assertEqual(label_id(a), label_id(b))

    def test_wrong_input_or_mask_and_missing_source_cannot_make_paired_training(self):
        base, _, _ = quality_fixture(TRAINING_SEEDS[0])
        for kind in ('input', 'mask', 'missing'):
            pairs = paired_fixture()
            if kind == 'input': pairs[0][0]['r0']['goal'] = torch.ones(3)
            if kind == 'mask': pairs[0][0]['r0']['mask'] = torch.zeros(1)
            if kind == 'missing': del pairs[0]
            with self.assertRaises(ValueError):
                build_teacher_schedules(base, pairs, key_sources=(0, 1, 2, 3))

    def test_joint_gate_needs_incremental_teacher_gain_and_all_source_successes(self):
        r0 = [row(1, True), row(2, False, True, 'S8_scene')]
        parent = [row(1, False, True), row(2, True, False, 'S8_scene')]
        good = [row(1, True), row(2, True, False, 'S8_scene')]
        parents = {s: parent for s in TRAINING_SEEDS}; better = {s: good for s in TRAINING_SEEDS}
        fit = {s: [row(i, True) for i in range(4)] for s in TRAINING_SEEDS}
        self.assertTrue(teacher_qualification(r0, parents, parents, parents, parents, better, fit,
                                             key_sources=(0, 1, 2, 3))['passed'])
        self.assertFalse(teacher_qualification(r0, parents, parents, parents, better, better, fit,
                                              key_sources=(0, 1, 2, 3))['passed'])
        fit[TRAINING_SEEDS[0]][0]['success'] = False
        self.assertFalse(teacher_qualification(r0, parents, parents, parents, parents, better, fit,
                                              key_sources=(0, 1, 2, 3))['passed'])

    def test_precheck_never_collects_and_run_refuses_existing_output(self):
        from scripts.preflight_retention_teacher import precheck, run
        with patch('scripts.preflight_retention_teacher.prepare_context', return_value={}), \
             patch('scripts.preflight_retention_teacher.collect_sources') as collect:
            precheck(); collect.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_retention_teacher.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_failed_source_gate_does_not_train(self):
        import json
        from scripts.preflight_retention_teacher import run
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            context = dict(config={'code': {}, 'frozen_files': {}}, source_gate_passed=False)
            with patch('scripts.preflight_retention_teacher.prepare_context', return_value=context), \
                 patch('scripts.preflight_retention_teacher.collect_sources', return_value=False), \
                 patch('scripts.preflight_retention_teacher.fit_point') as fit:
                run(output); fit.assert_not_called()
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'completed_teacher_gate_failed')
            self.assertFalse(report['training_started'])


if __name__ == '__main__': unittest.main()
