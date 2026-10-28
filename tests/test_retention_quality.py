from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_consumption import build_consumption_schedule
from experiments.retention_sampling import TRAINING_SEEDS
from experiments.retention_quality import audit_source_windows, build_quality_schedules, quality_qualification
from interfaces import GoalPose, VehicleState
from tests.test_retention_consumption import fixture
from tests.test_retention_sampling import sample, row


def quality_fixture(seed):
    metadata, old, novel = fixture()
    base, _ = build_consumption_schedule(old, novel, metadata, seed=seed, key_sources=(0, 1, 2, 3))
    # This synthetic fixture has fewer sources than the real bank; normalize the
    # residual same-source visits to old inputs, leaving exactly 12 novel slots.
    for index in range(4):
        positions = [(step, slot) for step, batch in enumerate(base) for slot in (0, 1)
                     if batch[slot]['source_index'] == index]
        for step, slot in positions:
            base[step][slot] = sample(index, 0)
        for (step, slot), frame in zip(positions[:3], (2, 5, 8)):
            base[step][slot] = sample(index, frame)
    novel += [sample(0, f) for f in (10, 11, 12, 13)]
    rows = [dict(source_index=0, frame_index=s['frame_index'], current_safe=True,
                 input_sha256=input_id(s), label_sha256=label_id(s),
                 clean=s['frame_index'] in (1, 2, 4, 7)) for s in novel if s['source_index'] == 0]
    return base, novel, rows


class RetentionQualityTests(unittest.TestCase):
    def test_reference_risk_and_execution_risk_are_not_conflated(self):
        checker = SimpleNamespace(pose_free=lambda x, y, yaw: x <= 1,
                                  swept_segment_free=lambda a, b: a[0] <= 1 and b[0] <= 1)
        contact = SimpleNamespace(swept_segment_free=lambda a, b: True)
        case = dict(start=VehicleState(0, 0, 0), goal=GoalPose(0, 0, 0),
                    planner=SimpleNamespace(_collision_checker=checker),
                    engine_kwargs={'collision_checker': contact})
        s = sample(0, 0); s['state'] = torch.zeros(2); s['points'] = torch.tensor([[[2., 0., 0.]]]).reshape(1, 3)
        rows = audit_source_windows([s], np.zeros((11, 5)), case)
        self.assertEqual(rows[0]['execution_margin_violations'], 0)
        self.assertEqual(rows[0]['reference_margin_violations'], 1)
        s['points'].zero_(); states = np.zeros((11, 5)); states[0, 0] = 2
        rows = audit_source_windows([s], states, case)
        self.assertEqual(rows[0]['reference_margin_violations'], 0)
        self.assertEqual(rows[0]['execution_margin_violations'], 2)
        self.assertFalse(rows[0]['clean'])

    def test_source_input_drift_empty_mask_and_out_of_range_fail(self):
        checker = SimpleNamespace(pose_free=lambda *args: True, swept_segment_free=lambda *args: True)
        case = dict(start=VehicleState(0, 0, 0), goal=GoalPose(0, 0, 0),
                    planner=SimpleNamespace(_collision_checker=checker), engine_kwargs={'collision_checker': checker})
        valid = sample(0, 0); valid['state'] = torch.zeros(2)
        for field, value in [('state', torch.ones(2)), ('goal', torch.ones(3)),
                             ('mask', torch.zeros(1)), ('frame_index', 999)]:
            bad = deepcopy(valid); bad[field] = value
            with self.assertRaises(ValueError):
                audit_source_windows([bad], np.zeros((11, 5)), case)

    def test_two_slots_same_source_and_all_other_samples_stay_identical(self):
        for seed in TRAINING_SEEDS:
            base, novel, rows = quality_fixture(seed)
            plans, audit = build_quality_schedules(base, novel, rows, source_index=0, target_frames=(5, 8))
            self.assertEqual(audit['replacement_frames'], {'clean_windows': [4, 7], 'risky_windows': [10, 11]})
            for arm, schedule in plans.items():
                changed = []
                for step, (a, b) in enumerate(zip(base, schedule)):
                    for slot in range(4):
                        if input_id(a[slot]) != input_id(b[slot]):
                            changed.append((step, slot)); self.assertLess(slot, 2)
                            self.assertEqual(a[slot]['source_index'], b[slot]['source_index'])
                        else:
                            self.assertEqual(label_id(a[slot]), label_id(b[slot]))
                self.assertEqual(len(changed), 2)
                self.assertEqual(audit['arms'][arm]['novel_unique_inputs'], 12)

    def test_selection_ignores_row_order_and_requires_two_eligible_alternatives(self):
        base, novel, rows = quality_fixture(TRAINING_SEEDS[0])
        a = build_quality_schedules(base, novel, rows, source_index=0, target_frames=(5, 8))[1]
        b = build_quality_schedules(base, novel[::-1], rows[::-1], source_index=0, target_frames=(5, 8))[1]
        self.assertEqual(a, b)
        for r in rows:
            if r['frame_index'] not in (2, 5, 8):
                r['current_safe'] = False
        with self.assertRaises(ValueError):
            build_quality_schedules(base, novel, rows, source_index=0, target_frames=(5, 8))

    def test_quality_gate_does_not_accept_ordinary_resampling_or_one_seed(self):
        r0 = [row(1, True), row(2, False, True, 'S8_scene')]
        parent = [row(1, False, True), row(2, True, False, 'S8_scene')]
        better = [row(1, True), row(2, True, False, 'S8_scene')]
        parents = {s: parent for s in TRAINING_SEEDS}; good = {s: better for s in TRAINING_SEEDS}
        self.assertTrue(quality_qualification(r0, parents, parents, parents, good, parents)['passed'])
        self.assertFalse(quality_qualification(r0, parents, parents, parents, good, good)['passed'])
        good[TRAINING_SEEDS[1]] = parent
        self.assertFalse(quality_qualification(r0, parents, parents, parents, good, parents)['passed'])

    def test_audit_identity_drift_and_duplicate_target_slots_are_rejected(self):
        base, novel, rows = quality_fixture(TRAINING_SEEDS[0])
        bad = deepcopy(rows); bad[0]['label_sha256'] = 'drift'
        with self.assertRaises(ValueError):
            build_quality_schedules(base, novel, bad, source_index=0, target_frames=(5, 8))
        base[0][0] = sample(0, 5)
        with self.assertRaises(ValueError):
            build_quality_schedules(base, novel, rows, source_index=0, target_frames=(5, 8))

    def test_precheck_does_not_train_or_evaluate_and_output_never_overwritten(self):
        from scripts.preflight_retention_quality import precheck, run
        with patch('scripts.preflight_retention_quality.prepare_context', return_value={}), \
             patch('scripts.preflight_retention_quality.fit_continuation') as fit, \
             patch('scripts.preflight_retention_quality.evaluate_cases') as evaluate:
            precheck(); fit.assert_not_called(); evaluate.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_retention_quality.prepare_context') as prepare:
                with self.assertRaises(FileExistsError):
                    run(Path(folder))
                prepare.assert_not_called()

    def test_failed_quality_gate_stops_before_gradient(self):
        import json
        from scripts.preflight_retention_quality import run
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            with patch('scripts.preflight_retention_quality.prepare_context', side_effect=ValueError('quality gate')), \
                 patch('scripts.preflight_retention_quality.fit_continuation') as fit:
                with self.assertRaises(ValueError):
                    run(output)
                fit.assert_not_called()
                self.assertFalse(json.loads((output / 'report.json').read_text(encoding='utf-8'))['allowed_to_prepare_training_plan'])


if __name__ == '__main__':
    unittest.main()
