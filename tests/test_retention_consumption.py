from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.effective_recovery_data import input_id
from experiments.retention_sampling import build_schedule, TRAINING_SEEDS
from experiments.retention_consumption import (
    build_consumption_schedule, consumption_audit, consumption_qualification,
)
from tests.test_retention_sampling import sample, meta, row


def fixture():
    metadata = [meta(i, scene=scene) for i, scene in enumerate((3, 6, 4, 6, 1, 2, 5))]
    old = [sample(i, f) for i in range(7) for f in (0, 3, 6, 9)]
    old += [sample(6, f + 50, 'recovery') for f in range(8)]
    novel = [sample(i, f) for i in range(4) for f in (1, 2, 4, 5, 7, 8)]
    return metadata, old, novel


class RetentionConsumptionTests(unittest.TestCase):
    def test_old_hierarchy_does_not_satisfy_new_consumption_contract(self):
        metadata, old, novel = fixture()
        schedule = build_schedule(old + novel, metadata, seed=TRAINING_SEEDS[0], steps=24,
                                  mode='hierarchy', recovery_reference=old)
        with self.assertRaises(ValueError):
            consumption_audit(schedule, schedule, old, novel, key_sources=(0, 1, 2, 3))

    def test_every_key_source_uses_three_distinct_novel_time_strata(self):
        metadata, old, novel = fixture()
        before = deepcopy(old + novel)
        for seed in TRAINING_SEEDS:
            schedule, audit = build_consumption_schedule(old, novel, metadata, seed=seed,
                                                         key_sources=(0, 1, 2, 3))
            self.assertTrue(audit['passed'])
            self.assertEqual(len(schedule), 24)
            for source in audit['required_sources']:
                self.assertEqual(source['frame_indices'], [2, 5, 8])
                self.assertEqual(source['unique_inputs'], 3)
            self.assertGreaterEqual(audit['novel_unique_inputs'], 12)
        for original, copied in zip(old + novel, before):
            self.assertTrue(torch.equal(original['bev'], copied['bev']))

    def test_scene_and_recovery_slots_are_unchanged_and_deterministic(self):
        metadata, old, novel = fixture()
        for seed in TRAINING_SEEDS:
            baseline = build_schedule(old + novel, metadata, seed=seed, steps=24,
                                      mode='hierarchy', recovery_reference=old)
            schedule, audit = build_consumption_schedule(old, novel, metadata, seed=seed,
                                                         key_sources=(0, 1, 2, 3))
            again, second = build_consumption_schedule(old, novel, metadata, seed=seed,
                                                       key_sources=(0, 1, 2, 3))
            self.assertEqual(audit, second)
            self.assertEqual([[input_id(s) for s in b] for b in schedule],
                             [[input_id(s) for s in b] for b in again])
            for a, b in zip(baseline, schedule):
                self.assertEqual([input_id(s) for s in a[2:]], [input_id(s) for s in b[2:]])
                self.assertEqual([metadata[s['source_index']]['scene_name'] for s in a[:2]],
                                 [metadata[s['source_index']]['scene_name'] for s in b[:2]])

    def test_missing_novel_source_duplicate_and_recovery_drift_fail_closed(self):
        metadata, old, novel = fixture()
        for bad in [novel[:-5], novel + [novel[0]], novel + [old[0]]]:
            with self.assertRaises(ValueError):
                build_consumption_schedule(old, bad, metadata, seed=TRAINING_SEEDS[0],
                                           key_sources=(0, 1, 2, 3))
        schedule, _ = build_consumption_schedule(old, novel, metadata, seed=TRAINING_SEEDS[0],
                                                 key_sources=(0, 1, 2, 3))
        baseline = build_schedule(old + novel, metadata, seed=TRAINING_SEEDS[0], steps=24,
                                  mode='hierarchy', recovery_reference=old)
        schedule[0][2] = sample(0, 99, 'recovery')
        with self.assertRaises(ValueError):
            consumption_audit(schedule, baseline, old, novel, key_sources=(0, 1, 2, 3))

    def test_wrong_roles_or_budget_cannot_pass_audit(self):
        metadata, old, novel = fixture()
        baseline = build_schedule(old + novel, metadata, seed=TRAINING_SEEDS[0], steps=24,
                                  mode='hierarchy', recovery_reference=old)
        schedule, _ = build_consumption_schedule(old, novel, metadata, seed=TRAINING_SEEDS[0],
                                                 key_sources=(0, 1, 2, 3))
        with self.assertRaises(ValueError):
            consumption_audit(schedule[:-1], baseline, old, novel, key_sources=(0, 1, 2, 3))
        schedule[0][0] = schedule[0][2]
        with self.assertRaises(ValueError):
            consumption_audit(schedule, baseline, old, novel, key_sources=(0, 1, 2, 3))

    def test_closed_loop_gate_requires_original_and_incremental_double_seed_gain(self):
        r0 = [row(1, True), row(2, False, True, 'S8_scene')]
        parent = [row(1, False, True), row(2, True, False, 'S8_scene')]
        better = [row(1, True), row(2, True, False, 'S8_scene')]
        parents = {seed: parent for seed in TRAINING_SEEDS}
        candidates = {seed: better for seed in TRAINING_SEEDS}
        result = consumption_qualification(r0, parents, parents, parents, candidates)
        self.assertTrue(result['passed'])
        self.assertFalse(consumption_qualification(r0, parents, parents, candidates, candidates)['passed'])
        candidates[TRAINING_SEEDS[1]] = parent
        self.assertFalse(consumption_qualification(r0, parents, parents, parents, candidates)['passed'])

    def test_precheck_never_trains_or_runs_rollout(self):
        from scripts.preflight_retention_consumption import precheck
        with patch('scripts.preflight_retention_consumption.prepare_context', return_value={'ready': True}), \
             patch('scripts.preflight_retention_consumption.fit_continuation') as fit, \
             patch('scripts.preflight_retention_consumption.evaluate_cases') as evaluate:
            self.assertEqual(precheck(), {'ready': True})
            fit.assert_not_called(); evaluate.assert_not_called()

    def test_existing_output_is_never_overwritten(self):
        from scripts.preflight_retention_consumption import run
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch('scripts.preflight_retention_consumption.prepare_context') as prepare:
                with self.assertRaises(FileExistsError):
                    run(output)
                prepare.assert_not_called()

    def test_identity_failure_stops_before_cache_load(self):
        from scripts.preflight_retention_consumption import prepare_context
        from experiments.executable_behavior import PreflightBudget
        with patch('scripts.preflight_retention_consumption.verify_files', side_effect=ValueError('drift')), \
             patch('scripts.preflight_retention_consumption.torch.load') as load:
            with self.assertRaises(ValueError):
                prepare_context(PreflightBudget(180))
            load.assert_not_called()

    def test_failed_consumption_gate_records_failure_without_training(self):
        import json
        from scripts.preflight_retention_consumption import run
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            with patch('scripts.preflight_retention_consumption.prepare_context', side_effect=ValueError('gate')), \
                 patch('scripts.preflight_retention_consumption.fit_continuation') as fit:
                with self.assertRaises(ValueError):
                    run(output)
                fit.assert_not_called()
                result = json.loads((output / 'report.json').read_text(encoding='utf-8'))
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['allowed_to_prepare_training_plan'])


if __name__ == '__main__':
    unittest.main()
