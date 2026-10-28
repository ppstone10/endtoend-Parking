import unittest
from copy import deepcopy

from experiments.effective_recovery_data import input_id, label_id
from experiments.retention_teacher_roles import source_role, fallback_pairs, build_role_schedules
from experiments.retention_sampling import TRAINING_SEEDS
from tests.test_retention_teacher import paired_fixture
from tests.test_retention_quality import quality_fixture
from tests.test_retention_sampling import sample


class TeacherRoleTests(unittest.TestCase):
    def test_success_preserve_failed_restore_and_invalid_teacher_refused(self):
        self.assertEqual(source_role({'success': True, 'query_gate': {'passed': True}}, {'success': True}), 'C240_preserve')
        self.assertEqual(source_role({'success': False}, {'success': True}), 'R0_restore')
        with self.assertRaises(ValueError): source_role({'success': False}, {'success': False})
        with self.assertRaises(ValueError): source_role({'success': True, 'query_gate': {'passed': False}}, {'success': True})

    def test_fallback_uses_three_clean_actual_R0_inputs_without_target_changes(self):
        samples = [sample(0, i) for i in range(6)]
        rows = [dict(frame_index=s['frame_index'], clean=True, input_sha256=input_id(s), label_sha256=label_id(s)) for s in samples]
        pairs = fallback_pairs(samples, rows)
        self.assertEqual([p['r0']['frame_index'] for p in pairs], [1, 3, 5])
        self.assertTrue(all(label_id(p['r0']) == label_id(p['c240']) for p in pairs))
        with self.assertRaises(ValueError): fallback_pairs(samples[:2], rows[:2])

    def test_mixed_teacher_only_changes_verified_source_targets(self):
        base, _, _ = quality_fixture(TRAINING_SEEDS[0])
        for b in base:
            for j in (0, 1):
                if b[j]['source_index'] in range(4) and b[j]['frame_index'] == 0:
                    b[j] = sample(10+b[j]['source_index'], 0)
        pairs = paired_fixture(); roles = {i: 'C240_preserve' if i >= 2 else 'R0_restore' for i in range(4)}
        for i in (0, 1):
            for p in pairs[i]: p['c240'] = dict(p['r0'])
        plans, audit = build_role_schedules(base, pairs, roles, key_sources=(0, 1, 2, 3))
        self.assertEqual(audit['changed_target_slots'], 6)
        for a, b in zip(plans['R0_targets'], plans['verified_targets']):
            self.assertEqual([input_id(x) for x in a], [input_id(x) for x in b])
        bad = deepcopy(pairs); bad[0][0]['c240'] = sample(0, 40)
        with self.assertRaises(ValueError): build_role_schedules(base, bad, roles, key_sources=(0, 1, 2, 3))


if __name__ == '__main__': unittest.main()
