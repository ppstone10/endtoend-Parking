from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.effective_recovery_data import (ARMS, data_readiness, fingerprint, input_id,
    matched_pools, novel_samples, query_sample, select_sources, uniform_contexts)
from scripts.build_selection_dataset import collect_metadata_ids
from scripts.preflight_behavior_repair import FIT, PROBE


def sample(role='recovery', index=1, value=0.):
    return {'bev': torch.full((5, 2, 2), value), 'goal': torch.zeros(3), 'state': torch.zeros(2),
            'points': torch.zeros(320, 3), 'stop': torch.ones(320), 'mask': torch.ones(320),
            'role': role, 'source_index': index, 'frame_index': 0}


class EffectiveRecoveryTests(unittest.TestCase):
    def test_duplicate_source_identity_and_ulp_are_not_new_supervision(self):
        original = sample(); renamed = dict(original, source_index=99, frame_index=8)
        novel, audit = novel_samples([original], [renamed] * 27)
        self.assertEqual(novel, []); self.assertEqual(audit['novel_inputs_vs_old'], 0)
        self.assertEqual(audit['novel_pairs_vs_old'], 0); self.assertEqual(audit['duplicates'], 27)
        rounded = deepcopy(original); rounded['goal'][0] += 1e-7
        self.assertNotEqual(input_id(original), input_id(rounded))
        self.assertEqual(novel_samples([original], [rounded])[1]['rounding_aliases'], 1)

    def test_new_conflicting_and_batch_duplicate_are_counted_separately(self):
        old = sample(); conflict = deepcopy(old); conflict['points'][0, 0] = 2
        new = sample(value=1.)
        accepted, audit = novel_samples([old], [conflict, new, new])
        self.assertEqual(len(accepted), 1); self.assertEqual(audit['conflicts'], 1)
        self.assertEqual(audit['batch_unique_inputs'], 2); self.assertEqual(audit['novel_inputs_vs_old'], 1)
        self.assertEqual(audit['novel_pairs_vs_old'], 2); self.assertEqual(audit['duplicates'], 1)
        reshaped = dict(old, bev=old['bev'].reshape(-1))
        self.assertNotEqual(input_id(old), input_id(reshaped))
        bad = deepcopy(old); bad['state'][0] = float('nan')
        with self.assertRaises(ValueError): input_id(bad)
        minus_zero = deepcopy(old); minus_zero['state'][0] = -0.
        self.assertEqual(input_id(old), input_id(minus_zero))

    def test_actual_input_first_teacher_frame_only_and_parent_cache_recomputed(self):
        teacher = {'kind': 'expert', 'pose': (1., 2., 3.), 'bev': np.zeros((5, 2, 2), np.float32),
                   'goal': np.zeros(3, np.float32), 'state': np.zeros(2, np.float32),
                   'points': np.zeros((320, 3), np.float32), 'length': 10, 'mask': np.ones(320, np.float32)}
        visited = deepcopy(teacher); visited['goal'][0] = 1e-7
        parent = SimpleNamespace(stop_threshold=.26, forward_with_stop=lambda *args:
                                  SimpleNamespace(stop_logits=torch.zeros(1, 320)))
        budget = SimpleNamespace(check=lambda: None)
        actual = query_sample(visited, [teacher] * 9, 1559, 320, 'C240', parent, budget)
        np.testing.assert_array_equal(actual['goal'], visited['goal'])
        self.assertEqual(actual['frame_index'], 32); self.assertEqual(actual['parent_length'], 1)
        self.assertEqual(actual['origin'], 'learner_query')
        different = deepcopy(teacher); different['bev'][0, 0, 0] = 1
        with self.assertRaises(ValueError): query_sample(different, [teacher], 1, 10, 'R0', parent, budget)

    def test_uniform_nonzero_safe_contexts_not_truncated_at_first_margin(self):
        contexts = [SimpleNamespace(step=i) for i in (0, 10, 20, 30, 40, 50)]
        self.assertEqual([c.step for c in uniform_contexts(contexts)], [10, 30, 50])
        self.assertEqual(uniform_contexts([SimpleNamespace(step=0)]), [])

    def test_splits_predeclared_and_disjoint_by_source_identity(self):
        scenes = (1, 2, 3, 4, 5, 6, 8)
        metadata = [{'task_id': f'task{i}', 'source_task_id': f'group{i//2}',
                     'scene_name': f'S{scenes[i%7]}_scene', 'task_type': f'T{i%5+1}',
                     'difficulty': {'maneuver': 'forward' if i%2 else 'reverse',
                                    'distance_tier': ('near', 'medium', 'long')[i%3]}} for i in range(2400)]
        old_confirmation = list(range(60)); reserved = collect_metadata_ids(metadata[65])
        groups = select_sources(metadata, old_confirmation, reserved)
        self.assertEqual(groups, select_sources(metadata, old_confirmation, reserved))
        self.assertEqual(len(groups['fresh_confirmation']), 60)
        self.assertEqual(len(groups['recovery_sources']), 12)
        used = reserved | collect_metadata_ids([metadata[i] for i in FIT + PROBE + old_confirmation])
        for name in ('fresh_confirmation', 'recovery_sources', 'replay_sources'):
            identities = collect_metadata_ids([metadata[i] for i in groups[name]])
            self.assertFalse(identities & used); used |= identities

    def test_readiness_is_joint_not_frame_count(self):
        metadata = [{'scene_name': f'S{i%6+1}_scene', 'task_type': f'T{i%5+1}',
                     'difficulty': {'maneuver': 'forward' if i%2 else 'reverse',
                                    'distance_tier': ('near', 'medium', 'long')[i%3]}} for i in range(24)]
        metadata += [{'scene_name': f'S{n}_scene'} for n in (4, 4, 8, 8)]
        replay = [sample('anchor', i) for i in range(24)]
        queries = [dict(sample(index=i), origin='learner_query', query_step=10) for i in range(24, 28)]
        self.assertTrue(data_readiness(replay, queries, metadata)['passed'])
        self.assertFalse(data_readiness(replay * 100, queries[:1] * 100, metadata)['passed'])
        self.assertFalse(data_readiness(replay[:23], queries, metadata)['passed'])
        queries[0]['query_step'] = 0
        self.assertFalse(data_readiness(replay, queries, metadata)['passed'])

    def test_factorial_changes_only_declared_pool(self):
        old = [sample('anchor'), sample()]; replay = [sample('anchor', 2)]; new = [sample(index=3)]
        pools = matched_pools(old, replay, new)
        self.assertEqual(set(pools), set(ARMS))
        self.assertEqual([len(pools[k]) for k in ARMS], [2, 3, 3, 4])
        self.assertIs(pools['broad_novel'][0], old[0])

    def test_new_identity_rejects_drift_without_rewriting_old(self):
        from scripts.explore_effective_replay import prepare, verify_identity, IdentityMismatch
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            with patch('scripts.explore_effective_replay.make_config', return_value={'smoke': True, 'version': 1}):
                prepare(output, smoke=True); verify_identity(output)
                with self.assertRaises(FileExistsError): prepare(output, smoke=True)
            with patch('scripts.explore_effective_replay.make_config', return_value={'smoke': True, 'version': 2}):
                with self.assertRaises(IdentityMismatch): verify_identity(output)

    def test_config_update_count_matches_actual_fitter_not_inherited_preflight_twelve(self):
        from scripts.explore_effective_replay import make_config
        groups = {'fresh_confirmation': [], 'recovery_sources': [], 'replay_sources': [], 'replay_cell_count': 0}
        metadata = [{'task_id': f'id{i}'} for i in range(2400)]
        with patch('scripts.explore_effective_replay.previous_identity', return_value=({'confirmation_indices': []}, {})), \
             patch('scripts.explore_effective_replay.frozen_config', return_value={'code': {}, 'optimizer_steps': 12}), \
             patch('scripts.explore_effective_replay.metadata_only', return_value=metadata), \
             patch('scripts.explore_effective_replay.select_sources', return_value=groups):
            config = make_config()
            self.assertEqual(config['steps'], 240); self.assertEqual(config['optimizer_steps'], 240)


if __name__ == '__main__':
    unittest.main()
