from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.behavior_exploration import source_balanced_batch, source_balanced_pools, TRAINING_SEEDS
from experiments.retention_sampling import (build_schedule, fit_continuation, qualification,
                                            sampling_audit, select_groups, spaced_indices,
                                            regression_candidates, choose_witnesses)
from scripts.build_selection_dataset import collect_metadata_ids


def sample(index=0, frame=0, role='anchor'):
    return {'bev': torch.full((1,), float(index * 100 + frame)), 'goal': torch.zeros(3),
            'state': torch.ones(2), 'points': torch.ones(1, 3), 'stop': torch.ones(1),
            'mask': torch.ones(1), 'role': role, 'source_index': index, 'frame_index': frame}


def meta(index, scene=1, task='T1', maneuver='forward'):
    return {'task_id': f'id-{index}', 'scene_name': f'S{scene}_scene', 'task_type': task,
            'difficulty': {'maneuver': maneuver, 'distance_tier': 'near'}}


class Tiny(torch.nn.Module):
    stop_threshold = .26
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(2, 4)
    def forward_with_stop(self, bev, goal, state):
        values = self.linear(state)
        return SimpleNamespace(points=values[:, :3, None].transpose(1, 2), stop_logits=values[:, 3:])


def row(i, success, collision=False, scene='S4_scene'):
    return {'dataset_index': i, 'task_id': f'task{i}', 'scene_name': scene,
            'success': success, 'collision': collision}


class RetentionSamplingTests(unittest.TestCase):
    def test_flat_continues_exact_original_batches(self):
        samples = [sample(i, j, role) for role in ('anchor', 'recovery') for i in (0, 1) for j in range(9)]
        for seed in TRAINING_SEEDS:
            schedule = build_schedule(samples, [meta(0), meta(1)], seed=seed, steps=24, mode='flat', recovery_reference=samples)
            pools = source_balanced_pools(samples, seed)
            for j, batch in enumerate(schedule):
                self.assertEqual([id(s) for s in batch], [id(s) for s in source_balanced_batch(pools, 240 + j)])

    def test_hierarchy_balances_scenes_and_spans_time_without_changing_recovery(self):
        metadata = [meta(i, scene=1 if i < 10 else 6) for i in range(11)]
        old = [sample(i, j) for i in range(11) for j in range(35)] + [sample(0, j, 'recovery') for j in range(5)]
        for seed in TRAINING_SEEDS:
            flat = build_schedule(old, metadata, seed=seed, steps=24, mode='flat', recovery_reference=old)
            balanced = build_schedule(old, metadata, seed=seed, steps=24, mode='hierarchy', recovery_reference=old)
            for a, b in zip(flat, balanced):
                self.assertEqual([id(s) for s in a[2:]], [id(s) for s in b[2:]])
            audit = sampling_audit(balanced, metadata)
            self.assertEqual(audit['role_counts'], {'anchor': 48, 'recovery': 48})
            self.assertEqual(audit['anchor_scene_counts'], {'S1_scene': 24, 'S6_scene': 24})
            selected = [s['frame_index'] for batch in balanced for s in batch[:2] if s['source_index'] == 10]
            self.assertEqual(min(selected), 0); self.assertEqual(max(selected), 34)
            self.assertEqual(len(set(selected)), 24)

    def test_spaced_indices_midpoint_single_and_full_coverage(self):
        self.assertEqual(spaced_indices(35, 1).tolist(), [17])
        self.assertEqual(spaced_indices(35, 4).tolist(), [0, 11, 22, 34])
        self.assertEqual(set(spaced_indices(5, 12)), set(range(5)))
        with self.assertRaises(ValueError): spaced_indices(0, 1)

    def test_augmenting_anchor_pool_cannot_change_recovery_rng_sequence(self):
        metadata = [meta(i, scene=i + 1) for i in range(3)]
        old = [sample(i, j, role) for role in ('anchor', 'recovery') for i in (0, 1) for j in range(5)]
        augmented = old + [sample(2, j) for j in range(9)]
        for seed in TRAINING_SEEDS:
            baseline = build_schedule(old, metadata, seed=seed, steps=24, mode='flat', recovery_reference=old)
            for mode in ('flat', 'hierarchy'):
                changed = build_schedule(augmented, metadata, seed=seed, steps=24, mode=mode, recovery_reference=old)
                self.assertEqual([[id(s) for s in b[2:]] for b in baseline], [[id(s) for s in b[2:]] for b in changed])

    def test_groups_exclude_prior_and_probe_source_identities(self):
        metadata = []
        for scene, task, maneuver in [(3, 'T4', 'reverse'), (4, 'T2', 'forward'),
                                      (6, 'T3', 'reverse'), (6, 'T5', 'forward'), (8, 'T5', 'reverse')]:
            for _ in range(16):
                index = len(metadata); metadata.append(meta(index, scene, task, maneuver))
        metadata[1]['source_task_id'] = metadata[0]['task_id']
        forbidden = collect_metadata_ids(metadata[0])
        groups = select_groups(metadata, forbidden)
        self.assertEqual(groups, select_groups(metadata, forbidden))
        self.assertEqual(len(groups['probes']), 6); self.assertEqual(len(groups['gap_candidates']), 8)
        a = collect_metadata_ids([metadata[i] for i in groups['probes']])
        b = collect_metadata_ids([metadata[i] for i in groups['gap_candidates']])
        self.assertFalse(a & b or a & forbidden or b & forbidden)
        expanded = select_groups(metadata, forbidden, gap_counts=(4, 12))
        self.assertEqual(expanded['probes'], groups['probes'])
        self.assertEqual(expanded['gap_candidates'][:8], groups['gap_candidates'])
        self.assertEqual(len(expanded['gap_candidates']), 16)

    def test_continuation_copies_parent_and_adam(self):
        parent = Tiny(); opt = torch.optim.Adam(parent.parameters(), lr=1e-5)
        parent.linear(torch.ones(1, 2)).sum().backward(); opt.step()
        for state in opt.state.values(): state['step'].fill_(240)
        before = deepcopy(parent.state_dict()); saved_opt = deepcopy(opt.state_dict())
        batch = [sample(0), sample(0), sample(0, role='recovery'), sample(0, role='recovery')]
        model, optimizer, history = fit_continuation(parent, saved_opt, [batch] * 24, seed=TRAINING_SEEDS[0], check_budget=lambda: None)
        for key, value in before.items(): torch.testing.assert_close(value, parent.state_dict()[key], atol=0, rtol=0)
        self.assertTrue(all(int(x['step']) == 264 for x in optimizer.state.values()))
        self.assertTrue(all(int(x['step']) == 240 for x in saved_opt['state'].values()))
        self.assertEqual(len(history), 24)

    def test_gate_rejects_fit_only_one_seed_s8_loss_and_dose_only(self):
        r0 = [row(1, True), row(2, False, True, 'S8_scene')]
        parent = [row(1, False, True), row(2, True, False, 'S8_scene')]
        better = [row(1, True), row(2, True, False, 'S8_scene')]
        baseline = {seed: parent for seed in TRAINING_SEEDS}
        arms = {'flat_old': baseline, 'balanced_augmented': {seed: better for seed in TRAINING_SEEDS}}
        self.assertEqual(qualification(r0, baseline, arms)['qualified'], ['balanced_augmented'])
        arms['balanced_augmented'][TRAINING_SEEDS[1]] = parent
        self.assertEqual(qualification(r0, baseline, arms)['qualified'], [])
        arms['balanced_augmented'] = {seed: [row(1, True), row(2, False, True, 'S8_scene')] for seed in TRAINING_SEEDS}
        self.assertEqual(qualification(r0, baseline, arms)['qualified'], [])
        arms['flat_old'] = {seed: better for seed in TRAINING_SEEDS}
        arms['balanced_augmented'] = arms['flat_old']
        self.assertEqual(qualification(r0, baseline, arms)['qualified'], [])

    def test_witness_selection_uses_baselines_not_new_model_outcomes(self):
        r0 = [row(1, True), row(2, True), row(3, False, True, 'S8_scene')]
        parents = {TRAINING_SEEDS[0]: [row(1, True), row(2, False, True), row(3, True, False, 'S8_scene')],
                   TRAINING_SEEDS[1]: [row(1, False, True), row(2, True), row(3, True, False, 'S8_scene')]}
        result = choose_witnesses(r0, parents, [1, 2])
        self.assertTrue(result['passed']); self.assertEqual(result['selected_indices'], [2, 1])
        parents[TRAINING_SEEDS[0]][1]['success'] = True
        self.assertFalse(choose_witnesses(r0, parents, [1, 2])['passed'])

    def test_regression_candidates_are_finite_disjoint_and_deterministic(self):
        metadata = []
        for scene, task, maneuver in [(3, 'T4', 'reverse'), (4, 'T2', 'forward'),
                                      (6, 'T3', 'reverse'), (6, 'T5', 'forward')]:
            for _ in range(10):
                metadata.append(meta(len(metadata), scene, task, maneuver))
        forbidden = collect_metadata_ids(metadata[0])
        candidates = regression_candidates(metadata, forbidden)
        self.assertEqual(len(candidates), 16); self.assertNotIn(0, candidates)
        self.assertEqual(candidates, regression_candidates(metadata, forbidden))

    def test_prepare_precheck_never_rollout_and_budget_modes_cannot_mix(self):
        from scripts.preflight_retention_sampling import (prepare_diagnostic, verify_prepared,
                                                        execution_budget, run, FrozenIdentityError)
        from experiments.executable_behavior import PreflightBudget
        self.assertIsInstance(execution_budget(False), PreflightBudget)
        self.assertEqual(execution_budget(True).limit, 900)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            with patch('scripts.preflight_retention_sampling.make_config', return_value={'version': 1}), \
                 patch('scripts.preflight_retention_sampling.run_case') as rollout:
                prepare_diagnostic(output); verify_prepared(output); rollout.assert_not_called()
            with patch('scripts.preflight_retention_sampling.make_config', return_value={'version': 2}):
                with self.assertRaises(FrozenIdentityError): verify_prepared(output)
            with self.assertRaises(ValueError): run(output, complete_diagnostic=True)

    def test_locked_prepared_run_cannot_overwrite_status(self):
        from scripts.preflight_retention_sampling import run
        from experiments.exploration_store import exploration_lock
        from training.reporting import atomic_write_json
        import json
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder); atomic_write_json(output / 'report.json', {'status': 'prepared'})
            before = (output / 'report.json').read_bytes()
            with patch('scripts.preflight_retention_sampling.verify_prepared', return_value={}):
                with exploration_lock(output):
                    with self.assertRaises(FileExistsError): run(output, complete_diagnostic=True, prepared=True)
            self.assertEqual((output / 'report.json').read_bytes(), before)


if __name__ == '__main__': unittest.main()
