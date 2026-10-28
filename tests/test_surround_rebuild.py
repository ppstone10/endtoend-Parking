"""CAM-PROJ-004：真实任务复原、小数据迁移与旧权重隔离。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataset import DatasetGenerator
from interfaces import BEVConfig, BEVTensor, GoalPose, VehicleState
from interfaces.bev import CAMERA_PROJECTION_CURRENT
from model import build_model
from sim import MINING_DRILL_RIG, TaskSampler, TaskType, Maneuver, NoiseLevel
from training.checkpoint import load_model_checkpoint
from verification.surround_rebuild import (repair_camera_targets, file_sha256, inspect_camera_target_repair,
                                          resolve_rebuild_task)


def write_source(root):
    sampler = TaskSampler(seed=123, vehicle_length=6, vehicle_width=3, collision_margin=MINING_DRILL_RIG.collision_margin)
    task = sampler.sample('S1_parking_lot', TaskType.T1_NEAR, 0,
                          maneuver=Maneuver.REVERSE, adjacent_occupancy=0, noise_level=NoiseLevel.CLEAN)
    goal = task.goal or task.candidate_goals[0]
    metadata = task.to_metadata()
    metadata['dataset'] = {'vehicle_model': MINING_DRILL_RIG.to_metadata(), 'selected_goal': goal.to_metadata()}
    recovery = json.loads(json.dumps(metadata))
    recovery['recovery'] = {'trigger': 'collision_backtrack', 'frame': 1}
    config = BEVConfig()
    bev = BEVTensor(np.zeros((5, *config.shape)), config.resolution, config.extent,
                    ['occupancy', 'height', 'density', 'target', 'vehicle'])
    rng = np.random.default_rng(11)
    bevs = rng.random((2, 5, *config.shape), dtype=np.float32)
    bevs[:, 3] = 0
    states = np.stack([task.start.to_array(), task.start.to_array()])
    states[1, 0] += .1
    source = root/'train.npz'
    np.savez_compressed(source, schema_version=np.asarray(2, np.uint16),
                        bev_meta=np.asarray(json.dumps(bev.to_metadata())),
                        task_meta=np.array([json.dumps(metadata), json.dumps(recovery)]), bevs=bevs,
                        states=states, goals=np.tile([goal.x, goal.y, goal.yaw], (2, 1)),
                        trajs=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
                        masks=np.array([[1, 1, 0], [1, 1, 1]], np.float32), dt=np.array([.2]))
    (root/'manifest.json').write_text(json.dumps({'schema_version': 1, 'seed': 123,
                                                 'vehicle_model': MINING_DRILL_RIG.to_metadata()}))
    return source


class TestSurroundRebuild(unittest.TestCase):
    def relabelled_source(self, root):
        source = write_source(root)
        with np.load(source) as archive:
            data = {key: archive[key] for key in archive.files}
        origin = root/'original.npz'
        sample_fields = ('bevs', 'states', 'goals', 'task_meta', 'trajs', 'masks')
        np.savez_compressed(origin, **{key: value[:1] if key in sample_fields else value for key, value in data.items()})
        parent, recovery = [json.loads(str(m)) for m in data['task_meta']]
        recovery['difficulty']['maneuver'] = 'forward'
        recovery['dataset'].update(source='closed_loop_recovery', goal_policy='original_selected_goal',
                                    maneuver_audit={'requested_maneuver': 'forward', 'consistent': True})
        recovery['recovery'] = {'schema_version': 1, 'source_dataset_index': 0, 'source_task_id': parent['task_id'],
                                'label_policy': 'expert_replan_from_learner_state'}
        data['task_meta'] = np.array([json.dumps(parent), json.dumps(recovery)])
        np.savez_compressed(source, **data)
        return source, origin, parent, recovery

    def test_recovery_relabel_uses_original_identity_without_changing_stored_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, original, parent, recovery = self.relabelled_source(root)
            with self.assertRaisesRegex(ValueError, '任务身份复原漂移'):
                repair_camera_targets(source, root/'without-parent')
            before, parent_before = file_sha256(source), file_sha256(original)
            report = repair_camera_targets(source, root/'fixed', original_data_path=original)
            self.assertEqual(report['rebuilt'], 2)
            self.assertEqual(report['recovery_labels_different'], 1)
            self.assertEqual(report['rows'][1]['lineage']['original_task_maneuver'], 'reverse')
            self.assertEqual(report['rows'][1]['lineage']['recovery_label_maneuver'], 'forward')
            self.assertEqual(before, file_sha256(source)); self.assertEqual(parent_before, file_sha256(original))
            self.assertEqual(inspect_camera_target_repair(root/'fixed')['status'], 'passed')
            with np.load(source) as a, np.load(report['output_data']) as b:
                for key in ('task_meta', 'states', 'goals', 'trajs', 'masks', 'dt'):
                    np.testing.assert_array_equal(a[key], b[key])
            loaded = DatasetGenerator.load(report['output_data'])
            self.assertEqual(loaded['task_meta'][1]['difficulty']['maneuver'], 'forward')

    def test_recovery_lineage_rejects_real_drift_and_forged_parent(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _, parent, recovery = self.relabelled_source(root)
            cases = []
            bad = deepcopy(recovery); bad['start']['yaw'] *= -1; cases.append(bad)
            bad = deepcopy(recovery); bad['goal']['x'] += 1; cases.append(bad)
            bad = deepcopy(recovery); bad['recovery']['source_task_id'] = 'wrong'; cases.append(bad)
            bad = deepcopy(recovery); bad['recovery']['source_dataset_index'] = 99; cases.append(bad)
            bad = deepcopy(recovery); bad['dataset']['maneuver_audit']['requested_maneuver'] = 'reverse'; cases.append(bad)
            bad = deepcopy(recovery); bad['difficulty']['noise_level'] = 'high'; cases.append(bad)
            bad = deepcopy(recovery); bad['dataset']['selected_goal']['yaw'] += .1; cases.append(bad)
            for bad in cases:
                with self.assertRaises(ValueError):
                    resolve_rebuild_task(bad, root_seed=123, vehicle=MINING_DRILL_RIG, original_tasks=[parent])

    def test_rebuild_preserves_duplicate_tasks_recovery_labels_and_other_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_source(root)
            before = file_sha256(source)
            report = repair_camera_targets(source, root/'fixed', budget_seconds=20)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['rebuilt'], 2)
            self.assertFalse(report['partial'])
            self.assertEqual(before, file_sha256(source))
            self.assertEqual(inspect_camera_target_repair(root/'fixed')['status'], 'passed')
            with np.load(source) as a, np.load(report['output_data']) as b:
                for field in a.files:
                    if field not in ('bevs', 'bev_meta'):
                        np.testing.assert_array_equal(a[field], b[field])
                np.testing.assert_array_equal(a['bevs'][:, [0, 1, 2, 4]], b['bevs'][:, [0, 1, 2, 4]])
                self.assertGreater(b['bevs'][:, 3].sum(), 0)
                self.assertEqual(json.loads(str(b['bev_meta']))['camera_projection'], CAMERA_PROJECTION_CURRENT)
            loaded = DatasetGenerator.load(report['output_data'])
            self.assertEqual(loaded['task_meta'][0]['task_id'], loaded['task_meta'][1]['task_id'])
            with self.assertRaises(FileExistsError):
                repair_camera_targets(source, root/'fixed')
            with self.assertRaises(FileExistsError):
                repair_camera_targets(source, root)

    def test_prefix_is_marked_partial_and_new_projection_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = write_source(root)
            report = repair_camera_targets(source, root/'prefix', limit=1, budget_seconds=20)
            self.assertTrue(report['partial'])
            self.assertEqual(report['total_source_samples'], 2)
            self.assertEqual(report['rebuilt'], 1)
            self.assertTrue(inspect_camera_target_repair(root/'prefix')['partial'])
            with self.assertRaisesRegex(ValueError, '需要旧image输入'):
                repair_camera_targets(report['output_data'], root/'again')
            self.assertEqual(json.loads((root/'again/report.json').read_text())['status'], 'incomplete')

    def test_merged_source_requires_explicit_original_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = write_source(root)
            merged = root/'merged'; merged.mkdir()
            destination = merged/source.name
            destination.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, 'manifest'):
                repair_camera_targets(destination, root/'missing')
            report = repair_camera_targets(destination, root/'explicit', manifest_path=root/'manifest.json')
            self.assertEqual(report['rebuilt'], 2)
            self.assertEqual(inspect_camera_target_repair(root/'explicit')['status'], 'passed')

    def test_invalid_goal_fails_without_skipping_or_writing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = write_source(root)
            with np.load(source) as archive:
                data = {key: archive[key] for key in archive.files}
            data['goals'][1, 0] += 1
            np.savez_compressed(source, **data)
            with self.assertRaisesRegex(ValueError, 'selected_goal'):
                repair_camera_targets(source, root/'bad', budget_seconds=20)
            self.assertFalse((root/'bad/train.npz').exists())
            self.assertEqual(json.loads((root/'bad/report.json').read_text())['status'], 'incomplete')

    def test_checkpoint_roundtrip_preserves_version_without_changing_weights(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            for version in (None, CAMERA_PROJECTION_CURRENT):
                config = {'max_horizon': 3, 'hidden_dim': 8, 'dt': .2}
                if version:
                    config['camera_projection'] = version
                model = build_model('net-v1', config)
                path = Path(tmp)/f'{version}.pt'
                torch.save({'schema_version': 1, 'model_name': 'net-v1', 'model_config': config,
                            'model_state': model.state_dict(), 'epoch': 0}, path)
                loaded = load_model_checkpoint(path)
                self.assertEqual(loaded.model_config, config)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.model.state_dict()[key]))
                bev = BEVTensor(np.zeros((5, 16, 16)), .5, (4, 4, 4, 4),
                                ['a', 'b', 'c', 'target', 'vehicle'], CAMERA_PROJECTION_CURRENT)
                if version:
                    loaded.model.predict(bev, GoalPose(2, 0, 0), VehicleState(0, 0, 0))
                else:
                    with self.assertRaisesRegex(ValueError, 'camera_projection'):
                        loaded.model.predict(bev, GoalPose(2, 0, 0), VehicleState(0, 0, 0))


if __name__ == '__main__':
    unittest.main()
