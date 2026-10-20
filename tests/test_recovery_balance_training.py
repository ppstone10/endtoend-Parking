import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from scripts.prepare_recovery_balance_ab import select_recovery_indices, file_digest
from scripts.train_fixed_threshold import exclusive_run, check_run_state, load_frozen_config, run_training, run_preflight
from training.trainer import Trainer, TrainerConfig


def sample(scene, maneuver, trigger=None, index=0):
    meta = {'task_id': f'{scene}-{maneuver}-{index}', 'scene_name': scene, 'difficulty': {'maneuver': maneuver}}
    if trigger:
        meta['recovery'] = {'source_task_id': meta['task_id'], 'rollout_step': index, 'trigger': trigger}
    return meta


class TestRecoverySelection(unittest.TestCase):
    def test_late_scene_and_all_backtracks_survive(self):
        metadata = [sample('A', 'forward', index=99)]
        metadata += [sample('A', 'reverse', 'stride', i) for i in range(4)]
        metadata += [sample('A', 'forward', 'stride', i) for i in range(8)]
        metadata += [sample('Z', 'forward', 'collision_backtrack', i) for i in range(2)]
        chosen = select_recovery_indices(metadata, seed=42)
        self.assertEqual(len(chosen), 9)
        self.assertIn(len(metadata)-1, chosen)
        self.assertIn(len(metadata)-2, chosen)
        self.assertEqual(chosen, select_recovery_indices(metadata, seed=42))

    def test_protected_samples_cannot_exceed_budget(self):
        metadata = [sample('A', 'reverse', 'stride')]
        metadata += [sample('Z', 'forward', 'collision_backtrack', i) for i in range(2)]
        with self.assertRaises(ValueError):
            select_recovery_indices(metadata, seed=0)

    def test_duplicate_source_states_are_rejected(self):
        row = sample('A', 'reverse', 'stride')
        with self.assertRaises(ValueError):
            select_recovery_indices([row, row, sample('B', 'forward', 'stride')], seed=0)

    def test_missing_source_identity_is_rejected(self):
        row = sample('A', 'reverse', 'stride')
        del row['recovery']['source_task_id']
        with self.assertRaises(ValueError):
            select_recovery_indices([row], seed=0)


class TestRunGuards(unittest.TestCase):
    def test_exclusive_lock_is_not_stolen_and_released_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                with exclusive_run(root):
                    with self.assertRaises(FileExistsError):
                        with exclusive_run(root):
                            self.fail('duplicate lock')
                    raise RuntimeError('simulate')
            self.assertFalse((root / 'run.lock').exists())

    def test_existing_run_requires_explicit_matching_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'arm': 'R1'}
            (root / 'run-identity.json').write_text(json.dumps(identity))
            with self.assertRaises(FileExistsError):
                check_run_state(root, resume=False, identity=identity)
            with self.assertRaises(ValueError):
                check_run_state(root, resume=True, identity={'arm': 'R2'})
            with self.assertRaisesRegex(ValueError, 'epoch'):
                check_run_state(root, resume=True, identity=identity)

    def test_config_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, manifest_path = root / 'config.json', root / 'manifest.json'
            config_path.write_text(json.dumps({'kind': 'recovery_balance_ab', 'arm': 'R1', 'manifest': str(manifest_path)}))
            manifest_path.write_text(json.dumps({'status': 'ready_for_user_training', 'arms': {'R1': {'config': str(config_path), 'sha256': 'wrong'}}}))
            with self.assertRaisesRegex(ValueError, '配置身份'):
                load_frozen_config(config_path)

    def test_input_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, manifest_path, data_path = root / 'config.json', root / 'manifest.json', root / 'data'
            data_path.write_text('changed')
            config_path.write_text(json.dumps({'kind': 'recovery_balance_ab', 'arm': 'R1', 'manifest': str(manifest_path)}))
            manifest_path.write_text(json.dumps({'status': 'ready_for_user_training',
                'arms': {'R1': {'config': str(config_path), 'sha256': file_digest(config_path)}},
                'code': {}, 'runtime': {}, 'inputs': {'train': {'path': str(data_path), 'sha256': 'old'}}}))
            with patch('scripts.train_fixed_threshold.execution_code_identity', return_value={}):
                with patch('scripts.train_fixed_threshold.runtime_identity', return_value={}):
                    with self.assertRaisesRegex(ValueError, '输入文件身份'):
                        load_frozen_config(config_path)

    def test_execution_code_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, manifest_path = root / 'config.json', root / 'manifest.json'
            config_path.write_text(json.dumps({'kind': 'recovery_balance_ab', 'arm': 'R1', 'manifest': str(manifest_path)}))
            manifest_path.write_text(json.dumps({'status': 'ready_for_user_training',
                'arms': {'R1': {'config': str(config_path), 'sha256': file_digest(config_path)}}, 'code': {'old': 'hash'}}))
            with patch('scripts.train_fixed_threshold.execution_code_identity', return_value={}):
                with self.assertRaisesRegex(ValueError, '执行代码'):
                    load_frozen_config(config_path)


class TinyModel(torch.nn.Module):
    """仅用于短时单元测试，不是实际泊车模型训练。"""

    stop_threshold = .26

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 9)

    def forward(self, bev, goal, state):
        return self.layer(bev).reshape(-1, 3, 3)


class TestSyntheticTrainingResume(unittest.TestCase):
    def test_smoke_never_calls_fit_or_updates_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = TrainerConfig(epochs=4, checkpoint_dir=directory)
            trainer = Trainer(TinyModel(), settings, model_name='toy')
            batch = (torch.ones(3, 1), torch.zeros(3, 3), torch.zeros(3, 2),
                     torch.zeros(3, 3, 3), torch.ones(3, 3), torch.tensor([0, 1, 2]))
            config = {'arm': 'R1', 'output': directory, 'batch_size': 3}
            with patch('scripts.train_fixed_threshold.load_frozen_config', return_value=(config, {'sample_counts': {'original': 1}}, 'id')):
                with patch('scripts.train_fixed_threshold.make_training', return_value=(trainer, (batch,), (batch,))):
                    with patch.object(Trainer, 'fit', side_effect=AssertionError('must not fit')):
                        report = run_preflight('unused', smoke=True)
            self.assertTrue(report['model_weights_unchanged'])
            self.assertFalse(report['formal_training_started'])
            self.assertFalse((Path(directory) / 'last.pt').exists())

    def test_resume_restores_optimizer_and_missing_snapshot(self):
        # 2个样本、3个点的玩具模型：不读取正式数据，不启动长任务。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'config.json'
            config_path.write_text('{}')
            batch = (torch.ones(2, 1), torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(2, 3, 3), torch.ones(2, 3))

            def run_arm(output, interrupt=False, resume=False):
                output.mkdir(exist_ok=True)
                settings = TrainerConfig(epochs=4, learning_rate=5e-5, checkpoint_dir=str(output / 'checkpoints'), patience=4,
                                         early_stopping_start_epoch=3, teacher_forcing_start=.44,
                                         teacher_forcing_end=.2, teacher_forcing_decay_epochs=3)
                config = {'output': str(output), 'arm': 'R1', 'model_config': {'stop_threshold': .26}}

                def make(*args, **kwargs):
                    torch.manual_seed(77)
                    trainer = Trainer(TinyModel(), settings, model_name='toy', model_config=config['model_config'])
                    if interrupt:
                        fit = trainer.fit

                        def stop(*a, on_epoch_end, **kw):
                            def callback(epoch, history):
                                on_epoch_end(epoch, history)
                                if epoch == 1:
                                    raise KeyboardInterrupt()
                            return fit(*a, on_epoch_end=callback, **kw)
                        trainer.fit = stop
                    return trainer, (batch,), (batch,)

                with patch('scripts.train_fixed_threshold.load_frozen_config', return_value=(config, {}, 'identity')):
                    with patch('scripts.train_fixed_threshold.make_training', side_effect=make):
                        return run_training(config_path, resume=resume)

            run_arm(root / 'full')
            with self.assertRaises(KeyboardInterrupt):
                run_arm(root / 'resumed', interrupt=True)
            (root / 'resumed/checkpoints/epoch0002.pt').unlink()
            last_path = root / 'resumed/checkpoints/last.pt'
            valid_checkpoint = torch.load(last_path, weights_only=False)
            invalid_checkpoint = {**valid_checkpoint, 'initialization': {'experiment_identity': {'arm': 'R2'}}}
            torch.save(invalid_checkpoint, last_path)
            with self.assertRaisesRegex(ValueError, '检查点不属于'):
                run_arm(root / 'resumed', resume=True)
            torch.save(valid_checkpoint, last_path)
            report = run_arm(root / 'resumed', resume=True)
            self.assertEqual(report['status'], 'completed')
            self.assertTrue((root / 'resumed/checkpoints/epoch0002.pt').is_file())
            full = torch.load(root / 'full/checkpoints/last.pt', weights_only=False)
            resumed = torch.load(root / 'resumed/checkpoints/last.pt', weights_only=False)
            for key, value in full['model_state'].items():
                self.assertTrue(torch.equal(value, resumed['model_state'][key]))
            self.assertEqual(full['history'], resumed['history'])
            self.assertFalse((root / 'resumed/deployment.pt').exists())
            with self.assertRaises(FileExistsError):
                run_arm(root / 'resumed', resume=True)


if __name__ == '__main__':
    unittest.main()
