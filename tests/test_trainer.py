"""Trainer early stopping 与 checkpoint 测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from training import (
    SafetyGeometry,
    SweptFootprintLoss,
    Trainer,
    TrainerConfig,
    epoch_batches,
)


class _TinyTrajectoryModel(nn.Module):
    def __init__(self, horizon: int = 4) -> None:
        super().__init__()
        self.horizon = horizon
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, bev, goal, state):
        return self.value.expand(bev.shape[0], self.horizon, 3)


def _batch():
    return (
        torch.zeros(2, 5, 8, 8),
        torch.zeros(2, 3),
        torch.zeros(2, 2),
        torch.ones(2, 4, 3),
        torch.ones(2, 4),
    )


class TestTrainer(unittest.TestCase):
    def test_epoch_batches_shuffle_samples_deterministically(self):
        source = []
        for start in range(0, 8, 2):
            batch = list(_batch())
            batch[1] = torch.arange(start, start + 2, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
            source.append(tuple(batch))

        def order(epoch):
            return [
                int(value)
                for batch in epoch_batches(tuple(source), shuffle=True, seed=23, epoch=epoch)
                for value in batch[1][:, 0]
            ]

        self.assertEqual(order(0), order(0))
        self.assertNotEqual(order(0), list(range(8)))
        self.assertNotEqual(order(0), order(1))
        self.assertEqual(sorted(order(0)), list(range(8)))

    def test_balanced_recovery_batches_contain_all_three_semantic_pools(self):
        source = []
        identifiers = torch.arange(12, dtype=torch.float32)
        groups = torch.tensor([0] * 8 + [1] * 2 + [2] * 2)
        for start in range(0, 12, 4):
            batch = list(_batch())
            batch = [field[:0] for field in batch]
            end = start + 4
            batch[0] = torch.zeros(end - start, 5, 8, 8)
            batch[1] = identifiers[start:end, None].repeat(1, 3)
            batch[2] = torch.zeros(end - start, 2)
            batch[3] = torch.zeros(end - start, 4, 3)
            batch[4] = torch.ones(end - start, 4)
            batch.append(groups[start:end])
            source.append(tuple(batch))

        def balanced(epoch):
            return epoch_batches(
                tuple(source),
                shuffle=True,
                seed=17,
                epoch=epoch,
                balance_recovery=True,
            )

        first = balanced(0)
        self.assertEqual(
            [[int((batch[-1] == value).sum()) for value in range(3)] for batch in first],
            [[2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1]],
        )
        self.assertEqual(
            [batch[1][:, 0].tolist() for batch in first],
            [batch[1][:, 0].tolist() for batch in balanced(0)],
        )
        self.assertNotEqual(
            [batch[1][:, 0].tolist() for batch in first],
            [batch[1][:, 0].tolist() for batch in balanced(1)],
        )

    def test_balanced_recovery_batches_require_all_pools(self):
        batch = tuple(field.repeat((2,) + (1,) * (field.ndim - 1)) for field in _batch()) + (
            torch.tensor([0, 0, 1, 1]),
        )
        with self.assertRaisesRegex(ValueError, "三个样本池"):
            epoch_batches(
                (batch,),
                shuffle=True,
                seed=0,
                epoch=0,
                balance_recovery=True,
            )

    def test_teacher_forcing_schedule_is_linear_and_bounded(self):
        config = TrainerConfig(
            teacher_forcing_start=1.0,
            teacher_forcing_end=0.2,
            teacher_forcing_decay_epochs=4,
        )
        self.assertEqual(config.teacher_forcing_ratio(0), 1.0)
        self.assertAlmostEqual(config.teacher_forcing_ratio(2), 0.6)
        self.assertEqual(config.teacher_forcing_ratio(4), 0.2)
        self.assertEqual(config.teacher_forcing_ratio(20), 0.2)

    def test_early_stopping_and_atomic_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            config = TrainerConfig(
                epochs=5,
                learning_rate=1e-6,
                patience=1,
                min_delta=10.0,
                checkpoint_dir=temp,
            )
            trainer = Trainer(
                _TinyTrajectoryModel(), config, model_name="test-model"
            )
            history = trainer.fit([_batch()], [_batch()])
            self.assertTrue(history.stopped_early)
            self.assertEqual(len(history.train_loss), 2)
            self.assertEqual(len(history.val_collision_loss), 2)
            self.assertEqual(len(history.train_rollout_ade_m), 2)
            self.assertEqual(len(history.val_rollout_fde_m), 2)
            self.assertEqual(len(history.teacher_forcing_ratio), 2)
            self.assertTrue(Path(temp, "best.pt").exists())
            self.assertTrue(Path(temp, "last.pt").exists())
            self.assertFalse(any(Path(temp).glob("*.tmp")))

    def test_early_stopping_waits_for_curriculum_warmup(self):
        with tempfile.TemporaryDirectory() as temp:
            config = TrainerConfig(
                epochs=5,
                learning_rate=1e-6,
                patience=1,
                min_delta=10.0,
                checkpoint_dir=temp,
                teacher_forcing_decay_epochs=2,
                early_stopping_start_epoch=2,
            )
            trainer = Trainer(_TinyTrajectoryModel(), config, model_name="test-model")

            history = trainer.fit([_batch()], [_batch()])

            self.assertTrue(history.stopped_early)
            self.assertEqual(len(history.train_loss), 3)
            self.assertEqual(history.early_stopping_active, [False, False, True])
            self.assertEqual(history.teacher_forcing_ratio[-1], 0.2)

    def test_resume_reconstructs_post_warmup_patience(self):
        with tempfile.TemporaryDirectory() as temp:
            initial = TrainerConfig(
                epochs=5,
                learning_rate=1e-6,
                patience=2,
                min_delta=10.0,
                checkpoint_dir=temp,
                early_stopping_start_epoch=4,
            )
            Trainer(_TinyTrajectoryModel(), initial, model_name="same").fit(
                [_batch()], [_batch()]
            )
            resumed_config = TrainerConfig(
                epochs=7,
                learning_rate=1e-6,
                patience=2,
                min_delta=10.0,
                checkpoint_dir=temp,
                early_stopping_start_epoch=4,
            )
            resumed = Trainer(_TinyTrajectoryModel(), resumed_config, model_name="same")

            history = resumed.fit(
                [_batch()], [_batch()], resume_from=Path(temp, "last.pt")
            )

            self.assertTrue(history.stopped_early)
            self.assertEqual(len(history.train_loss), 6)

    def test_checkpoint_rejects_different_model_name(self):
        with tempfile.TemporaryDirectory() as temp:
            config = TrainerConfig(
                epochs=1, patience=1, checkpoint_dir=temp
            )
            first = Trainer(_TinyTrajectoryModel(), config, model_name="left")
            first.fit([_batch()], [_batch()])
            other = Trainer(_TinyTrajectoryModel(), config, model_name="right")
            with self.assertRaisesRegex(RuntimeError, "模型变体"):
                other.load_checkpoint(Path(temp, "last.pt"))

    def test_checkpoint_rejects_deployment_artifact_for_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            config = TrainerConfig(epochs=1, patience=1, checkpoint_dir=temp)
            trainer = Trainer(_TinyTrajectoryModel(), config, model_name="same")
            trainer.fit([_batch()], [_batch()])
            payload = torch.load(Path(temp, "last.pt"), weights_only=False)
            payload["resumable"] = False
            deployment = Path(temp, "deployment.pt")
            torch.save(payload, deployment)

            with self.assertRaisesRegex(RuntimeError, "不可恢复训练"):
                trainer.load_checkpoint(deployment)

    def test_checkpoint_rejects_incompatible_training_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temp:
            first_config = TrainerConfig(
                epochs=1, patience=1, checkpoint_dir=temp
            )
            first = Trainer(_TinyTrajectoryModel(), first_config, model_name="same")
            first.fit([_batch()], [_batch()])
            changed = Trainer(
                _TinyTrajectoryModel(),
                TrainerConfig(
                    epochs=2,
                    learning_rate=2e-3,
                    patience=1,
                    checkpoint_dir=temp,
                ),
                model_name="same",
            )
            with self.assertRaisesRegex(RuntimeError, "训练超参数"):
                changed.load_checkpoint(Path(temp, "last.pt"))

    def test_checkpoint_rejects_changed_training_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Trainer(
                _TinyTrajectoryModel(),
                TrainerConfig(epochs=1, patience=1, checkpoint_dir=temp),
                model_name="same",
            )
            first.fit([_batch()], [_batch()])
            changed = Trainer(
                _TinyTrajectoryModel(),
                TrainerConfig(
                    epochs=2,
                    patience=1,
                    checkpoint_dir=temp,
                    shuffle_train=False,
                ),
                model_name="same",
            )
            with self.assertRaisesRegex(RuntimeError, "训练超参数"):
                changed.load_checkpoint(Path(temp, "last.pt"))

    def test_rejects_early_stopping_start_outside_run(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_start_epoch"):
            TrainerConfig(epochs=3, early_stopping_start_epoch=3)

    def test_collision_loss_is_recorded_and_safety_geometry_is_resume_semantics(self):
        def safety(length: float):
            return SweptFootprintLoss(
                SafetyGeometry(length, 0.25, 0.0, 0.25, (1.0, 1.0, 1.0, 1.0), 0),
                extra_margin_m=0.0,
                sample_spacing_m=0.25,
            )

        batch = list(_batch())
        batch[0][:, 0, 4, 4] = 1.0
        with tempfile.TemporaryDirectory() as temp:
            config = TrainerConfig(
                epochs=1,
                patience=1,
                checkpoint_dir=temp,
                collision_loss_weight=0.5,
            )
            trainer = Trainer(
                _TinyTrajectoryModel(),
                config,
                model_name="safe",
                safety_loss=safety(0.5),
            )
            history = trainer.fit([tuple(batch)], [tuple(batch)])
            self.assertGreater(history.val_collision_loss[0], 0.0)
            changed = Trainer(
                _TinyTrajectoryModel(),
                TrainerConfig(
                    epochs=2,
                    patience=1,
                    checkpoint_dir=temp,
                    collision_loss_weight=0.5,
                ),
                model_name="safe",
                safety_loss=safety(0.75),
            )
            with self.assertRaisesRegex(RuntimeError, "安全几何"):
                changed.load_checkpoint(Path(temp, "last.pt"))


if __name__ == "__main__":
    unittest.main()
