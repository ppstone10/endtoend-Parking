"""Trainer early stopping 与 checkpoint 测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from training import Trainer, TrainerConfig, epoch_batches


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
            self.assertEqual(len(history.train_rollout_ade_m), 2)
            self.assertEqual(len(history.val_rollout_fde_m), 2)
            self.assertEqual(len(history.teacher_forcing_ratio), 2)
            self.assertTrue(Path(temp, "best.pt").exists())
            self.assertTrue(Path(temp, "last.pt").exists())
            self.assertFalse(any(Path(temp).glob("*.tmp")))

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


if __name__ == "__main__":
    unittest.main()
