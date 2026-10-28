"""终点保持的梯度、teacher输入及旧checkpoint兼容回归。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from model import hold_terminal_targets, terminal_hold_loss, variable_loss_fn
from model.prediction import TrajectoryPrediction
from training.trainer import Trainer, TrainerConfig, TrainingHistory


class _TeacherSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))
        self.teacher = None

    def forward_with_stop(self, bev, goal, state, *, teacher_points=None, **kwargs):
        self.teacher = teacher_points
        return TrajectoryPrediction(self.value.expand(len(bev), 4, 3),
                                    self.value.expand(len(bev), 4))


def tensors():
    target = torch.zeros(2, 4, 3)
    target[0, :2, 0] = torch.tensor([1., 2.])
    target[1, :3, 0] = torch.tensor([1., 2., 3.])
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.]])
    return target, mask


class TestTerminalHold(unittest.TestCase):
    def test_teacher_repeats_endpoint_without_mutating_labels(self):
        target, mask = tensors()
        old = target.clone()
        held = hold_terminal_targets(target, mask)
        self.assertTrue(torch.equal(target, old))
        self.assertTrue(torch.equal(held[mask.bool()], target[mask.bool()]))
        self.assertEqual(held[0, 2:, 0].tolist(), [2., 2.])
        self.assertEqual(held[1, 3:, 0].tolist(), [3.])

    def test_only_suffix_has_direct_gradient_and_sequences_are_equally_weighted(self):
        target, mask = tensors()
        held = hold_terminal_targets(target, mask)
        pred = held.clone()
        pred[0, 2:, 0] += 1.
        pred[1, 3:, 0] += 3.
        pred.requires_grad_()
        value = terminal_hold_loss(pred, target, mask)
        self.assertAlmostEqual(value.item(), 5.)  # (1^2 + 3^2) / 2, not /3 points
        value.backward()
        self.assertEqual(torch.count_nonzero(pred.grad[mask.bool()]).item(), 0)
        self.assertTrue((pred.grad[~mask.bool(), 0] > 0).all())

    def test_no_suffix_is_differentiable_zero(self):
        target, _ = tensors()
        pred = target.clone().requires_grad_()
        value = terminal_hold_loss(pred, target, torch.ones(2, 4))
        value.backward()
        self.assertEqual(value.item(), 0.)
        self.assertEqual(torch.count_nonzero(pred.grad).item(), 0)

    def test_wrapped_yaw_and_configured_weight(self):
        target = torch.zeros(1, 2, 3)
        pred = target.clone(); pred[0, 1, 2] = 2 * torch.pi
        mask = torch.tensor([[1., 0.]])
        self.assertLess(terminal_hold_loss(pred, target, mask, yaw_error_mode='wrapped').item(), 1e-10)
        self.assertGreater(terminal_hold_loss(pred, target, mask, yaw_weight=2.).item(), 70.)

    def test_bad_masks_and_nonfinite_are_rejected(self):
        target, mask = tensors()
        for bad in [torch.zeros_like(mask), torch.tensor([[1., 0., 1., 0.], [1., 1., 1., 0.]]), mask * .5]:
            with self.subTest(mask=bad), self.assertRaises(ValueError):
                hold_terminal_targets(target, bad)
        bad_target = target.clone(); bad_target[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            terminal_hold_loss(target, bad_target, mask)
        with self.assertRaises(ValueError):
            terminal_hold_loss(target[:, :2], target, mask)

    def test_trainer_teacher_and_loss_are_opt_in(self):
        target, mask = tensors()
        batch = (torch.zeros(2, 5, 8, 8), torch.zeros(2, 3), torch.zeros(2, 2), target, mask)
        base = TrainerConfig(stop_target_mode='cumulative')
        model = _TeacherSpy()
        trainer = Trainer(model, base, model_name='spy')
        original, _ = trainer._batch_loss(batch, training=True, epoch=0, batch_index=0)
        self.assertTrue(torch.equal(model.teacher, target))
        expected = variable_loss_fn(model.value.expand(2, 4, 3), model.value.expand(2, 4),
                                    target, mask, stop_weight=.2, balance_stop=True,
                                    stop_target_mode='cumulative')
        self.assertTrue(torch.equal(original, expected))
        held_model = _TeacherSpy()
        held_trainer = Trainer(held_model, replace(base, terminal_hold_weight=.25), model_name='spy')
        held, _ = held_trainer._batch_loss(batch, training=True, epoch=0, batch_index=0)
        self.assertTrue(torch.equal(held_model.teacher, hold_terminal_targets(target, mask)))
        self.assertGreater(held.item(), original.item())
        val, _ = held_trainer._batch_loss(batch, training=False, epoch=0, batch_index=0)
        self.assertIsNone(held_model.teacher)
        self.assertAlmostEqual(val.item(), held.item())
        self.assertEqual(target[0, 2, 0].item(), 0.)

    def test_old_checkpoint_missing_weight_defaults_to_zero_and_cross_policy_resume_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            base = TrainerConfig(stop_target_mode='cumulative', checkpoint_dir=temp)
            trainer = Trainer(_TeacherSpy(), base, model_name='spy')
            payload = trainer._checkpoint_payload(0, TrainingHistory())
            payload['trainer_config'].pop('terminal_hold_weight')
            path = Path(temp)/'old.pt'; torch.save(payload, path)
            trainer.load_checkpoint(path)
            changed = Trainer(_TeacherSpy(), replace(base, terminal_hold_weight=.25), model_name='spy')
            with self.assertRaises(RuntimeError):
                changed.load_checkpoint(path)
            new_path = Path(temp)/'new.pt'
            torch.save(changed._checkpoint_payload(0, TrainingHistory()), new_path)
            with self.assertRaises(RuntimeError):
                trainer.load_checkpoint(new_path)

    def test_config_validation(self):
        self.assertEqual(TrainerConfig().terminal_hold_weight, 0.)
        for weight in [-1., float('nan'), float('inf')]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                TrainerConfig(terminal_hold_weight=weight)
        with self.assertRaises(ValueError):
            TrainerConfig(terminal_hold_weight=.25, stop_target_mode='terminal')


if __name__ == '__main__':
    unittest.main()
