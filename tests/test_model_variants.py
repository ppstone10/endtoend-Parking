"""MineParkingNet v1/v2 变长输出测试。"""

import unittest

import torch
import torch.nn.functional as F

from model import MineParkingNetV1, MineParkingNetV2, variable_loss_fn


class TestVariableModels(unittest.TestCase):
    def setUp(self):
        self.bev = torch.randn(2, 5, 32, 32)
        self.goal = torch.randn(2, 3)
        self.state = torch.randn(2, 2)
        self.target = torch.randn(2, 6, 3)

    def test_v1_returns_points_and_stop_logits_with_teacher_forcing(self):
        model = MineParkingNetV1(max_horizon=6, hidden_dim=32)
        prediction = model.forward_with_stop(
            self.bev, self.goal, self.state, teacher_points=self.target
        )
        self.assertEqual(prediction.points.shape, (2, 6, 3))
        self.assertEqual(prediction.stop_logits.shape, (2, 6))

    def test_v2_returns_same_public_output_contract(self):
        model = MineParkingNetV2(
            max_horizon=6,
            hidden_dim=32,
            base_channels=8,
            attention_heads=4,
        )
        prediction = model.forward_with_stop(self.bev, self.goal, self.state)
        self.assertEqual(prediction.points.shape, (2, 6, 3))
        self.assertEqual(prediction.stop_logits.shape, (2, 6))

    def test_variable_loss_is_finite_with_partial_and_empty_masks(self):
        model = MineParkingNetV1(max_horizon=6, hidden_dim=32)
        prediction = model.forward_with_stop(self.bev, self.goal, self.state)
        mask = torch.zeros(2, 6)
        mask[0, :4] = 1.0
        loss = variable_loss_fn(
            prediction.points, prediction.stop_logits, self.target, mask
        )
        self.assertTrue(torch.isfinite(loss))

    def test_balanced_stop_loss_prevents_all_negative_shortcut(self):
        points = torch.zeros(1, 100, 3)
        logits = torch.full((1, 100), -5.0)
        target = torch.zeros_like(points)
        mask = torch.ones(1, 100)

        unbalanced = variable_loss_fn(
            points, logits, target, mask, stop_weight=1.0, balance_stop=False
        )
        balanced = variable_loss_fn(
            points, logits, target, mask, stop_weight=1.0, balance_stop=True
        )

        self.assertGreater(float(balanced), float(unbalanced) * 10.0)
        single = variable_loss_fn(
            points[:, :1], logits[:, :1], target[:, :1], mask[:, :1],
            stop_weight=1.0, balance_stop=True,
        )
        self.assertTrue(torch.isfinite(single))

    def test_cumulative_stop_supervision_covers_terminal_suffix(self):
        points = torch.zeros(1, 4, 3)
        target = torch.zeros_like(points)
        logits = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        loss = variable_loss_fn(
            points,
            logits,
            target,
            mask,
            stop_weight=1.0,
            balance_stop=False,
            stop_target_mode="cumulative",
        )
        expected_targets = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        expected = F.binary_cross_entropy_with_logits(logits, expected_targets)

        self.assertTrue(torch.allclose(loss, expected))

    def test_stop_target_mode_is_strict(self):
        with self.assertRaisesRegex(ValueError, "stop_target_mode"):
            variable_loss_fn(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2),
                torch.zeros(1, 2, 3),
                torch.ones(1, 2),
                stop_target_mode="unknown",
            )

    def test_scheduled_sampling_is_deterministic_and_changes_feedback(self):
        model = MineParkingNetV1(max_horizon=6, hidden_dim=32)
        teacher = model.forward_with_stop(
            self.bev,
            self.goal,
            self.state,
            teacher_points=self.target,
            teacher_forcing_ratio=1.0,
        )
        free = model.forward_with_stop(
            self.bev,
            self.goal,
            self.state,
            teacher_points=self.target,
            teacher_forcing_ratio=0.0,
        )
        first_generator = torch.Generator().manual_seed(17)
        second_generator = torch.Generator().manual_seed(17)
        mixed_a = model.forward_with_stop(
            self.bev,
            self.goal,
            self.state,
            teacher_points=self.target,
            teacher_forcing_ratio=0.5,
            sampling_generator=first_generator,
        )
        mixed_b = model.forward_with_stop(
            self.bev,
            self.goal,
            self.state,
            teacher_points=self.target,
            teacher_forcing_ratio=0.5,
            sampling_generator=second_generator,
        )

        self.assertFalse(torch.allclose(teacher.points, free.points))
        self.assertTrue(torch.allclose(mixed_a.points, mixed_b.points))


if __name__ == "__main__":
    unittest.main()
