"""MineParkingNet v1/v2 变长输出测试。"""

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
