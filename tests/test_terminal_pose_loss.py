"""终点位姿损失测试：只压末点、航向加权、形状与取值校验。"""

from __future__ import annotations

import unittest

import torch

from model import terminal_pose_loss
from training.trainer import TrainerConfig


class TestTerminalPoseLoss(unittest.TestCase):
    def _tensors(self, *, pred_tail_delta: float = 0.0, yaw_delta: float = 0.0):
        """构造 2 条样本：长度 4 与 3，末点带可控偏差。"""
        pred = torch.zeros((2, 5, 3), dtype=torch.float32)
        target = torch.zeros((2, 5, 3), dtype=torch.float32)
        for batch, length in enumerate((4, 3)):
            for step in range(length):
                pred[batch, step, 0] = float(step)
                target[batch, step, 0] = float(step)
        pred[0, 3, 0] += pred_tail_delta
        pred[1, 2, 0] += pred_tail_delta
        pred[0, 3, 2] += yaw_delta
        pred[1, 2, 2] += yaw_delta
        mask = torch.zeros((2, 5), dtype=torch.float32)
        mask[0, :4] = 1.0
        mask[1, :3] = 1.0
        return pred, target, mask

    def test_perfect_prediction_is_zero(self):
        pred, target, mask = self._tensors()
        self.assertAlmostEqual(float(terminal_pose_loss(pred, target, mask)), 0.0, places=6)

    def test_only_terminal_point_matters(self):
        """改动非末点不应影响损失——这正是与逐点损失的关键区别。"""
        pred, target, mask = self._tensors()
        baseline = float(terminal_pose_loss(pred, target, mask))
        perturbed = pred.clone()
        perturbed[0, 0, 0] += 5.0  # 首点大幅偏移
        perturbed[1, 1, 0] += 5.0  # 中间点大幅偏移
        self.assertAlmostEqual(
            float(terminal_pose_loss(perturbed, target, mask)), baseline, places=6
        )

    def test_terminal_position_error_enters_quadratically(self):
        pred, target, mask = self._tensors(pred_tail_delta=0.3)
        value = float(terminal_pose_loss(pred, target, mask))
        # 两条样本各 0.3² = 0.09，均值 0.09
        self.assertAlmostEqual(value, 0.09, places=5)

    def test_yaw_weight_scales_yaw_term(self):
        pred, target, mask = self._tensors(yaw_delta=0.2)
        unit = float(terminal_pose_loss(pred, target, mask, yaw_weight=1.0))
        weighted = float(terminal_pose_loss(pred, target, mask, yaw_weight=10.0))
        self.assertAlmostEqual(unit, 0.04, places=5)
        self.assertAlmostEqual(weighted, 0.40, places=4)

    def test_default_yaw_weight_equates_5p7deg_with_0p3m(self):
        """设计声明的等效关系：0.1rad(≈5.7°) 航向 ≈ 0.3m 位置（默认权重 10）。"""
        yaw_pred, target, mask = self._tensors(yaw_delta=0.1)
        pos_pred, _, _ = self._tensors(pred_tail_delta=0.3)
        yaw_value = float(terminal_pose_loss(yaw_pred, target, mask))
        pos_value = float(terminal_pose_loss(pos_pred, target, mask))
        self.assertAlmostEqual(yaw_value / pos_value, 10.0 * 0.01 / 0.09, places=4)

    def test_default_yaw_weight_is_not_equal_magnitude(self):
        """明确否定"数值同量级"的解读：等量级要求权重约 0.33，会让 3° 等同 0.3m。"""
        yaw_pred, target, mask = self._tensors(yaw_delta=0.17453292519943295)
        pos_pred, _, _ = self._tensors(pred_tail_delta=0.1)
        yaw_value = float(terminal_pose_loss(yaw_pred, target, mask))
        pos_value = float(terminal_pose_loss(pos_pred, target, mask))
        # 默认权重下 10° 航向远重于 0.1m 位置，说明并未被位置项淹没
        self.assertGreater(yaw_value / pos_value, 10.0)

    def test_rejects_shape_mismatch(self):
        pred, target, mask = self._tensors()
        with self.assertRaises(ValueError):
            terminal_pose_loss(pred, target[:, :3], mask)

    def test_rejects_empty_sample(self):
        pred, target, mask = self._tensors()
        mask[1, :] = 0.0
        with self.assertRaises(ValueError):
            terminal_pose_loss(pred, target, mask)

    def test_rejects_negative_yaw_weight(self):
        pred, target, mask = self._tensors()
        with self.assertRaises(ValueError):
            terminal_pose_loss(pred, target, mask, yaw_weight=-1.0)


class TestTrainerConfigTerminalPose(unittest.TestCase):
    def test_defaults_disabled(self):
        config = TrainerConfig()
        self.assertEqual(config.terminal_pose_weight, 0.0)
        self.assertEqual(config.terminal_yaw_weight, 10.0)

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            TrainerConfig(terminal_pose_weight=-0.1)

    def test_rejects_negative_yaw_weight(self):
        with self.assertRaises(ValueError):
            TrainerConfig(terminal_yaw_weight=-1.0)

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            TrainerConfig(terminal_pose_weight=float("nan"))


if __name__ == "__main__":
    unittest.main()
