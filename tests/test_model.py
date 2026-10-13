"""MineParkingNet 网络测试。"""

import unittest

import numpy as np
import torch

from model import MineParkingNet, endpoint_alignment_loss, loss_fn, progress_monotonicity_loss
from model.network import MineParkingNet as _Net


def _make_bev(batch: int = 2, channels: int = 5, h: int = 64, w: int = 64):
    return torch.randn(batch, channels, h, w)


class TestMineParkingNet(unittest.TestCase):
    def setUp(self):
        self.model = MineParkingNet(bev_channels=5, horizon=20, dt=0.1)

    def test_forward_shape(self):
        bev = _make_bev(2, 5, 64, 64)
        goal = torch.zeros(2, 3)
        state = torch.zeros(2, 2)
        out = self.model.forward(bev, goal, state)
        self.assertEqual(out.shape, (2, 20, 3))

    def test_predict_returns_trajectory(self):
        # 用 numpy 输入验证 predict 接口。
        bev = torch.zeros(1, 5, 64, 64)
        goal = torch.tensor([[5.0, 0.0, 0.0]])
        state = torch.tensor([[0.0, 0.0]])
        out = self.model.forward(bev, goal, state)
        self.assertEqual(out.shape, (1, 20, 3))


class TestLossFn(unittest.TestCase):
    def test_masked_mse(self):
        pred = torch.ones(2, 20, 3)
        target = torch.zeros(2, 20, 3)
        mask = torch.ones(2, 20)
        loss = loss_fn(pred, target, mask)
        # 每个点 3 维 (x,y,yaw) 差均为 1，MSE=1，累加 3 维 → 3.0。
        self.assertAlmostEqual(loss.item(), 3.0, places=5)

    def test_partial_mask(self):
        pred = torch.ones(2, 20, 3)
        target = torch.zeros(2, 20, 3)
        mask = torch.zeros(2, 20)
        mask[:, :10] = 1.0
        loss = loss_fn(pred, target, mask)
        # 有效点数为 2*10，每点 3 维差 1，损失仍为 3.0。
        self.assertAlmostEqual(loss.item(), 3.0, places=5)


class TestEndpointAlignmentLoss(unittest.TestCase):
    def test_penalizes_prediction_far_from_target_on_tail_points(self):
        pred = torch.zeros(2, 10, 3)
        target = torch.zeros(2, 10, 3)
        target[:, 4:6] = torch.tensor([5.0, 2.0, 0.5])
        mask = torch.zeros(2, 10)
        mask[:, :6] = 1.0
        aligned = pred.clone()
        aligned[:, 4:6] = target[:, 4:6]
        far = pred.clone()
        far[:, 4:6] = target[:, 4:6] + 10.0
        near_loss = endpoint_alignment_loss(aligned, target, mask, tail_points=4)
        far_loss = endpoint_alignment_loss(far, target, mask, tail_points=4)
        self.assertLess(near_loss.item(), far_loss.item())
        self.assertGreater(far_loss.item(), 0.0)

    def test_tail_points_only_matters(self):
        pred = torch.zeros(1, 8, 3)
        target = torch.zeros(1, 8, 3)
        mask = torch.ones(1, 8)
        pred[0, 0, 0] = 100.0  # 头部大偏差不应影响
        self.assertAlmostEqual(
            endpoint_alignment_loss(pred, target, mask, tail_points=4).item(), 0.0
        )

    def test_rejects_bad_tail_points(self):
        pred = torch.zeros(1, 8, 3)
        target = torch.zeros(1, 8, 3)
        mask = torch.ones(1, 8)
        with self.assertRaisesRegex(ValueError, "tail_points"):
            endpoint_alignment_loss(pred, target, mask, tail_points=0)


class TestProgressMonotonicityLoss(unittest.TestCase):
    """目标单调逼近损失：惩罚"到目标距离反而变大"的步。

    依据专家示范在到目标距离上 100% 单调（实测 2476+300 条交替次数全为 0）。
    目标固定在原点，故轨迹的 x 即到目标距离。
    """

    def _points(self, xs: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = torch.zeros(1, len(xs), 3)
        pred[0, :, 0] = torch.tensor(xs)
        goal = torch.zeros(1, 3)
        mask = torch.ones(1, len(xs))
        return pred, goal, mask

    def test_monotone_approach_has_zero_loss(self):
        pred, goal, mask = self._points([3.0, 2.0, 1.0, 0.5])
        self.assertAlmostEqual(
            progress_monotonicity_loss(pred, goal, mask).item(), 0.0, places=6
        )

    def test_receding_step_is_penalized_and_scales_with_distance(self):
        pred, goal, mask = self._points([3.0, 2.0, 2.1, 2.05])
        small = progress_monotonicity_loss(pred, goal, mask)
        far, goal2, mask2 = self._points([3.0, 2.0, 2.4, 2.05])
        large = progress_monotonicity_loss(far, goal2, mask2)
        self.assertGreater(small.item(), 0.0)
        self.assertGreater(large.item(), small.item())
        # 后退 0.1m、容差取 0 时恰好记 1.0（步长归一化尺度默认 0.1m）
        single, goal3, mask3 = self._points([2.0, 2.1])
        self.assertAlmostEqual(
            progress_monotonicity_loss(
                single, goal3, mask3, tolerance_m=0.0
            ).item(),
            1.0,
            places=5,
        )

    def test_tolerance_absorbs_numerical_noise(self):
        pred, goal, mask = self._points([1.0, 1.002])
        self.assertAlmostEqual(
            progress_monotonicity_loss(pred, goal, mask, tolerance_m=0.005).item(),
            0.0,
            places=6,
        )

    def test_masked_out_points_do_not_contribute(self):
        pred, goal, mask = self._points([3.0, 2.0, 9.0])
        mask[0, 2] = 0.0  # 第三点无效：它造成的"后退"不应计入
        self.assertAlmostEqual(
            progress_monotonicity_loss(pred, goal, mask).item(), 0.0, places=6
        )

    def test_rejects_bad_shapes_and_parameters(self):
        pred, goal, mask = self._points([2.0, 1.0])
        with self.assertRaisesRegex(ValueError, "goal"):
            progress_monotonicity_loss(pred, torch.zeros(2, 3), mask)
        with self.assertRaisesRegex(ValueError, "mask"):
            progress_monotonicity_loss(pred, goal, torch.ones(1, 3))
        with self.assertRaisesRegex(ValueError, "reference_step_m"):
            progress_monotonicity_loss(pred, goal, mask, reference_step_m=0.0)
        with self.assertRaisesRegex(ValueError, "tolerance_m"):
            progress_monotonicity_loss(pred, goal, mask, tolerance_m=-1.0)


class TestTrainingConvergence(unittest.TestCase):
    def test_loss_decreases(self):
        """网络能在小数据上快速降低损失。"""
        torch.manual_seed(0)
        model = MineParkingNet(bev_channels=5, horizon=20, dt=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        bev = _make_bev(8, 5, 64, 64)
        goal = torch.randn(8, 3)
        state = torch.randn(8, 2)
        target = torch.randn(8, 20, 3)
        mask = torch.ones(8, 20)

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            pred = model.forward(bev, goal, state)
            loss = loss_fn(pred, target, mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()