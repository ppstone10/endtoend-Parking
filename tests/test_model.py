"""MineParkingNet 网络测试。"""

import unittest

import numpy as np
import torch

from model import MineParkingNet, loss_fn
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