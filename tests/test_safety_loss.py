"""完整车体连续扫掠安全损失测试。"""

import unittest

import torch

from training.safety import (
    SafetyGeometry,
    SweptFootprintLoss,
    build_clearance_fields,
)


class TestSweptFootprintLoss(unittest.TestCase):
    def setUp(self):
        geometry = SafetyGeometry(
            vehicle_length_m=2.0,
            vehicle_width_m=1.0,
            collision_margin_m=0.0,
            bev_resolution_m=0.25,
            bev_extent_m=(5.0, 5.0, 5.0, 5.0),
            occupancy_channel=0,
        )
        self.loss = SweptFootprintLoss(
            geometry,
            extra_margin_m=0.0,
            sample_spacing_m=0.25,
            max_swept_substeps=16,
        )

    @staticmethod
    def _bev_with_obstacle(x: float, y: float) -> torch.Tensor:
        bev = torch.zeros(1, 1, 40, 40)
        row = int((5.0 - x) / 0.25)
        col = int((y + 5.0) / 0.25)
        bev[0, 0, row, col] = 1.0
        return bev

    def test_collision_has_higher_loss_and_gradient_than_safe_path(self):
        bev = self._bev_with_obstacle(2.0, 0.0)
        mask = torch.ones(1, 2)
        safe = torch.tensor([[[0.0, -2.0, 0.0], [2.0, -2.0, 0.0]]])
        collision = torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], requires_grad=True
        )
        safe_loss = self.loss(bev, safe, mask)
        collision_loss = self.loss(bev, collision, mask)
        self.assertGreater(float(collision_loss.detach()), float(safe_loss.detach()))
        collision_loss.backward()
        self.assertIsNotNone(collision.grad)
        self.assertGreater(float(collision.grad.abs().sum()), 0.0)

    def test_swept_rotation_detects_intermediate_collision(self):
        bev = self._bev_with_obstacle(0.0, 0.9)
        mask = torch.ones(1, 1)
        rotating = torch.tensor([[[0.0, 0.0, torch.pi / 2.0]]])
        stationary = torch.tensor([[[0.0, 0.0, 0.0]]])
        self.assertGreater(float(self.loss(bev, rotating, mask)), float(self.loss(bev, stationary, mask)))

    def test_out_of_bounds_is_penalized(self):
        bev = torch.zeros(1, 1, 40, 40)
        mask = torch.ones(1, 1)
        inside = torch.tensor([[[0.0, 0.0, 0.0]]])
        outside = torch.tensor([[[5.0, 0.0, 0.0]]])
        self.assertGreater(float(self.loss(bev, outside, mask)), float(self.loss(bev, inside, mask)))


class TestClearanceFieldLoss(unittest.TestCase):
    def setUp(self):
        self.geometry = SafetyGeometry(
            vehicle_length_m=2.0,
            vehicle_width_m=1.0,
            collision_margin_m=0.25,
            bev_resolution_m=0.25,
            bev_extent_m=(5.0, 5.0, 5.0, 5.0),
            occupancy_channel=0,
        )
        self.loss = SweptFootprintLoss(
            self.geometry,
            extra_margin_m=0.25,
            sample_spacing_m=0.25,
            max_swept_substeps=16,
            mode="clearance_field",
        )

    @staticmethod
    def _bev_with_obstacle(x: float, y: float) -> torch.Tensor:
        bev = torch.zeros(1, 1, 40, 40)
        row = int((5.0 - x) / 0.25)
        col = int((y + 5.0) / 0.25)
        bev[0, 0, row, col] = 1.0
        return bev

    def test_signed_clearance_contains_metric_obstacle_and_boundary_distance(self):
        bev = self._bev_with_obstacle(0.0, 0.0)
        field = build_clearance_fields(
            bev, self.geometry, extra_margin_m=0.25
        ).float()
        row, col = 20, 20
        self.assertAlmostEqual(float(field[0, 0, row, col]), -0.25, places=3)
        self.assertAlmostEqual(float(field[0, 0, row, col + 1]), 0.25, places=3)
        self.assertAlmostEqual(float(field[0, 0, 0, 0]), 0.125, places=3)

    def test_clearance_gradient_pushes_vehicle_away_before_collision(self):
        bev = self._bev_with_obstacle(2.0, 0.0)
        field = build_clearance_fields(
            bev, self.geometry, extra_margin_m=0.25
        )
        mask = torch.ones(1, 1)
        near = torch.tensor([[[0.75, 0.0, 0.0]]], requires_grad=True)
        far = torch.tensor([[[0.0, 0.0, 0.0]]])
        near_loss = self.loss(bev, near, mask, field)
        far_loss = self.loss(bev, far, mask, field)
        self.assertGreater(float(near_loss.detach()), float(far_loss.detach()))
        near_loss.backward()
        self.assertGreater(float(near.grad[0, 0, 0]), 0.0)

    def test_clearance_loss_requires_precomputed_field(self):
        with self.assertRaisesRegex(ValueError, "预计算净空场"):
            self.loss(
                torch.zeros(1, 1, 40, 40),
                torch.zeros(1, 1, 3),
                torch.ones(1, 1),
            )


if __name__ == "__main__":
    unittest.main()
