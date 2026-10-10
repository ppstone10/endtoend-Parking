"""分层净空损失（tiered_clearance）测试。

覆盖设计目标：
- 连续净空场让"近但未接触"可罚（occupancy_max 的二值形式做不到）；
- 接触比近障罚得更重（分层）；
- power-mean 聚合让最危险时刻主导梯度（平均聚合会稀释掉）；
- 目标豁免按"不实际碰撞即视为无碰撞"降权必要路径。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from training.safety import SafetyGeometry, SweptFootprintLoss


def _geometry(required: float = 0.2) -> SafetyGeometry:
    return SafetyGeometry(
        vehicle_length_m=6.0,
        vehicle_width_m=3.0,
        collision_margin_m=required,
        bev_resolution_m=0.25,
        bev_extent_m=(20.0, 20.0, 20.0, 20.0),
        occupancy_channel=0,
    )


def _loss(**kwargs) -> SweptFootprintLoss:
    defaults = dict(
        extra_margin_m=0.0,
        sample_spacing_m=0.5,
        max_swept_substeps=4,
        out_of_bounds_weight=0.0,
        mode="tiered_clearance",
    )
    defaults.update(kwargs)
    return SweptFootprintLoss(_geometry(), **defaults)


def _bev_and_field(clearance_value: float, size: int = 160):
    """构造常量净空场与空 occupancy，便于精确控制被采样到的净空值。"""
    bev = torch.zeros((1, 1, size, size), dtype=torch.float32)
    field = torch.full((1, 1, size, size), float(clearance_value), dtype=torch.float32)
    return bev, field


def _points(length: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """沿前向的直线轨迹（BEV 前向为 +x），航向 0。"""
    points = torch.zeros((1, length, 3), dtype=torch.float32)
    points[0, :, 0] = torch.arange(length, dtype=torch.float32) * 0.5
    mask = torch.ones((1, length), dtype=torch.float32)
    return points, mask


class TestTieredClearanceMode(unittest.TestCase):
    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            SweptFootprintLoss(_geometry(), mode="magic")

    def test_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            _loss(contact_weight=-1.0)
        with self.assertRaises(ValueError):
            _loss(near_weight=-1.0)
        with self.assertRaises(ValueError):
            _loss(aggregation_power=0.5)

    def test_requires_clearance_field(self):
        loss = _loss()
        points, mask = _points()
        bev, _ = _bev_and_field(1.0)
        with self.assertRaises(ValueError):
            loss(bev, points, mask)

    def test_safe_clearance_has_no_penalty(self):
        loss = _loss()
        points, mask = _points()
        bev, field = _bev_and_field(1.0)  # 远大于 required=0.2
        value = float(loss(bev, points, mask, clearance_field=field))
        self.assertAlmostEqual(value, 0.0, places=6)

    def test_near_obstacle_is_penalised_though_no_contact(self):
        """核心性质：净空 0.1m（无接触）必须被罚——二值 occupancy 做不到。"""
        loss = _loss()
        points, mask = _points()
        bev, field = _bev_and_field(0.10)
        value = float(loss(bev, points, mask, clearance_field=field))
        self.assertGreater(value, 0.0)

    def test_penalty_decreases_monotonically_with_clearance(self):
        loss = _loss()
        points, mask = _points()
        values = []
        for clearance in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            bev, field = _bev_and_field(clearance)
            values.append(float(loss(bev, points, mask, clearance_field=field)))
        for earlier, later in zip(values, values[1:]):
            self.assertGreaterEqual(earlier + 1e-9, later)

    def test_contact_penalised_more_than_near_miss(self):
        loss = _loss()
        points, mask = _points()
        bev, contact_field = _bev_and_field(-0.10)  # 车体压入障碍
        bev2, near_field = _bev_and_field(0.10)  # 有间隙但空间不足
        contact = float(loss(bev, points, mask, clearance_field=contact_field))
        near = float(loss(bev2, points, mask, clearance_field=near_field))
        self.assertGreater(contact, near)

    def test_worst_pose_governs_aggregation(self):
        """半条轨迹处于近障带时，聚合值应等于最危险位姿的惩罚（max 语义）。

        构造：把 BEV 前向 x < 5m 的区域净空设为 0.05m（低于 required=0.2），
        其余为 1.0m。轨迹前段落在危险带内，后段安全。
        """
        loss = _loss()
        points, mask = _points(length=20)
        size = 160
        field = torch.full((1, 1, size, size), 1.0, dtype=torch.float32)
        bev = torch.zeros((1, 1, size, size), dtype=torch.float32)
        for row in range(size):
            x = 20.0 - (row + 0.5) * 0.25
            if x < 5.0:
                field[0, 0, row, :] = 0.05
        value = float(loss(bev, points, mask, clearance_field=field))
        # 危险带内单一位姿的惩罚：near_gap = (0.2 - 0.05)/0.25 = 0.6
        expected = 1.0 * 0.6
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, expected, places=4)

    def test_safe_trajectory_aggregates_to_zero(self):
        """全轨迹安全时 max 聚合为 0（不会因平均产生残差）。"""
        loss = _loss()
        points, mask = _points(length=10)
        bev, field = _bev_and_field(1.0)
        self.assertAlmostEqual(
            float(loss(bev, points, mask, clearance_field=field)), 0.0, places=6
        )

    def test_goal_exemption_reduces_penalty_near_goal(self):
        loss = _loss(goal_exempt_radius_m=3.0, goal_exempt_weight=0.1)
        points, mask = _points()
        bev, field = _bev_and_field(0.05)
        # goal 需要 (B,3)，取轨迹末端位姿作为目标
        goal = points[0, -1].clone().unsqueeze(0)
        with_exemption = float(
            loss(bev, points, mask, clearance_field=field, goal=goal)
        )
        bev2, field2 = _bev_and_field(0.05)
        without = float(
            _loss(goal_exempt_radius_m=0.0)(bev2, points, mask, clearance_field=field2)
        )
        self.assertLess(with_exemption, without)

    def test_metadata_round_trip(self):
        loss = _loss(contact_weight=4.0, near_weight=1.0, aggregation_power=4.0)
        payload = loss.to_metadata()
        for key in ("contact_weight", "near_weight", "aggregation_power", "mode"):
            self.assertIn(key, payload)
        self.assertEqual(payload["mode"], "tiered_clearance")


if __name__ == "__main__":
    unittest.main()
