"""跨重规划一致性迟滞（ConsistentPlanSource）测试。

用途：验证 `--consistency-hold-limit` 的机理实现正确（方向相反即保持上一条计划、
达到上限即放行、无明确方向不计翻转、统计口径正确）。

注意：该组件经 2026-10-13 实测**不改善闭环成绩**（抖动 18.7%→14.6%、振荡失败
62→47，但碰撞 58.7%→68.7%、成功仍 0%），仅作为诊断能力保留，默认关闭。
"""

from __future__ import annotations

import unittest

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from runtime.sources import ConsistentPlanSource


class _StubSource:
    """按顺序吐出预定计划的替身源（世界系，与 NetworkSource 的输出约定一致）。"""

    def __init__(self, plans: list[list[list[float]]]) -> None:
        self.plans = plans
        self.calls = 0

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self.calls = 0

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        plan = self.plans[min(self.calls, len(self.plans) - 1)]
        self.calls += 1
        return Trajectory(np.asarray(plan, dtype=np.float64), dt=0.2), 1.0


def _approach() -> list[list[float]]:
    """朝目标（x=10）前进的计划。"""
    return [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]


def _recede() -> list[list[float]]:
    """背离目标（x 变负）的计划。"""
    return [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [-6.0, 0.0, 0.0]]


class TestConsistentPlanSource(unittest.TestCase):
    def setUp(self):
        self.goal = GoalPose(10.0, 0.0, 0.0)
        self.start = VehicleState(0.0, 0.0, 0.0)

    def _run(self, plans, positions, hold_limit=3):
        source = ConsistentPlanSource(_StubSource(plans), hold_limit=hold_limit)
        source.begin(self.start, self.goal)
        outputs = [
            source.next_trajectory(VehicleState(x, 0.0, 0.0))[0].points
            for x in positions
        ]
        return source, outputs

    def test_first_plan_is_passed_through(self):
        source, outputs = self._run([_approach()], [0.0])
        self.assertAlmostEqual(float(outputs[0][-1][0]), 6.0)
        self.assertEqual(source.consistency_stats()["held"], 0)

    def test_same_direction_plan_is_accepted(self):
        source, outputs = self._run([_approach(), _approach()], [0.0, 0.5])
        self.assertEqual(source.consistency_stats()["direction_flips"], 0)
        self.assertEqual(source.consistency_stats()["held"], 0)

    def test_opposite_direction_holds_previous_plan(self):
        source, outputs = self._run([_approach(), _recede()], [0.0, 0.5])
        stats = source.consistency_stats()
        self.assertEqual(stats["direction_flips"], 1)
        self.assertEqual(stats["held"], 1)
        # 保持的是上一条计划（朝目标），不是新的背离计划
        self.assertGreater(float(outputs[1][-1][0]), 0.0)

    def test_hold_is_released_after_limit(self):
        source, _ = self._run(
            [_approach()] + [_recede()] * 4,
            [0.0, 0.5, 1.0, 1.5, 2.0],
            hold_limit=2,
        )
        stats = source.consistency_stats()
        # 第 2、3 次调用方向相反且在保持上限内 → 保持；第 4 次仍相反但已达上限 → 放行；
        # 第 5 次新计划与已放行的上一条同向 → 不构成翻转。
        self.assertEqual(stats["checks"], 5)
        self.assertEqual(stats["direction_flips"], 3)
        self.assertEqual(stats["held"], 2)
        self.assertEqual(stats["released"], 1)

    def test_hold_limit_zero_never_holds(self):
        source, _ = self._run([_approach(), _recede()], [0.0, 0.5], hold_limit=0)
        stats = source.consistency_stats()
        self.assertEqual(stats["held"], 0)
        self.assertEqual(stats["released"], 1)

    def test_negligible_progress_is_not_a_flip(self):
        # 第二条计划基本原地（净变化 < 0.02m 死区），不应被计为方向翻转
        flat = [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.008, 0.0, 0.0]]
        source, _ = self._run([_approach(), flat], [0.0, 0.5])
        stats = source.consistency_stats()
        self.assertEqual(stats["direction_flips"], 0)
        self.assertEqual(stats["held"], 0)

    def test_exhausted_previous_plan_is_not_held(self):
        # 车辆已停在旧计划末端：剩余段不足 3 点，必须放行新计划而不是空转
        source, _ = self._run([_approach(), _recede()], [0.0, 6.0])
        stats = source.consistency_stats()
        self.assertEqual(stats["held"], 0)

    def test_rejects_negative_hold_limit(self):
        with self.assertRaisesRegex(ValueError, "hold_limit"):
            ConsistentPlanSource(_StubSource([_approach()]), hold_limit=-1)


if __name__ == "__main__":
    unittest.main()
