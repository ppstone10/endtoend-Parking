"""车辆运动模型独立验收：把"执行层"从控制与规划里单独拆出来验证。

E1/E3 只证明"MPC + 车辆模型"这个组合在闭环里可用，不能单独说明车辆模型本身
是否按差分驱动语义推进。本测试直接给定控制序列，用解析解核对状态推进：

    x' = x + v·cos(yaw)·dt,  y' = y + v·sin(yaw)·dt,  yaw' = yaw + ω·dt

并覆盖限幅、航向环绕、零速度保持与长时积分漂移四项边界。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from interfaces import ControlCmd, VehicleState
from sim import DifferentialDriveModel

DT = 0.1


def _analytic_step(state: VehicleState, cmd: ControlCmd, dt: float) -> tuple[float, float, float]:
    """与模型约定同构的解析推进（不做限幅，供未越界用例比对）。"""
    return (
        state.x + cmd.v * math.cos(state.yaw) * dt,
        state.y + cmd.v * math.sin(state.yaw) * dt,
        state.yaw + cmd.omega * dt,
    )


class TestDifferentialDriveModelAcceptance(unittest.TestCase):
    def setUp(self):
        self.model = DifferentialDriveModel(max_v=2.0, max_omega=1.0)

    def test_straight_line_advance(self):
        state = VehicleState(0.0, 0.0, 0.0)
        advanced = self.model.step(state, ControlCmd(v=1.0, omega=0.0), DT)
        self.assertAlmostEqual(advanced.x, 0.1, places=9)
        self.assertAlmostEqual(advanced.y, 0.0, places=9)
        self.assertAlmostEqual(advanced.yaw, 0.0, places=9)
        self.assertAlmostEqual(advanced.v, 1.0)
        self.assertAlmostEqual(advanced.omega, 0.0)

    def test_pure_rotation_does_not_translate(self):
        state = VehicleState(3.0, -2.0, 0.7)
        advanced = self.model.step(state, ControlCmd(v=0.0, omega=0.8), DT)
        self.assertAlmostEqual(advanced.x, 3.0, places=9)
        self.assertAlmostEqual(advanced.y, -2.0, places=9)
        self.assertAlmostEqual(advanced.yaw, 0.7 + 0.08, places=9)

    def test_matches_analytic_solution_over_random_sequence(self):
        rng = np.random.default_rng(7)
        state = VehicleState(0.0, 0.0, 0.0)
        reference = (0.0, 0.0, 0.0)
        for _ in range(200):
            cmd = ControlCmd(
                v=float(rng.uniform(-2.0, 2.0)), omega=float(rng.uniform(-1.0, 1.0))
            )
            state = self.model.step(state, cmd, DT)
            reference = _analytic_step(
                VehicleState(*reference), cmd, DT
            )
            self.assertAlmostEqual(state.x, reference[0], places=9)
            self.assertAlmostEqual(state.y, reference[1], places=9)
            self.assertAlmostEqual(state.yaw, reference[2], places=9)

    def test_clamps_commands_to_configured_limits(self):
        state = VehicleState(0.0, 0.0, 0.0)
        advanced = self.model.step(state, ControlCmd(v=99.0, omega=-99.0), DT)
        self.assertAlmostEqual(advanced.v, 2.0)
        self.assertAlmostEqual(advanced.omega, -1.0)
        self.assertAlmostEqual(advanced.x, 0.2, places=9)

    def test_negative_speed_moves_backwards(self):
        state = VehicleState(0.0, 0.0, 0.0)
        advanced = self.model.step(state, ControlCmd(v=-1.0, omega=0.0), DT)
        self.assertAlmostEqual(advanced.x, -0.1, places=9)

    def test_yaw_is_not_wrapped_but_stays_consistent(self):
        """模型不做航向环绕；调用方用 sin/cos 语义，故只需保证连续性。"""
        state = VehicleState(0.0, 0.0, math.pi - 0.01)
        advanced = self.model.step(state, ControlCmd(v=0.0, omega=1.0), DT)
        self.assertGreater(advanced.yaw, math.pi)
        delta = advanced.yaw - (math.pi - 0.01)
        self.assertAlmostEqual(delta, 0.1, places=9)

    def test_zero_speed_holds_position(self):
        state = VehicleState(1.0, 2.0, 0.5)
        for _ in range(50):
            state = self.model.step(state, ControlCmd(v=0.0, omega=0.0), DT)
        self.assertAlmostEqual(state.x, 1.0, places=9)
        self.assertAlmostEqual(state.y, 2.0, places=9)
        self.assertAlmostEqual(state.yaw, 0.5, places=9)

    def test_long_run_straight_line_has_no_integration_drift(self):
        """500 步匀速直线：终态应与解析解一致（欧拉积分对常速直线是精确的）。"""
        state = VehicleState(0.0, 0.0, 0.0)
        cmd = ControlCmd(v=1.5, omega=0.0)
        for _ in range(500):
            state = self.model.step(state, cmd, DT)
        self.assertAlmostEqual(state.x, 1.5 * 500 * DT, places=9)
        self.assertAlmostEqual(state.y, 0.0, places=9)
        self.assertAlmostEqual(state.yaw, 0.0, places=9)

    def test_circle_radius_matches_analytic_radius(self):
        """恒速恒角速度：轨迹应落在半径 v/ω 的圆周上（欧拉积分的已知偏差在容差内）。"""
        v, omega = 1.0, 0.5
        state = VehicleState(0.0, 0.0, 0.0)
        positions = [(state.x, state.y)]
        for _ in range(200):
            state = self.model.step(state, ControlCmd(v=v, omega=omega), DT)
            positions.append((state.x, state.y))
        points = np.asarray(positions)
        # 圆的解析圆心在 (0, v/ω) 一侧。
        center = np.array([0.0, v / omega])
        radius = np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])
        self.assertAlmostEqual(float(radius.mean()), v / omega, delta=0.05)
        # 欧拉积分使半径单调略增，但 200 步内不应超过 5% 偏差。
        self.assertLess(float(radius.max() - radius.min()) / (v / omega), 0.05)


if __name__ == "__main__":
    unittest.main()
