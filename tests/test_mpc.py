"""MPC 轨迹跟踪控制器测试。"""

import unittest

import numpy as np

from controller import MPCController
from interfaces import ControlCmd, Trajectory, VehicleState
from sim import DifferentialDriveModel


def _straight_trajectory(length: float = 5.0, n: int = 50, dt: float = 0.1):
    xs = np.linspace(0.0, length, n)
    pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
    return Trajectory(points=pts, dt=dt)


class TestMPCController(unittest.TestCase):
    def setUp(self):
        self.controller = MPCController(dt=0.1, horizon=10)
        self.model = DifferentialDriveModel(max_v=2.0, max_omega=1.0)

    def test_forward_command_on_straight(self):
        traj = _straight_trajectory()
        state = VehicleState(0.0, 0.0, 0.0)
        cmd = self.controller.compute(traj, state)
        self.assertGreater(cmd.v, 0.0)  # 前方轨迹应驱动前进
        self.assertAlmostEqual(cmd.omega, 0.0, delta=0.1)  # 直线接近零转向

    def test_closed_loop_straight_tracking(self):
        """闭环：沿直线轨迹推进，应收敛到轨迹终点附近。"""
        traj = _straight_trajectory(length=5.0, n=100)
        state = VehicleState(0.0, 0.0, 0.0)
        for _ in range(200):
            cmd = self.controller.compute(traj, state)
            state = self.model.step(state, cmd, self.controller.dt)
            if state.x >= 5.0 - 0.1:
                break
        self.assertGreater(state.x, 4.0)
        self.assertLess(abs(state.y), 0.3)

    def test_turning_trajectory(self):
        """转向轨迹：车辆应向右侧偏转。"""
        dt = 0.1
        pts = []
        x, y, yaw = 0.0, 0.0, 0.0
        for _ in range(50):
            x += 0.3 * np.cos(yaw) * dt * 10
            y += 0.3 * np.sin(yaw) * dt * 10
            yaw += 0.1 * dt * 10
            pts.append((x, y, yaw))
        traj = Trajectory(points=np.array(pts), dt=dt)
        state = VehicleState(0.0, 0.0, 0.0)
        cmd = self.controller.compute(traj, state)
        # 曲线朝 y 正方向偏转，omega 应非零且朝该方向。
        self.assertNotAlmostEqual(cmd.omega, 0.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()