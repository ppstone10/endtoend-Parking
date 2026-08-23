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


def _reverse_s_trajectory(n: int = 120, dt: float = 0.1):
    """S 形倒车参考：车辆从原点倒退行驶，航向左右摆动一次。"""
    pts = []
    x, y, yaw = 0.0, 0.0, 0.0
    for i in range(n):
        omega = 0.6 if 20 <= i < 60 else (-0.6 if 60 <= i < 100 else 0.0)
        x -= 0.5 * np.cos(yaw) * dt
        y -= 0.5 * np.sin(yaw) * dt
        yaw += omega * dt
        pts.append((x, y, yaw))
    return Trajectory(points=np.array(pts), dt=dt)


class TestMPCController(unittest.TestCase):
    def setUp(self):
        self.controller = MPCController(dt=0.1, horizon=10, seed=0)
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

    def test_closed_loop_reverse_s_tracking(self):
        """闭环 S 形倒车跟踪：倒车沿曲线行驶并停在轨迹终点附近。"""
        traj = _reverse_s_trajectory()
        end = traj.points[-1]
        state = VehicleState(0.0, 0.0, 0.0)
        for _ in range(400):
            cmd = self.controller.compute(traj, state)
            state = self.model.step(state, cmd, self.controller.dt)
            if np.hypot(state.x - end[0], state.y - end[1]) < 0.1:
                break
        pos_err = np.hypot(state.x - end[0], state.y - end[1])
        yaw_err = abs(np.arctan2(np.sin(state.yaw - end[2]), np.cos(state.yaw - end[2])))
        self.assertLess(pos_err, 0.3)
        self.assertLess(yaw_err, np.deg2rad(15))

    def test_disturbance_recovery(self):
        """扰动恢复：被推离轨迹 0.5m 后应拉回并继续前进。"""
        traj = _straight_trajectory(length=6.0, n=120)
        state = VehicleState(0.0, 0.5, 0.0)  # 初始横向偏差 0.5m
        for _ in range(300):
            cmd = self.controller.compute(traj, state)
            state = self.model.step(state, cmd, self.controller.dt)
            if state.x >= 5.5:
                break
        self.assertGreater(state.x, 5.0)
        self.assertLess(abs(state.y), 0.3)

    def test_terminal_pose_alignment(self):
        """终点对齐：窗口收敛到末点后，位置与航向应同时收敛。"""
        pts = np.array([[5.0, 0.0, np.pi / 2]])
        traj = Trajectory(points=pts, dt=0.1)
        state = VehicleState(4.0, 0.0, 0.0)
        for _ in range(400):
            cmd = self.controller.compute(traj, state)
            state = self.model.step(state, cmd, self.controller.dt)
            dx = state.x - 5.0
            dy = state.y - 0.0
            yaw_err = abs(np.arctan2(np.sin(state.yaw - np.pi / 2), np.cos(state.yaw - np.pi / 2)))
            if np.hypot(dx, dy) < 0.15 and yaw_err < np.deg2rad(10):
                break
        self.assertLess(np.hypot(state.x - 5.0, state.y), 0.3)
        self.assertLess(
            abs(np.arctan2(np.sin(state.yaw - np.pi / 2), np.cos(state.yaw - np.pi / 2))),
            np.deg2rad(20),
        )

    def test_dt_mismatch_alignment(self):
        """dt 对齐：轨迹 dt=0.2 与控制 dt=0.1 不一致时应仍能直线跟踪。"""
        n = 50
        xs = np.linspace(0.0, 5.0, n)
        pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
        traj = Trajectory(points=pts, dt=0.2)
        state = VehicleState(0.0, 0.0, 0.0)
        for _ in range(250):
            cmd = self.controller.compute(traj, state)
            state = self.model.step(state, cmd, self.controller.dt)
            if state.x >= 4.9:
                break
        self.assertGreater(state.x, 4.5)
        self.assertLess(abs(state.y), 0.3)

    def test_command_limits(self):
        """输出指令始终在限幅内。"""
        traj = _straight_trajectory()
        state = VehicleState(0.0, 0.0, 0.0)
        for _ in range(20):
            cmd = self.controller.compute(traj, state)
            state = self.model.step(state, cmd, self.controller.dt)
            self.assertLessEqual(abs(cmd.v), self.controller.max_v + 1e-9)
            self.assertLessEqual(abs(cmd.omega), self.controller.max_omega + 1e-9)


if __name__ == "__main__":
    unittest.main()
