"""闭环执行器测试：MPC+车辆模型生产路径与理想执行路径。"""

import unittest

import numpy as np

from controller import MPCController
from interfaces import ControlCmd, GoalPose, Trajectory, VehicleState
from runtime import ExecutionError, IdealPathExecutor, MpcVehicleExecutor
from sim import DifferentialDriveModel


def _straight_trajectory(length: float = 5.0, n: int = 51, dt: float = 0.2) -> Trajectory:
    xs = np.linspace(0.0, length, n)
    pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
    return Trajectory(points=pts, dt=dt)


class TestMpcVehicleExecutor(unittest.TestCase):
    def setUp(self):
        self.mpc = MPCController(dt=0.1, horizon=10, seed=0)
        self.model = DifferentialDriveModel(max_v=2.0, max_omega=1.0)
        self.executor = MpcVehicleExecutor(self.mpc, self.model)

    def test_advances_along_trajectory(self):
        trajectory = _straight_trajectory()
        state = VehicleState(0.0, 0.0, 0.0)
        for _ in range(50):
            state = self.executor.propose(state, trajectory, 0.1)
        self.assertGreater(state.x, 2.0)
        self.assertLess(abs(state.y), 0.3)

    def test_last_cmd_is_the_command_used(self):
        """记录指令必须等于本次推进实际使用的 MPC 指令（CEM 有随机性，
        故用固定指令桩验证透传，而不是重复调用真实 MPC 比较）。"""
        expected = ControlCmd(v=0.7, omega=-0.2)

        class _StubMpc:
            def reset(self):
                pass

            def compute(self, trajectory, state):
                return expected

        executor = MpcVehicleExecutor(_StubMpc(), self.model)
        state = VehicleState(0.0, 0.0, 0.0)
        advanced = executor.propose(state, _straight_trajectory(), 0.1)
        self.assertAlmostEqual(executor.last_cmd.v, expected.v)
        self.assertAlmostEqual(executor.last_cmd.omega, expected.omega)
        self.assertAlmostEqual(advanced.x, expected.v * 0.1, places=9)
        self.assertAlmostEqual(advanced.yaw, expected.omega * 0.1, places=9)

    def test_reset_clears_last_cmd(self):
        trajectory = _straight_trajectory()
        self.executor.propose(VehicleState(0.0, 0.0, 0.0), trajectory, 0.1)
        self.executor.reset()
        self.assertEqual(self.executor.last_cmd.v, 0.0)
        self.assertEqual(self.executor.last_cmd.omega, 0.0)


class TestIdealPathExecutor(unittest.TestCase):
    def test_follows_arc_length_exactly(self):
        trajectory = _straight_trajectory(length=5.0, n=51, dt=0.2)
        executor = IdealPathExecutor(control_steps_per_point=2.0)
        state = VehicleState(0.0, 0.0, 0.0)
        executor.begin_trajectory(trajectory)
        for _ in range(20):
            state = executor.propose(state, trajectory, 0.1)
        self.assertAlmostEqual(state.y, 0.0, places=9)
        # 点距 0.1m、每点 2 个控制周期 → 每周期 0.05m，20 步 ≈ 1.0m。
        self.assertAlmostEqual(state.x, 1.0, delta=0.05)
        audit = executor.audit()
        self.assertEqual(audit["kind"], "ideal_path")
        self.assertAlmostEqual(audit["executed_distance_m"], 1.0, delta=0.05)
        self.assertEqual(audit["advance_failures"], 0)

    def test_single_step_per_point_matches_point_spacing(self):
        trajectory = _straight_trajectory(length=5.0, n=51, dt=0.2)
        executor = IdealPathExecutor(control_steps_per_point=1.0)
        state = VehicleState(0.0, 0.0, 0.0)
        executor.begin_trajectory(trajectory)
        for _ in range(10):
            state = executor.propose(state, trajectory, 0.1)
        self.assertAlmostEqual(state.x, 1.0, delta=0.05)

    def test_headings_interpolated_on_turn(self):
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, np.pi / 2.0],
            ]
        )
        executor = IdealPathExecutor(control_steps_per_point=1.0)
        executor.begin_trajectory(Trajectory(points=pts, dt=0.2))
        state = VehicleState(0.0, 0.0, 0.0)
        states = [executor.propose(state, Trajectory(points=pts, dt=0.2), 0.1) for _ in range(4)]
        self.assertAlmostEqual(states[0].x, 1.0, delta=1e-6)
        self.assertAlmostEqual(states[1].y, 1.0, delta=1e-6)
        # Trajectory 内部把点存为 float32，航向精度上限约 1e-7 rad。
        self.assertAlmostEqual(states[1].yaw, np.pi / 2.0, delta=1e-6)
        # 到达轨迹末端后保持不动，不再前进。
        self.assertAlmostEqual(states[3].x, 1.0, delta=1e-6)

    def test_begin_trajectory_resets_cursor(self):
        trajectory = _straight_trajectory(length=5.0, n=51)
        executor = IdealPathExecutor(control_steps_per_point=1.0)
        state = VehicleState(0.0, 0.0, 0.0)
        executor.begin_trajectory(trajectory)
        for _ in range(10):
            state = executor.propose(state, trajectory, 0.1)
        self.assertGreater(state.x, 0.5)
        executor.begin_trajectory(trajectory)
        state = executor.propose(state, trajectory, 0.1)
        self.assertAlmostEqual(state.x, 0.1, delta=0.02)

    def test_single_point_trajectory_rejected(self):
        executor = IdealPathExecutor()
        with self.assertRaises(ExecutionError):
            executor.begin_trajectory(Trajectory(points=np.array([[0.0, 0.0, 0.0]]), dt=0.2))

    def test_zero_steps_per_point_rejected(self):
        with self.assertRaises(ValueError):
            IdealPathExecutor(control_steps_per_point=0.0)


if __name__ == "__main__":
    unittest.main()
