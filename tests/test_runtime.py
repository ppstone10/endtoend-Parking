"""滚动闭环引擎测试。"""

import unittest

import numpy as np

from controller import MPCController
from interfaces import ControlCmd, GoalPose, Trajectory, VehicleState
from runtime import ClosedLoopEngine, ExpertSource, TerminalChecker
from runtime.termination import classify_oscillation
from sim import DifferentialDriveModel, ParkingEnvironment, RectangleObstacle
from planner import HybridAStarPlanner


class _StraightSource:
    """直线轨迹源（测试用）：始终返回起点到目标的直线。"""

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self.goal = goal
        n = 50
        xs = np.linspace(start.x, goal.x, n)
        ys = np.linspace(start.y, goal.y, n)
        yaws = np.linspace(start.yaw, goal.yaw, n)
        self.traj = Trajectory(points=np.stack([xs, ys, yaws], axis=1), dt=0.1)

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        return self.traj, 1.0


class _StubModel:
    """固定输出直行轨迹的桩网络（测试 NetworkSource 管线）。"""

    def __init__(self) -> None:
        self.dt = 0.2

    def predict(self, bev, goal, state):
        n = 10
        xs = np.linspace(0.0, 2.0, n)
        pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
        return Trajectory(points=pts, dt=self.dt)


class TestTerminalChecker(unittest.TestCase):
    def test_reached_dual_threshold(self):
        checker = TerminalChecker(tol_pos=0.3, tol_yaw=np.deg2rad(10.0))
        goal = GoalPose(5.0, 0.0, 0.0)
        ok = VehicleState(5.2, 0.1, np.deg2rad(5.0))
        bad_pos = VehicleState(5.5, 0.1, 0.0)
        bad_yaw = VehicleState(5.1, 0.0, np.deg2rad(20.0))
        self.assertTrue(checker.reached(ok, goal))
        self.assertFalse(checker.reached(bad_pos, goal))
        self.assertFalse(checker.reached(bad_yaw, goal))

    def test_yaw_wraps(self):
        checker = TerminalChecker()
        goal = GoalPose(0.0, 0.0, np.pi - 0.05)
        state = VehicleState(0.0, 0.0, -np.pi + 0.05)  # 跨 ±pi 同一方向
        self.assertTrue(checker.reached(state, goal))


class TestOscillation(unittest.TestCase):
    def test_steady_not_oscillation(self):
        v = np.full(50, 0.5)
        self.assertFalse(classify_oscillation(v))

    def test_chattering_is_oscillation(self):
        v = np.tile([0.5, -0.5], 20)
        self.assertTrue(classify_oscillation(v))

    def test_legit_reversal_not_oscillation(self):
        v = np.concatenate([np.full(30, 0.5), np.full(30, -0.5)])
        self.assertFalse(classify_oscillation(v, ref_flips=1))


class TestClosedLoopEngine(unittest.TestCase):
    def _engine(self, source, env=None, **kwargs):
        mpc = MPCController(dt=0.1, horizon=10, seed=0)
        vehicle = DifferentialDriveModel(max_v=2.0, max_omega=1.0)
        return ClosedLoopEngine(
            vehicle_model=vehicle,
            mpc=mpc,
            source=source,
            env=env,
            max_steps=kwargs.pop("max_steps", 600),
            **kwargs,
        )

    def test_straight_source_success(self):
        source = _StraightSource()
        engine = self._engine(source)
        result = engine.run(VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0))
        self.assertTrue(result.success)
        self.assertIsNone(result.failure)
        self.assertLess(result.final_pos_err, 0.3)
        self.assertLess(result.final_yaw_err, np.deg2rad(10.0))
        self.assertGreater(result.path_length, 4.0)
        self.assertEqual(result.steps, result.record.n_steps)

    def test_timeout_classification(self):
        source = _StraightSource()
        engine = self._engine(source, max_steps=3)
        result = engine.run(VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0))
        self.assertFalse(result.success)
        self.assertEqual(result.failure, "timeout")

    def test_collision_detection(self):
        # 车道尽头是墙：车辆沿直线撞墙应判碰撞。
        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(4.0, 10.0, -5.0, 5.0)],
        )
        source = _StraightSource()
        engine = self._engine(source, env=env)
        result = engine.run(VehicleState(0.0, 0.0, 0.0), GoalPose(6.0, 0.0, 0.0))
        self.assertTrue(result.collision)
        self.assertEqual(result.failure, "collision")

    def test_expert_source_closed_loop(self):
        env = ParkingEnvironment(world_size=40.0)
        planner = HybridAStarPlanner(env=env)
        source = ExpertSource(planner)
        engine = self._engine(source, env=env, vehicle_length=4.0, vehicle_width=2.0)
        result = engine.run(
            VehicleState(-5.0, -5.0, np.pi / 4),
            GoalPose(5.0, 5.0, -np.pi / 4),
        )
        self.assertTrue(result.success, f"failure={result.failure} pos={result.final_pos_err:.2f} yaw={result.final_yaw_err:.2f}")


class TestNetworkSourcePlumbing(unittest.TestCase):
    def test_predict_to_global(self):
        from runtime import NetworkSource
        from dataset import SensorBEVPipeline
        from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
        from sim import SimulatedCamera, SimulatedLiDAR
        from interfaces import CameraIntrinsics

        env = ParkingEnvironment(world_size=40.0)
        intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        )
        pipeline = SensorBEVPipeline(
            lidar_sensor=SimulatedLiDAR(env, beams=360, max_range=20.0),
            camera_sensor=SimulatedCamera(env, intrinsics),
            lidar2bev=LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
            camera2bev=Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
            bev_fusion=BEVFusion(),
        )
        source = NetworkSource(pipeline, _StubModel())
        goal = GoalPose(8.0, 0.0, 0.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), goal)
        self.assertEqual(env.parking_spots, [goal])
        traj, ms = source.next_trajectory(VehicleState(1.0, 2.0, np.pi / 2))
        # 桩网络在局部系输出 +x 方向 2m，车辆 yaw=90° 时全局应为 +y 方向。
        end = traj.points[-1]
        self.assertAlmostEqual(end[0], 1.0, delta=0.05)
        self.assertAlmostEqual(end[1], 4.0, delta=0.05)
        self.assertGreaterEqual(ms, 0.0)


if __name__ == "__main__":
    unittest.main()
