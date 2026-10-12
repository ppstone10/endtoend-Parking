"""运行时轨迹安全门禁测试。"""

import unittest

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from planner.collision import RectangleFootprintCollisionChecker
from controller import MPCController
from runtime import (
    FootprintTrajectorySafetyChecker,
    SafetyShieldSource,
)
from sim import ParkingEnvironment, RectangleObstacle
from sim import DifferentialDriveModel
from runtime import ClosedLoopEngine, TerminalChecker


class _Source:
    def __init__(self, points):
        self.trajectory = Trajectory(np.asarray(points, dtype=np.float64), dt=0.2)
        self.calls = 0

    def begin(self, start, goal):
        pass

    def next_trajectory(self, state):
        self.calls += 1
        return self.trajectory, 1.0


class _SequenceSource:
    """按调用次序返回不同轨迹，用于构造"先安全、后不安全"的门禁场景。"""

    def __init__(self, trajectories):
        self.trajectories = list(trajectories)
        self.calls = 0

    def begin(self, start, goal):
        self.calls = 0

    def next_trajectory(self, state):
        trajectory = self.trajectories[min(self.calls, len(self.trajectories) - 1)]
        self.calls += 1
        return trajectory, 1.0


class TestSafetyShieldSource(unittest.TestCase):
    def setUp(self):
        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(1.8, 2.2, -0.2, 0.2)],
        )
        footprint = RectangleFootprintCollisionChecker(
            env,
            vehicle_length=1.0,
            vehicle_width=0.5,
            collision_margin=0.0,
            resolution=0.05,
        )
        self.checker = FootprintTrajectorySafetyChecker(footprint)
        self.start = VehicleState(0.0, 0.0, 0.0)
        self.goal = GoalPose(3.0, 0.0, 0.0)

    def test_safe_primary_passes_without_fallback(self):
        primary = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        fallback = _Source([[0.0, -2.0, 0.0], [3.0, -2.0, 0.0]])
        shield = SafetyShieldSource(primary, fallback, self.checker)
        shield.begin(self.start, self.goal)
        trajectory, elapsed = shield.next_trajectory(self.start)
        self.assertIs(trajectory, primary.trajectory)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual(shield.safety_stats()["interventions"], 0)
        self.assertEqual(elapsed, 1.0)

    def test_swept_collision_switches_to_safe_fallback_and_records_reason(self):
        primary = _Source([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        fallback = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        shield = SafetyShieldSource(primary, fallback, self.checker)
        shield.begin(self.start, self.goal)
        trajectory, elapsed = shield.next_trajectory(self.start)
        stats = shield.safety_stats()
        self.assertIs(trajectory, fallback.trajectory)
        self.assertEqual(stats["checks"], 1)
        self.assertEqual(stats["interventions"], 1)
        self.assertEqual(stats["reasons"], {"swept_collision": 1})
        self.assertEqual(elapsed, 2.0)

    def test_unsafe_fallback_degrades_to_minimal_risk_hold(self):
        """主轨迹与回退都不安全时：不终止回合，退化为原地保持（SAFE-SHIELD-003）。"""
        unsafe = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        shield = SafetyShieldSource(_Source(unsafe), _Source(unsafe), self.checker)
        shield.begin(self.start, self.goal)
        trajectory, _ = shield.next_trajectory(self.start)
        stats = shield.safety_stats()

        self.assertTrue(shield.checker.check(self.start, trajectory).safe)
        # 退化结果是"停在当前位姿"的单点轨迹，既不是主轨迹也不是回退轨迹
        self.assertEqual(trajectory.points.shape, (1, 3))
        np.testing.assert_allclose(
            np.asarray(trajectory.points, dtype=np.float64)[0],
            [self.start.x, self.start.y, self.start.yaw],
        )
        self.assertEqual(stats["fallback_failures"], 0)
        self.assertEqual(stats["expert_fallback_rejected"], 1)
        self.assertEqual(stats["minimal_risk_holds"], 1)
        self.assertEqual(stats["minimal_risk_reuses"], 0)
        self.assertEqual(stats["minimal_risk_rate"], 1.0)

    def test_fallback_planner_failure_degrades_instead_of_terminating(self):
        class _BrokenFallback(_Source):
            def next_trajectory(self, state):
                raise ValueError("专家无可行轨迹")

        unsafe = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        shield = SafetyShieldSource(_Source(unsafe), _BrokenFallback(unsafe), self.checker)
        shield.begin(self.start, self.goal)
        trajectory, _ = shield.next_trajectory(self.start)
        stats = shield.safety_stats()
        self.assertEqual(trajectory.points.shape, (1, 3))
        self.assertEqual(stats["fallback_failures"], 0)
        self.assertEqual(stats["expert_fallback_unavailable"], 1)
        self.assertEqual(stats["minimal_risk_holds"], 1)

    def test_unsafe_fallback_reuses_last_verified_safe_tail(self):
        """有已验证安全的参考时，退化优先沿用它的尾段（计划保持），而不是停车。"""
        safe = Trajectory(
            np.asarray([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]], dtype=np.float64), dt=0.2
        )
        unsafe = Trajectory(
            np.asarray([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64), dt=0.2
        )
        shield = SafetyShieldSource(
            _SequenceSource([safe, unsafe]), _Source(unsafe.points), self.checker
        )
        shield.begin(self.start, self.goal)
        first, _ = shield.next_trajectory(self.start)
        self.assertIs(first, safe)
        # 车辆已前进到 (2,-1)：不安全主轨迹 + 不安全回退 → 沿用 safe 的尾段
        tail, _ = shield.next_trajectory(VehicleState(2.0, -1.0, 0.0))
        stats = shield.safety_stats()
        self.assertEqual(stats["minimal_risk_reuses"], 1)
        self.assertEqual(stats["minimal_risk_holds"], 0)
        self.assertEqual(stats["fallback_failures"], 0)
        # 尾段是"离当前位置最近点起"的后缀，因此只剩最后一点
        np.testing.assert_allclose(
            np.asarray(tail.points, dtype=np.float64), [[3.0, -1.0, 0.0]]
        )

    def test_transition_is_safe_accepts_pose_already_inside_required_margin(self):
        """当前位姿已在要求净空层内时，被否决的是状态而不是这次转移。"""
        footprint = RectangleFootprintCollisionChecker(
            ParkingEnvironment(
                world_size=20.0,
                obstacles=[RectangleObstacle(1.8, 2.2, -0.2, 0.2)],
            ),
            vehicle_length=1.0,
            vehicle_width=0.5,
            collision_margin=0.2,
            resolution=0.05,
        )
        checker = FootprintTrajectorySafetyChecker(footprint)
        shield = SafetyShieldSource(_Source([[1.0, 0.0, 0.0]]), _Source([[1.0, 0.0, 0.0]]), checker)
        shield.begin(self.start, self.goal)

        inside = VehicleState(1.2, 0.0, 0.0)
        self.assertEqual(checker.check(inside, Trajectory(np.asarray([[1.2, 0.0, 0.0]]), dt=0.0)).reason,
                         "current_pose_collision")
        self.assertTrue(shield.transition_is_safe(inside, inside))
        # 其它原因（真正由这次转移引入的扫掠碰撞）仍然否决
        self.assertFalse(
            shield.transition_is_safe(VehicleState(0.0, 0.0, 0.0), VehicleState(2.0, 0.0, 0.0))
        )

    def test_transition_guard_switches_before_entering_planner_margin(self):
        primary = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        fallback = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        shield = SafetyShieldSource(primary, fallback, self.checker)
        shield.begin(self.start, self.goal)
        replacement, _ = shield.guard_transition(
            self.start, VehicleState(1.4, 0.0, 0.0)
        )
        stats = shield.safety_stats()
        self.assertIs(replacement, fallback.trajectory)
        self.assertEqual(stats["transition_checks"], 1)
        self.assertEqual(stats["prevented_transitions"], 1)
        self.assertEqual(stats["interventions"], 0)
        self.assertEqual(stats["transition_prevention_rate"], 1.0)
        self.assertEqual(stats["reasons"], {"next_state_swept_collision": 1})

    def test_unavailable_fallback_keeps_episode_alive_instead_of_safety_stop(self):
        """安全层无路可走不再是终结态：回合继续（原地保持）并以任务性失败收尾。"""
        unsafe = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        shield = SafetyShieldSource(_Source(unsafe), _Source(unsafe), self.checker)
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(max_v=1.0, max_omega=1.0),
            mpc=MPCController(dt=0.1, horizon=3, seed=0),
            source=shield,
            terminal=TerminalChecker(),
            max_steps=3,
        )
        result = engine.run(self.start, self.goal)
        stats = result.meta["safety_shield"]
        self.assertFalse(result.collision)
        self.assertEqual(result.failure, "timeout")
        self.assertEqual(stats["fallback_failures"], 0)
        self.assertEqual(stats["safety_stops"], 0)
        self.assertGreaterEqual(stats["minimal_risk_holds"], 1)

    def test_primary_runtime_error_is_not_misclassified_as_safety_stop(self):
        class BrokenPrimary(_Source):
            def next_trajectory(self, state):
                raise RuntimeError("model failure")

        safe = [[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]]
        shield = SafetyShieldSource(
            BrokenPrimary(safe),
            _Source(safe),
            self.checker,
        )
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(max_v=1.0, max_omega=1.0),
            mpc=MPCController(dt=0.1, horizon=3, seed=0),
            source=shield,
            terminal=TerminalChecker(),
            max_steps=3,
        )
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            engine.run(self.start, self.goal)


    def test_engine_uses_minimal_risk_when_replaced_transition_is_still_unsafe(self):
        """替换参考的推进仍不安全时，引擎取用最小风险轨迹而非结束回合。"""

        class _StuckSource:
            """始终要求替换，替换后仍然不安全；提供最小风险轨迹后应当收敛。"""

            def __init__(self):
                self.minimal_risk_calls = 0

            def begin(self, start, goal):
                self.minimal_risk_calls = 0

            def next_trajectory(self, state):
                return Trajectory(
                    np.asarray([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
                    dt=0.2,
                ), 0.0

            def guard_transition(self, state, proposed_state):
                return Trajectory(
                    np.asarray([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
                    dt=0.2,
                ), 0.0

            def transition_is_safe(self, state, proposed_state):
                return False

            def minimal_risk_trajectory(self, state):
                self.minimal_risk_calls += 1
                return Trajectory(
                    np.asarray([[state.x, state.y, state.yaw]], dtype=np.float64),
                    dt=0.1,
                )

            def record_safety_stop(self):
                self.safety_stops = getattr(self, "safety_stops", 0) + 1

        source = _StuckSource()
        source.safety_stops = 0
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(max_v=1.0, max_omega=1.0),
            mpc=MPCController(dt=0.1, horizon=3, seed=0),
            source=source,
            terminal=TerminalChecker(),
            max_steps=2,
        )
        result = engine.run(self.start, self.goal)
        # 这个源连保持轨迹都判不安全，因此最终仍安全停止——但引擎确实先取用了
        # 最小风险轨迹，说明退化阶梯被走完，而不是在替换失败处直接放弃。
        self.assertGreaterEqual(source.minimal_risk_calls, 1)
        self.assertEqual(source.safety_stops, 1)
        self.assertEqual(result.failure, "safety_stop")
        self.assertFalse(result.collision)


if __name__ == "__main__":
    unittest.main()
