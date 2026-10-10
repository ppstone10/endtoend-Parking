"""推理侧轨迹级几何过滤测试。

覆盖四类行为：
1. 可行轨迹必须**逐位不变**地通过（过滤器不得引入无谓改动）；
2. 不可行轨迹被**最小侧向平移**修回可行域，且位移量级与几何缺口同阶；
3. 位移场在相邻位姿间连贯（不允许只移动被挡的那一个点造成折角）；
4. 投影无解时退化为安全截断（停到最后一个可行位姿），绝不放行不可行轨迹；
   连第一个位姿都不可行时原地保持。
"""

import unittest

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from planner.collision import RectangleFootprintCollisionChecker
from runtime import GeometricFilterSource, SweptFootprintProjector
from sim import ParkingEnvironment, RectangleObstacle


class _Source:
    """固定轨迹源，用于验证包装源的过滤与统计。"""

    def __init__(self, points):
        self.trajectory = Trajectory(
            np.asarray(points, dtype=np.float64), dt=0.2
        )
        self.calls = 0

    def begin(self, start, goal):
        pass

    def next_trajectory(self, state):
        self.calls += 1
        return self.trajectory, 1.0


class TestGeometricFilterSource(unittest.TestCase):
    def setUp(self):
        self.env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(1.8, 2.2, -0.2, 0.2)],
        )
        self.projector = SweptFootprintProjector(_free_space(self.env))
        self.start = VehicleState(0.0, 0.0, 0.0)
        self.goal = GoalPose(3.0, 0.0, 0.0)

    def test_feasible_trajectory_is_passed_through_untouched(self):
        primary = _Source([[0.5, 1.5, 0.0], [1.0, 1.5, 0.0], [1.5, 1.5, 0.0]])
        source = GeometricFilterSource(primary, self.projector)
        source.begin(self.start, self.goal)
        trajectory, elapsed = source.next_trajectory(self.start)
        self.assertIs(trajectory, primary.trajectory)
        self.assertEqual(elapsed, 1.0)
        stats = source.filter_stats()
        self.assertEqual(stats["checks"], 1)
        self.assertEqual(stats["unmodified"], 1)
        self.assertEqual(stats["intervention_rate"], 0.0)

    def test_blocked_trajectory_is_repaired_and_counted(self):
        primary = _Source([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        source = GeometricFilterSource(primary, self.projector)
        source.begin(self.start, self.goal)
        trajectory, _ = source.next_trajectory(self.start)
        self.assertIsNot(trajectory, primary.trajectory)
        self.assertTrue(
            self.projector.is_feasible(
                np.asarray([self.start.x, self.start.y, self.start.yaw]),
                np.asarray(trajectory.points, dtype=np.float64),
            )
        )
        stats = source.filter_stats()
        self.assertEqual(stats["repaired"], 1)
        self.assertEqual(stats["intervention_rate"], 1.0)
        self.assertGreater(stats["max_offset_m"], 0.0)

    def test_stats_reset_per_episode(self):
        primary = _Source([[0.5, 1.5, 0.0], [1.0, 1.5, 0.0], [1.5, 1.5, 0.0]])
        source = GeometricFilterSource(primary, self.projector)
        source.begin(self.start, self.goal)
        source.next_trajectory(self.start)
        source.begin(self.start, self.goal)
        self.assertEqual(source.filter_stats()["checks"], 0)


class _SequenceSource:
    """按调用次序返回不同轨迹，用于验证"修不好时保持上一条可行参考"。"""

    def __init__(self, trajectories):
        self.trajectories = list(trajectories)
        self.index = 0

    def begin(self, start, goal):
        self.index = 0

    def next_trajectory(self, state):
        trajectory = self.trajectories[min(self.index, len(self.trajectories) - 1)]
        self.index += 1
        return trajectory, 1.0


class TestPlanPersistence(unittest.TestCase):
    def setUp(self):
        # 一道横墙：一旦预测轨迹指向墙里，侧向平移无解。
        self.env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(2.0, 3.0, -5.0, 5.0)],
        )
        self.projector = SweptFootprintProjector(_free_space(self.env))
        self.start = VehicleState(0.0, 0.0, 0.0)
        self.goal = GoalPose(3.0, 0.0, 0.0)

    def test_unrepairable_prediction_keeps_following_last_feasible_plan(self):
        feasible = Trajectory(
            np.asarray([[0.5, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64), dt=0.2
        )
        hopeless = Trajectory(
            np.asarray(
                [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            dt=0.2,
        )
        source = GeometricFilterSource(
            _SequenceSource([feasible, hopeless]), self.projector
        )
        source.begin(self.start, self.goal)
        first, _ = source.next_trajectory(self.start)
        np.testing.assert_allclose(first.points, feasible.points)
        second, _ = source.next_trajectory(self.start)
        # 没有退化成"原地停车"的截断，而是继续跟随上一条可行参考。
        self.assertGreaterEqual(second.points.shape[0], 2)
        self.assertTrue(
            self.projector.is_feasible(
                np.asarray([self.start.x, self.start.y, self.start.yaw]),
                np.asarray(second.points, dtype=np.float64),
            )
        )
        self.assertEqual(source.filter_stats()["persisted"], 1)
        self.assertEqual(source.filter_stats()["truncated"], 0)

    def test_without_history_unrepairable_prediction_truncates(self):
        hopeless = Trajectory(
            np.asarray(
                [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            dt=0.2,
        )
        source = GeometricFilterSource(_Source(hopeless.points), self.projector)
        source.begin(self.start, self.goal)
        trajectory, _ = source.next_trajectory(self.start)
        self.assertLess(trajectory.points.shape[0], hopeless.points.shape[0])
        stats = source.filter_stats()
        self.assertEqual(stats["persisted"], 0)
        self.assertEqual(stats["truncated"], 1)

    def test_unusable_prediction_requests_safety_stop(self):
        from runtime import SafetyStopError

        broken = _Source([[0.5, 1.0, 0.0]])
        broken.trajectory = Trajectory(
            np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float64), dt=0.2
        )
        source = GeometricFilterSource(broken, self.projector)
        source.begin(self.start, self.goal)
        with self.assertRaises(SafetyStopError):
            source.next_trajectory(self.start)
        # 引擎在 SafetyStopError 路径上会回调 record_safety_stop，源必须实现它。
        source.record_safety_stop()
        stats = source.filter_stats()
        self.assertEqual(stats["reasons"], {"unusable_prediction": 1})
        self.assertEqual(stats["safety_stops"], 1)


def _free_space(env, margin=0.0, resolution=0.05):
    return RectangleFootprintCollisionChecker(
        env,
        vehicle_length=1.0,
        vehicle_width=0.5,
        collision_margin=margin,
        resolution=resolution,
    )


def _straight(xs, y=0.0, yaw=0.0):
    return np.asarray([[x, y, yaw] for x in xs], dtype=np.float64)


class TestSweptFootprintProjector(unittest.TestCase):
    def setUp(self):
        # 障碍在 y 正方向，占 [1.8, 2.2] × [-0.2, 0.2]，不足以拦住 0.5m 宽的车。
        self.env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(1.8, 2.2, 3.2, 3.6)],
        )
        self.start = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    def test_feasible_trajectory_passes_through_bit_identical(self):
        projector = SweptFootprintProjector(_free_space(self.env))
        points = _straight([0.5, 1.0, 1.5, 2.0, 2.5])
        outcome = projector.repair(self.start, points)
        self.assertFalse(outcome.modified)
        self.assertFalse(outcome.truncated)
        self.assertEqual(outcome.blocked_poses, 0)
        np.testing.assert_array_equal(outcome.points, points)

    def test_blocked_trajectory_is_pushed_sideways_into_feasible_set(self):
        # 车辆沿 y=3.4 直行会从障碍 [3.2,3.6] 正中穿过，须侧移 ~0.35m 以上。
        projector = SweptFootprintProjector(_free_space(self.env))
        points = _straight([0.5, 1.0, 1.5, 2.0, 2.5], y=3.4)
        outcome = projector.repair(self.start + np.asarray([0.0, 3.4, 0.0]), points)
        self.assertGreater(outcome.blocked_poses, 0)
        self.assertFalse(outcome.truncated)
        self.assertEqual(outcome.residual_blocked, 0)
        self.assertTrue(outcome.modified)
        # 位移必须是侧向（y 方向），且量级与几何缺口同阶而非整条重排。
        shift = outcome.points[:, 1] - points[:, 1]
        self.assertGreater(shift.min(), 0.0)
        self.assertLess(abs(outcome.max_offset_m), 1.0)
        self.assertGreater(shift.max() - shift.min(), -1e-9)  # 单调外扩，无反向折角

    def test_offsets_respect_reachable_slope_limit(self):
        """相邻位姿的侧移差必须受可达锥约束（不允许单点硬折）。"""
        projector = SweptFootprintProjector(
            _free_space(self.env), max_lateral_slope=0.5
        )
        start = self.start + np.asarray([0.0, 3.4, 0.0])
        points = _straight([0.5, 1.0, 1.5, 2.0, 2.5], y=3.4)
        outcome = projector.repair(start, points)
        self.assertFalse(outcome.truncated)
        shift = outcome.points[:, 1] - points[:, 1]
        arcs = np.concatenate([[np.hypot(0.5, 0.0)], np.full(4, 0.5)])
        limit = 0.5 * arcs[1:]
        # 可达锥是迭代投影，允许 1% 的收敛残差。
        self.assertLessEqual(np.abs(np.diff(shift)).max(), limit.max() * 1.01 + 1e-6)
        # 需要侧移的点把上游一起抬起来（越早越少），而不是只挪一个点。
        self.assertGreater(shift[0], 0.0)

    def test_unreachable_projection_truncates_to_safe_prefix(self):
        # 障碍是一道横墙：侧向平移无法绕开，必须截断在墙前。
        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(2.0, 3.0, -5.0, 5.0)],
        )
        projector = SweptFootprintProjector(_free_space(env))
        points = _straight([0.5, 1.0, 1.5, 2.0, 2.5])
        outcome = projector.repair(self.start, points)
        self.assertTrue(outcome.truncated)
        self.assertFalse(outcome.held)
        self.assertIsNotNone(outcome.reason)
        # 截断结果必须仍然可行，且短于原轨迹。
        self.assertLess(outcome.points.shape[0], points.shape[0])
        self.assertGreater(outcome.points.shape[0], 0)
        self.assertTrue(
            projector.is_feasible(self.start, outcome.points)
            or outcome.points.shape[0] == 1
        )

    def test_hopeless_first_pose_holds_instead_of_releasing(self):
        # 车辆起步位姿就在障碍里：任何前向点都不可达，只能原地保持。
        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(-5.0, 5.0, -5.0, 5.0)],
        )
        projector = SweptFootprintProjector(_free_space(env))
        points = _straight([0.5, 1.0, 1.5, 2.0])
        outcome = projector.repair(self.start, points)
        self.assertTrue(outcome.held)
        self.assertEqual(outcome.points.shape, (1, 3))
        np.testing.assert_allclose(outcome.points[0], self.start)

    def test_start_pose_inside_required_margin_may_still_move_out(self):
        """车辆已落在要求净空层内时，只要求它不再真正接触，并从首个预测点恢复净空。"""
        required = _free_space(self.env, margin=0.6)
        contact = _free_space(self.env, margin=0.0)
        projector = SweptFootprintProjector(required, contact=contact)
        # 起点 y=3.0 落在 0.6m 膨胀层内（接触自由），首个预测点起要求 0.6m 净空。
        start = np.asarray([0.0, 3.0, 0.0], dtype=np.float64)
        points = np.asarray(
            [[0.5, 3.0, 0.0], [1.0, 3.0, 0.0], [1.5, 3.0, 0.0]], dtype=np.float64
        )
        outcome = projector.repair(start, points)
        self.assertFalse(outcome.held)
        self.assertEqual(outcome.residual_blocked, 0)
        self.assertGreater(outcome.max_offset_m, 0.0)

    def test_parameter_validation(self):
        space = _free_space(self.env)
        with self.assertRaises(ValueError):
            SweptFootprintProjector(space, max_offset_m=0.0)
        with self.assertRaises(ValueError):
            SweptFootprintProjector(space, bisection_steps=0)
        with self.assertRaises(ValueError):
            SweptFootprintProjector(space, max_rounds=0)
        with self.assertRaises(ValueError):
            SweptFootprintProjector(space, max_lateral_slope=0.0)
        projector = SweptFootprintProjector(space)
        with self.assertRaises(ValueError):
            projector.repair(self.start, np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            projector.repair(self.start, np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            projector.repair(self.start, np.asarray([[np.nan, 0.0, 0.0]]))

    def test_feasibility_helper_matches_repair_decision(self):
        projector = SweptFootprintProjector(_free_space(self.env))
        free = _straight([0.5, 1.0, 1.5])
        blocked = _straight([0.5, 1.0, 1.5], y=3.4)
        self.assertTrue(projector.is_feasible(self.start, free))
        self.assertFalse(
            projector.is_feasible(self.start + np.asarray([0.0, 3.4, 0.0]), blocked)
        )


if __name__ == "__main__":
    unittest.main()
