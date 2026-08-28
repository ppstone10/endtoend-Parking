"""Hybrid A* 规划器测试。"""

import unittest

import numpy as np

from interfaces import GoalPose, VehicleState
from dataset.maneuver import audit_maneuver_consistency
from planner import HybridAStarPlanner
from sim import CircleObstacle, MINING_DRILL_RIG, Maneuver, ParkingEnvironment, RectangleObstacle
from sim.scenes import build_scene


def _open_env():
    """无障碍开阔环境，车辆从原点前进到正前方目标。"""
    return ParkingEnvironment(world_size=40.0, obstacles=[])


def _channel_env():
    """通道环境：车辆沿通道前进并转向，验证转向轨迹。"""
    return ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=-6.0, y_max=-2.0),
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=2.0, y_max=6.0),
        ],
    )


class TestHybridAStarPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = HybridAStarPlanner(env=_open_env())

    def test_plan_to_forward_goal(self):
        start = VehicleState(0.0, 0.0, 0.0)
        goal = GoalPose(6.0, 0.0, 0.0)
        traj = self.planner.plan(start, goal)
        self.assertGreater(traj.horizon, 2)
        # 终点应接近目标位姿。
        final = traj.points[-1]
        self.assertLess(np.hypot(final[0] - 6.0, final[1] - 0.0), 0.6)
        # 轨迹全程无碰撞（终点与起点均自由）。
        self.assertGreaterEqual(traj.dt, 0.0)

    def test_plan_endpoint_reaches_goal(self):
        start = VehicleState(0.0, 0.0, 0.0)
        goal = GoalPose(4.0, 4.0, np.pi / 2.0)
        traj = self.planner.plan(start, goal)
        final = traj.points[-1]
        self.assertLess(np.hypot(final[0] - 4.0, final[1] - 4.0), 0.6)

    def test_plan_collision_raises(self):
        env = ParkingEnvironment(
            world_size=40.0,
            obstacles=[RectangleObstacle(x_min=0.0, x_max=20.0, y_min=-20.0, y_max=20.0)],
        )
        planner = HybridAStarPlanner(env=env)
        start = VehicleState(0.0, 0.0, 0.0)
        goal = GoalPose(5.0, 0.0, 0.0)
        with self.assertRaises((ValueError, RuntimeError)):
            planner.plan(start, goal)

    def test_trajectory_points_free(self):
        env = _channel_env()
        planner = HybridAStarPlanner(env=env)
        start = VehicleState(0.0, 0.0, 0.0)
        goal = GoalPose(8.0, 0.0, 0.0)
        traj = planner.plan(start, goal)
        for px, py in traj.points[:, :2]:
            self.assertTrue(env.is_free(float(px), float(py)))

    def test_collision_margin_rejects_tight_goal(self):
        """膨胀裕度语义：贴墙目标无膨胀时自由、膨胀后冲突。

        直接验证 _pose_free 的 C-space 膨胀行为与 plan 入口拒绝路径。
        """
        env = ParkingEnvironment(
            world_size=40.0,
            obstacles=[RectangleObstacle(x_min=-20.0, x_max=20.0, y_min=3.8, y_max=20.0)],
        )
        start = VehicleState(0.0, 0.0, 0.0)
        goal = GoalPose(8.0, 2.0, 0.0)
        plain = HybridAStarPlanner(env=env, vehicle_length=6.0, vehicle_width=3.0)
        inflated = HybridAStarPlanner(
            env=env, collision_margin=0.4, vehicle_length=6.0, vehicle_width=3.0
        )
        # 无膨胀：目标位姿自由；膨胀 0.4 后：角点 y_max=3.9 侵入墙体。
        self.assertTrue(plain._pose_free(goal.x, goal.y, goal.yaw))
        self.assertFalse(inflated._pose_free(goal.x, goal.y, goal.yaw))
        with self.assertRaises(ValueError):
            inflated.plan(start, goal)

    def test_collision_margin_keeps_clearance(self):
        """collision_margin > 0 时，轨迹上车身矩形与障碍保持至少 margin 净空。"""
        env = _channel_env()
        margin = 0.2
        planner = HybridAStarPlanner(env=env, collision_margin=margin, vehicle_length=4.0, vehicle_width=2.0)
        start = VehicleState(0.0, 0.0, 0.0)
        goal = GoalPose(8.0, 0.0, 0.0)
        traj = planner.plan(start, goal)
        half_l, half_w = 2.0, 1.0
        for px, py, pyaw in traj.points:
            cos_yaw, sin_yaw = np.cos(pyaw), np.sin(pyaw)
            corners = [
                (px + half_l * cos_yaw - half_w * sin_yaw, py + half_l * sin_yaw + half_w * cos_yaw),
                (px + half_l * cos_yaw + half_w * sin_yaw, py + half_l * sin_yaw - half_w * cos_yaw),
                (px - half_l * cos_yaw - half_w * sin_yaw, py - half_l * sin_yaw + half_w * cos_yaw),
                (px - half_l * cos_yaw + half_w * sin_yaw, py - half_l * sin_yaw - half_w * cos_yaw),
            ]
            for cx, cy in corners:
                # 车身角点必须仍离障碍至少 margin（4m 通道内即 |y| <= 1 - 0.2）。
                self.assertLessEqual(abs(cy), 2.0 - margin)

    def test_analytic_expansion_plans_tight_s3_and_s5_scenes(self):
        """S3/S5 从入口倒车入位，仅展开起点即由 RS 曲线精确接管。"""
        cases = [
            ("S3_maintenance", -3.5),
            ("S5_crusher", -3.5),
        ]
        for scene_name, start_y in cases:
            with self.subTest(scene=scene_name):
                bundle = build_scene(scene_name)
                goal = bundle.free_spots()[0].pose
                start = VehicleState(goal.x, start_y, goal.yaw)
                planner = HybridAStarPlanner(
                    env=bundle.env,
                    vehicle_length=6.0,
                    vehicle_width=3.0,
                    collision_margin=0.1,
                    analytic_expansion_distance=20.0,
                    max_expansions=1,
                )
                traj = planner.plan(start, goal)
                np.testing.assert_allclose(traj.points[-1], [goal.x, goal.y, goal.yaw], atol=1e-6)
                for px, py, pyaw in traj.points:
                    self.assertTrue(planner._pose_free(float(px), float(py), float(pyaw)))

    def test_vehicle_scaled_analytic_neighborhood_handles_turning_approaches(self):
        """v5 的 T3 距离解析邻域也覆盖 S3/S5 非直线入位。"""
        cases = [
            ("S3_maintenance", VehicleState(0.0, -8.0, np.pi / 2.0)),
            ("S5_crusher", VehicleState(-1.0, -5.0, 0.0)),
        ]
        for scene_name, start in cases:
            with self.subTest(scene=scene_name):
                bundle = build_scene(scene_name)
                goal = bundle.free_spots()[0].pose
                planner = HybridAStarPlanner(
                    env=bundle.env,
                    max_expansions=5000,
                    **MINING_DRILL_RIG.planner_kwargs(),
                )
                self.assertEqual(planner.analytic_expansion_distance, 30.0)
                traj = planner.plan(start, goal)
                np.testing.assert_allclose(traj.points[-1], [goal.x, goal.y, goal.yaw], atol=1e-6)

    def test_invalid_analytic_interval_raises(self):
        with self.assertRaises(ValueError):
            HybridAStarPlanner(env=_open_env(), analytic_expansion_interval=0)

    def test_planning_wall_clock_limit_raises(self):
        planner = HybridAStarPlanner(
            env=_open_env(),
            max_planning_time_s=1e-9,
        )
        with self.assertRaisesRegex(RuntimeError, "规划超时"):
            planner.plan(VehicleState(0.0, 0.0, 0.0), GoalPose(6.0, 0.0, 0.0))

    def test_requested_direction_biases_search_without_forbidding_other_direction(self):
        planner = HybridAStarPlanner(
            env=_open_env(),
            enable_pivot=True,
            direction_mismatch_penalty=10.0,
        )
        start = VehicleState(0.0, 0.0, 0.0)
        goal = GoalPose(-6.0, 0.0, 0.0)

        reverse = planner.plan(start, goal, preferred_direction=-1)
        forward = planner.plan(start, goal, preferred_direction=1)

        self.assertTrue(
            audit_maneuver_consistency(reverse.points, Maneuver.REVERSE).consistent
        )
        self.assertTrue(
            audit_maneuver_consistency(forward.points, Maneuver.FORWARD).consistent
        )

    def test_reverse_and_pivot_use_two_to_one_cost_without_overriding_reverse_request(
        self,
    ):
        planner = HybridAStarPlanner(
            env=_open_env(),
            plan_v=0.5,
            max_omega=0.5,
            pivot_omega=0.5,
            enable_pivot=True,
            rotation_penalty=2.0,
            direction_mismatch_penalty=2.0,
        )

        forward_cost = planner._directional_translation_cost((1.0,), None)
        reverse_cost = planner._directional_translation_cost((-1.0,), None)
        requested_reverse_cost = planner._directional_translation_cost(
            (-1.0,), -1
        )
        self.assertAlmostEqual(reverse_cost, 2.0 * forward_cost)
        self.assertAlmostEqual(requested_reverse_cost, forward_cost)

        pivot_cost, _ = planner._tracked_direct_candidates(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, np.pi / 2.0),
        )[0]
        pivot_duration = (np.pi / 2.0) / planner.pivot_omega
        self.assertAlmostEqual(pivot_cost, 2.0 * pivot_duration)

    def test_analytic_expansion_can_be_disabled_for_rollback(self):
        planner = HybridAStarPlanner(env=_open_env(), analytic_expansion_distance=0.0)
        trajectory = planner.plan(VehicleState(0.0, 0.0, 0.0), GoalPose(4.0, 0.0, 0.0))
        self.assertGreater(trajectory.horizon, 2)

    def test_tracked_rig_pivots_about_fixed_center_to_match_goal_heading(self):
        planner = HybridAStarPlanner(
            env=_open_env(),
            vehicle_length=6.0,
            vehicle_width=3.0,
            plan_v=0.5,
            max_omega=0.35,
            pivot_omega=0.35,
            enable_pivot=True,
        )
        trajectory = planner.plan(
            VehicleState(0.0, 0.0, 0.0),
            GoalPose(0.0, 0.0, np.pi / 2.0),
        )

        np.testing.assert_allclose(trajectory.points[:, :2], 0.0, atol=1e-7)
        self.assertGreater(np.ptp(trajectory.points[:, 2]), 1.4)
        np.testing.assert_allclose(
            trajectory.points[-1], [0.0, 0.0, np.pi / 2.0], atol=1e-6
        )

    def test_rotation_sweep_rejects_obstacle_missed_by_endpoint_poses(self):
        obstacle = CircleObstacle(1.232, 1.866, 0.08)
        planner = HybridAStarPlanner(
            env=ParkingEnvironment(world_size=20.0, obstacles=[obstacle]),
            vehicle_length=4.0,
            vehicle_width=2.0,
            collision_check_resolution=0.05,
        )
        self.assertTrue(planner._pose_free(0.0, 0.0, 0.0))
        self.assertTrue(planner._pose_free(0.0, 0.0, np.pi / 2.0))
        self.assertFalse(
            planner._swept_segment_free(
                (0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2.0)
            )
        )

    def test_full_footprint_rejects_obstacle_inside_vehicle_not_at_corners(self):
        planner = HybridAStarPlanner(
            env=ParkingEnvironment(
                world_size=20.0,
                obstacles=[CircleObstacle(0.0, 0.0, 0.1)],
            ),
            vehicle_length=6.0,
            vehicle_width=3.0,
        )
        self.assertFalse(planner._pose_free(0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
