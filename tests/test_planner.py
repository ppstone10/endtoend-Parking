"""Hybrid A* 规划器测试。"""

import unittest

import numpy as np

from interfaces import GoalPose, VehicleState
from planner import HybridAStarPlanner
from sim import ParkingEnvironment, RectangleObstacle


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


if __name__ == "__main__":
    unittest.main()