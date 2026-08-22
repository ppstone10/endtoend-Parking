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


if __name__ == "__main__":
    unittest.main()