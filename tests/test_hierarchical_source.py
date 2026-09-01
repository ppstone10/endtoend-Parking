"""分层轨迹源（路线 B）子目标选择与接口测试。"""

from __future__ import annotations

import unittest

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from runtime import HierarchicalPlanningSource


class _FakeNetwork:
    """固定返回直线参考轨迹的网络源替身。"""

    def __init__(self, length: float = 10.0, n: int = 21) -> None:
        self.traj = Trajectory(
            points=np.stack(
                [np.linspace(0.0, length, n), np.zeros(n), np.zeros(n)], axis=1
            ),
            dt=0.2,
        )

    def begin(self, start, goal) -> None:
        pass

    def next_trajectory(self, state):
        return self.traj, 0.0


class _FakePlanner:
    """记录子目标并返回一段短轨迹的局部规划器替身。"""

    def __init__(self) -> None:
        self.last_goal: GoalPose | None = None

    def plan(self, state, goal):
        self.last_goal = goal
        return Trajectory(
            points=np.stack(
                [np.linspace(0.0, 2.0, 11), np.zeros(11), np.zeros(11)], axis=1
            ),
            dt=0.1,
        )


class TestHierarchicalPlanningSource(unittest.TestCase):
    def _make(self, lookahead=3.0):
        network = _FakeNetwork(length=10.0)
        planner = _FakePlanner()
        source = HierarchicalPlanningSource(network, planner, lookahead=lookahead)
        return network, planner, source

    def test_subgoal_picked_ahead_of_current_state(self):
        _, planner, source = self._make(lookahead=3.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(10.0, 0.0, 0.0))
        traj, _ = source.next_trajectory(VehicleState(1.0, 0.0, 0.0))
        self.assertIsNotNone(planner.last_goal)
        # 子目标应在当前状态前方约 lookahead 弧长（沿参考）。
        self.assertGreater(planner.last_goal.x, 1.0)
        self.assertAlmostEqual(planner.last_goal.x, 4.0, delta=0.6)

    def test_falls_back_to_closer_subgoal_when_far_unreachable(self):
        # 局部规划器拒绝 x>5 的子目标，验证逐步缩短 lookahead 回退。
        network = _FakeNetwork(length=10.0)
        planner = _FakePlanner()
        planner.max_x = 5.0

        def plan(state, goal):
            if goal.x > planner.max_x:
                raise ValueError("子目标不可达")
            planner.last_goal = goal
            return Trajectory(
                points=np.stack(
                    [np.linspace(0.0, 2.0, 11), np.zeros(11), np.zeros(11)], axis=1
                ),
                dt=0.1,
            )

        planner.plan = plan
        source = HierarchicalPlanningSource(network, planner, lookahead=3.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(10.0, 0.0, 0.0))
        source.next_trajectory(VehicleState(0.0, 0.0, 0.0))
        self.assertIsNotNone(planner.last_goal)
        self.assertLessEqual(planner.last_goal.x, 5.0)

    def test_subgoal_at_end_when_reference_short(self):
        _, planner, source = self._make(lookahead=20.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(10.0, 0.0, 0.0))
        source.next_trajectory(VehicleState(0.0, 0.0, 0.0))
        # 参考总长 10m < lookahead 20m → 取参考终点附近。
        self.assertAlmostEqual(planner.last_goal.x, 10.0, delta=0.6)

    def test_empty_reference_falls_back_to_goal(self):
        network = _FakeNetwork(length=10.0)
        network.traj = Trajectory(points=np.zeros((0, 3)), dt=0.2)
        planner = _FakePlanner()
        source = HierarchicalPlanningSource(network, planner, lookahead=3.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 2.0, 1.0))
        source.next_trajectory(VehicleState(0.0, 0.0, 0.0))
        self.assertEqual(planner.last_goal.x, 5.0)
        self.assertEqual(planner.last_goal.y, 2.0)

    def test_all_subgoals_unreachable_raises(self):
        network = _FakeNetwork(length=10.0)
        planner = _FakePlanner()

        def plan(state, goal):
            raise ValueError("全部不可达")

        planner.plan = plan
        source = HierarchicalPlanningSource(network, planner, lookahead=3.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(10.0, 0.0, 0.0))
        from runtime import SafetyStopError
        with self.assertRaises(SafetyStopError):
            source.next_trajectory(VehicleState(0.0, 0.0, 0.0))

    def test_returned_trajectory_is_local_plan(self):
        _, _, source = self._make(lookahead=3.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(10.0, 0.0, 0.0))
        traj, _ = source.next_trajectory(VehicleState(1.0, 0.0, 0.0))
        self.assertEqual(traj.horizon, 11)
        self.assertAlmostEqual(traj.dt, 0.1)


if __name__ == "__main__":
    unittest.main()