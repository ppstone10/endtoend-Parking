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
    def _make(self, lookahead=3.0, near_threshold=5.0):
        network = _FakeNetwork(length=10.0)
        planner = _FakePlanner()
        source = HierarchicalPlanningSource(
            network, planner, lookahead=lookahead, near_threshold=near_threshold
        )
        return network, planner, source

    def test_far_state_prioritizes_goal(self):
        # 全局目标优先，避免长距离跟随参考振荡。
        network = _FakeNetwork(length=10.0)  # 参考从 0 到 10m
        planner = _FakePlanner()
        source = HierarchicalPlanningSource(network, planner, lookahead=3.0)
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(10.0, 0.0, 0.0))
        source.next_trajectory(VehicleState(0.0, 0.0, 0.0))
        self.assertIsNotNone(planner.last_goal)
        self.assertAlmostEqual(planner.last_goal.x, 10.0)

    def test_far_state_falls_back_to_reference_when_goal_unreachable(self):
        network = _FakeNetwork(length=10.0)
        planner = _FakePlanner()
        planner.max_x = 5.0

        def plan(state, goal):
            if goal.x > planner.max_x:
                raise ValueError("目标不可达")
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
        # 远距目标不可达时回退沿参考渐进点（<6m）。
        self.assertIsNotNone(planner.last_goal)
        self.assertLessEqual(planner.last_goal.x, 6.0)

    def test_all_unreachable_falls_back_to_network_when_safe(self):
        network = _FakeNetwork(length=10.0)
        planner = _FakePlanner()

        def plan(state, goal):
            raise ValueError("全部不可达")

        planner.plan = plan

        class _Safe:
            def check(self, state, trajectory):
                from runtime import SafetyDecision
                return SafetyDecision(safe=True)

        source = HierarchicalPlanningSource(
            network, planner, lookahead=3.0, safety_checker=_Safe()
        )
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(10.0, 0.0, 0.0))
        traj, _ = source.next_trajectory(VehicleState(0.0, 0.0, 0.0))
        self.assertEqual(traj.horizon, 21)  # 安全则回退纯网络

    def test_all_unreachable_raises_when_network_unsafe(self):
        network = _FakeNetwork(length=10.0)
        planner = _FakePlanner()

        def plan(state, goal):
            raise ValueError("全部不可达")

        planner.plan = plan

        class _Unsafe:
            def check(self, state, trajectory):
                from runtime import SafetyDecision
                return SafetyDecision(safe=False, reason="collision")

        source = HierarchicalPlanningSource(
            network, planner, lookahead=3.0, safety_checker=_Unsafe()
        )
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