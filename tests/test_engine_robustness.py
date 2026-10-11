"""闭环引擎对轨迹源失败的健壮性测试。

回归背景：`ReplanningExpertSource` 从当前状态重规划时，若起点落在障碍内，
规划器会抛 ValueError。旧实现让异常穿透整个批量评测（实测在 E1r 主集上直接
中断整批），现在改为回合内归因：首步失败记 `planning_failure`，中途失败记
`timeout`，并把原因写进 meta。
"""

from __future__ import annotations

import unittest

import numpy as np

from controller import MPCController
from interfaces import GoalPose, Trajectory, VehicleState
from metrics import EpisodeResult
from planner import HybridAStarPlanner
from runtime import (
    ClosedLoopEngine,
    ExpertSource,
    ReplanningExpertSource,
    TerminalChecker,
)
from runtime.termination import FAILURE_PLANNING, FAILURE_TIMEOUT
from sim import MINING_DRILL_RIG, DifferentialDriveModel, ParkingEnvironment, RectangleObstacle


class _FailingSource:
    """始终无法规划的轨迹源。"""

    def __init__(self, message: str = "起始位姿与障碍冲突") -> None:
        self.message = message

    def begin(self, start, goal) -> None:
        pass

    def next_trajectory(self, state):
        raise ValueError(self.message)


class _FailsAfterFirstSource:
    """第一次成功、之后每次重规划都失败。"""

    def __init__(self) -> None:
        self.calls = 0

    def begin(self, start, goal) -> None:
        self.calls = 0

    def next_trajectory(self, state):
        self.calls += 1
        if self.calls > 1:
            raise ValueError("重规划起点落在障碍内")
        points = np.asarray(
            [[state.x + 0.1 * i, state.y, state.yaw] for i in range(1, 40)],
            dtype=np.float64,
        )
        return Trajectory(points=points, dt=0.1), 0.0


def _engine(source, *, replan_every: int = 5, max_steps: int = 100) -> ClosedLoopEngine:
    return ClosedLoopEngine(
        vehicle_model=DifferentialDriveModel(max_v=2.0, max_omega=1.0),
        mpc=MPCController(dt=0.1, horizon=10, seed=0),
        source=source,
        terminal=TerminalChecker(0.3, np.deg2rad(10.0)),
        replan_every=replan_every,
        max_steps=max_steps,
    )


class TestTrajectorySourceFailureHandling(unittest.TestCase):
    def test_initial_planning_failure_returns_episode_not_exception(self):
        result = _engine(_FailingSource()).run(
            VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0)
        )
        self.assertIsInstance(result, EpisodeResult)
        self.assertFalse(result.success)
        self.assertEqual(result.failure, FAILURE_PLANNING)
        self.assertEqual(result.steps, 0)
        self.assertIn("initial", result.meta["planning_failure"])

    def test_replan_failure_after_progress_is_timeout_with_reason(self):
        source = _FailsAfterFirstSource()
        result = _engine(source, replan_every=5).run(
            VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0)
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure, FAILURE_TIMEOUT)
        self.assertGreater(result.steps, 0)
        self.assertIn("replan@5", result.meta["planning_failure"])

    def test_progress_before_failure_is_preserved_in_path_length(self):
        source = _FailsAfterFirstSource()
        result = _engine(source, replan_every=5).run(
            VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0)
        )
        self.assertGreater(result.path_length, 0.0)

    def test_run_does_not_raise_on_any_source_error(self):
        for message in ("起始位姿与障碍冲突", "无法到达目标", "超出搜索预算"):
            with self.subTest(message=message):
                result = _engine(_FailingSource(message)).run(
                    VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0)
                )
                self.assertFalse(result.success)
                self.assertIn(message, result.meta["planning_failure"])

    def test_runtime_error_still_propagates(self):
        """回归既有契约：模型级 RuntimeError 不得被降级成 planning_failure。"""
        class _BrokenSource(_FailingSource):
            def next_trajectory(self, state):
                raise RuntimeError("model failure")

        with self.assertRaisesRegex(RuntimeError, "model failure"):
            _engine(_BrokenSource()).run(
                VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0)
            )


class _UnreachablePlanner:
    """规划器能构造、但任何目标都找不到可行轨迹（搜索耗尽）。"""

    def plan(self, start, goal):
        raise RuntimeError("Hybrid A* 未能找到可行轨迹")


class TestExpertPlannerInfeasibility(unittest.TestCase):
    """回归背景：规划器用 RuntimeError 表达"搜索耗尽/无可行轨迹"，与 ValueError 的
    "起点或目标在障碍内"同属**回合内不可行**。旧实现让这个 RuntimeError 穿透整批
    评测，在 V8 全场景协议的 E1r 上实测中断全部 300 条。"""

    def test_expert_source_attributes_infeasible_plan_per_episode(self):
        result = _engine(ExpertSource(_UnreachablePlanner())).run(
            VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0)
        )
        self.assertIsInstance(result, EpisodeResult)
        self.assertFalse(result.success)
        self.assertEqual(result.failure, FAILURE_PLANNING)
        self.assertIn("专家规划无可行轨迹", result.meta["planning_failure"])

    def test_replanning_expert_source_attributes_infeasible_replan(self):
        source = ReplanningExpertSource(_UnreachablePlanner())
        source.begin(VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "专家规划无可行轨迹"):
            source.next_trajectory(VehicleState(0.0, 0.0, 0.0))

    def test_real_planner_in_a_blocked_start_still_aborts_per_episode(self):
        """真规划器 + 起点在障碍内：仍按回合归档为 planning_failure，不抛异常。"""
        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(-5.0, 5.0, -5.0, 5.0)],
        )
        planner = HybridAStarPlanner(env=env, vehicle_length=4.0, vehicle_width=2.0)
        result = _engine(ExpertSource(planner)).run(
            VehicleState(0.0, 0.0, 0.0), GoalPose(5.0, 0.0, 0.0)
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure, FAILURE_PLANNING)
        self.assertIn("planning_failure", result.meta)


if __name__ == "__main__":
    unittest.main()
