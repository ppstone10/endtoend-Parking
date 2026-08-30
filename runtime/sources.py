"""轨迹源：闭环引擎的轨迹供给策略。

ExpertSource 一次规划全程复用（M1 地基验收 / M4 经典上界基线）；
NetworkSource 每次重规划时感知 → BEV → 网络推理（端到端主线）。
输出轨迹一律为全局坐标，车辆状态与目标位姿同为全局坐标。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from .safety import SafetyShieldStats, TrajectorySafetyChecker


class TrajectorySource(Protocol):
    """轨迹源接口：begin 初始化回合，next_trajectory 供给参考轨迹。"""

    def begin(self, start: VehicleState, goal: GoalPose) -> None: ...

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        """返回 (全局坐标轨迹, 本次耗时 ms)。"""
        ...


class ExpertSource:
    """专家规划轨迹源：回合开始时规划一次，之后复用。"""

    def __init__(self, planner) -> None:
        self.planner = planner
        self._traj: Trajectory | None = None

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self._traj = self.planner.plan(start, goal)

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        assert self._traj is not None, "begin 未调用"
        return self._traj, 0.0


class ReplanningExpertSource:
    """可信回退源：每次从当前状态重新规划到回合目标。"""

    def __init__(self, planner) -> None:
        self.planner = planner
        self._goal: GoalPose | None = None

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self._goal = goal

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        import time

        assert self._goal is not None, "begin 未调用"
        started = time.perf_counter()
        trajectory = self.planner.plan(state, self._goal)
        return trajectory, (time.perf_counter() - started) * 1000.0


class SafetyShieldSource:
    """审查主轨迹，不安全时从当前状态切换到可信回退源。"""

    def __init__(self, primary, fallback, checker: TrajectorySafetyChecker) -> None:
        self.primary = primary
        self.fallback = fallback
        self.checker = checker
        self.stats = SafetyShieldStats()

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self.stats = SafetyShieldStats()
        self.primary.begin(start, goal)
        self.fallback.begin(start, goal)

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        primary, primary_ms = self.primary.next_trajectory(state)
        self.stats.checks += 1
        decision = self.checker.check(state, primary)
        if decision.safe:
            return primary, primary_ms
        self.stats.record_intervention(decision.reason)
        try:
            fallback, fallback_ms = self.fallback.next_trajectory(state)
        except (RuntimeError, ValueError) as exc:
            self.stats.fallback_failures += 1
            raise RuntimeError(
                f"安全门禁拒绝主轨迹（{decision.reason}），可信回退规划失败：{exc}"
            ) from exc
        fallback_decision = self.checker.check(state, fallback)
        if not fallback_decision.safe:
            self.stats.fallback_failures += 1
            raise RuntimeError(
                "安全门禁拒绝主轨迹且回退轨迹仍不安全："
                f"{fallback_decision.reason}"
            )
        return fallback, primary_ms + fallback_ms

    def safety_stats(self) -> dict:
        return self.stats.to_dict()


class NetworkSource:
    """端到端网络轨迹源：每次调用重感知并推理。

    sensor_pipeline 需提供 capture_bev(x, y, yaw) -> BEVTensor；
    model 提供 predict(bev, goal, state) -> Trajectory（车辆中心局部坐标）。
    网络输入的目标位姿与运动状态均为当前位姿局部系。
    """

    def __init__(self, sensor_pipeline, model) -> None:
        self.sensor_pipeline = sensor_pipeline
        self.model = model
        self._goal: GoalPose | None = None

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self._goal = goal
        set_target_goals = getattr(self.sensor_pipeline, "set_target_goals", None)
        if callable(set_target_goals):
            set_target_goals([goal])

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        import time

        assert self._goal is not None, "begin 未调用"
        t0 = time.perf_counter()
        bev = self.sensor_pipeline.capture_bev(state.x, state.y, state.yaw)
        goal_local = self._to_local_goal(state)
        from interfaces import GoalPose as _GoalPose
        from interfaces import VehicleState as _VehicleState

        traj_local = self.model.predict(
            bev,
            _GoalPose(goal_local[0], goal_local[1], goal_local[2]),
            _VehicleState(state.x, state.y, state.yaw, state.v, state.omega),
        )
        points_global = self._to_global(traj_local.points, state)
        traj = Trajectory(points=points_global, dt=traj_local.dt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return traj, elapsed_ms

    def _to_local_goal(self, state: VehicleState) -> np.ndarray:
        assert self._goal is not None
        dx = self._goal.x - state.x
        dy = self._goal.y - state.y
        cos_yaw, sin_yaw = np.cos(state.yaw), np.sin(state.yaw)
        return np.array(
            [
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
                float(np.arctan2(np.sin(self._goal.yaw - state.yaw), np.cos(self._goal.yaw - state.yaw))),
            ]
        )

    @staticmethod
    def _to_global(points_local: np.ndarray, state: VehicleState) -> np.ndarray:
        cos_yaw, sin_yaw = np.cos(state.yaw), np.sin(state.yaw)
        out = np.empty_like(points_local)
        out[:, 0] = state.x + cos_yaw * points_local[:, 0] - sin_yaw * points_local[:, 1]
        out[:, 1] = state.y + sin_yaw * points_local[:, 0] + cos_yaw * points_local[:, 1]
        out[:, 2] = points_local[:, 2] + state.yaw
        return out
