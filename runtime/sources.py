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


class SafetyStopError(RuntimeError):
    """门禁无法提供安全控制时请求以 safety_stop 结束当前回合。"""


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


class HierarchicalPlanningSource:
    """分层轨迹源：网络提供全局参考，局部规划器生成短距轨迹（路线 B）。

    每次重规划：
    1. 网络输出全局参考轨迹（长程意图，可含轻微误差）；
    2. 从当前状态沿参考轨迹累计弧长，取 lookahead 弧长处的目标位姿作为子目标；
    3. 局部规划器（Hybrid A*）从当前状态规划到子目标，得到短距可执行轨迹；
    4. MPC 只跟踪该短段。

    收益：网络只需"长程大致正确"，近端精度由局部规划器兜底，绕开纯网络
    近端轨迹退化（振荡）。网络参考不可达时回退到全局目标。
    """

    def __init__(self, network_source, local_planner, lookahead: float = 3.0) -> None:
        self.network = network_source
        self.local_planner = local_planner
        self.lookahead = lookahead
        self._goal: GoalPose | None = None

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self._goal = goal
        self.network.begin(start, goal)

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        assert self._goal is not None, "begin 未调用"
        import time

        reference, elapsed_ms = self.network.next_trajectory(state)
        candidates = self._subgoal_candidates(reference, state)
        started = time.perf_counter()
        last_error: Exception | None = None
        for subgoal in candidates:
            try:
                trajectory = self.local_planner.plan(state, subgoal)
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                continue
            return trajectory, elapsed_ms + (time.perf_counter() - started) * 1000.0
        raise SafetyStopError(
            f"分层局部规划无法到达任何候选子目标（{len(candidates)} 个）："
            f"{last_error}"
        ) from last_error

    def _subgoal_candidates(self, reference: Trajectory, state: VehicleState) -> list[GoalPose]:
        """生成候选子目标：lookahead 逐步缩短（远→近），最后回退全局目标。"""
        pts = np.asarray(reference.points, dtype=np.float64)
        assert self._goal is not None
        if pts.shape[0] == 0:
            return [self._goal]
        segments = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        from_start = np.concatenate([[0.0], np.cumsum(segments)]) if segments.shape[0] else np.array([0.0])
        state_to_start = np.hypot(pts[0, 0] - state.x, pts[0, 1] - state.y)
        candidates: list[GoalPose] = []
        for fraction in (1.0, 0.6, 0.35, 0.15):
            target_arc = state_to_start + self.lookahead * fraction
            idx = int(np.searchsorted(from_start, target_arc))
            idx = min(max(idx, 1), pts.shape[0] - 1)
            candidates.append(self._to_goal_pose(pts[idx], self._goal))
        candidates.append(self._goal)
        # 去重（相同位姿只保留一次）。
        unique: list[GoalPose] = []
        for candidate in candidates:
            if not any(
                abs(candidate.x - old.x) < 1e-6
                and abs(candidate.y - old.y) < 1e-6
                and abs(candidate.yaw - old.yaw) < 1e-6
                for old in unique
            ):
                unique.append(candidate)
        return unique

    @staticmethod
    def _to_goal_pose(point: np.ndarray, fallback: GoalPose | None) -> GoalPose:
        if point.shape[0] >= 3:
            return GoalPose(float(point[0]), float(point[1]), float(point[2]))
        if fallback is not None:
            return fallback
        return GoalPose(float(point[0]), float(point[1]), 0.0)

    def record_safety_stop(self) -> None:
        """与 SafetyShieldSource 接口对齐：safety_stop 回合计数（当前无统计）。"""


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
        return self._fallback_trajectory(state, decision.reason, primary_ms)

    def _fallback_trajectory(
        self,
        state: VehicleState,
        reason: str | None,
        elapsed_ms: float = 0.0,
        *,
        record_intervention: bool = True,
    ) -> tuple[Trajectory, float]:
        if record_intervention:
            self.stats.record_intervention(reason)
        try:
            fallback, fallback_ms = self.fallback.next_trajectory(state)
        except (RuntimeError, ValueError) as exc:
            self.stats.fallback_failures += 1
            raise SafetyStopError(
                f"安全门禁拒绝主轨迹（{reason}），可信回退规划失败：{exc}"
            ) from exc
        fallback_decision = self.checker.check(state, fallback)
        if not fallback_decision.safe:
            self.stats.fallback_failures += 1
            raise SafetyStopError(
                "安全门禁拒绝主轨迹且回退轨迹仍不安全："
                f"{fallback_decision.reason}"
            )
        return fallback, elapsed_ms + fallback_ms

    def guard_transition(
        self, state: VehicleState, proposed_state: VehicleState
    ) -> tuple[Trajectory | None, float]:
        """在执行控制前阻止离开专家规划安全集合的状态转移。"""
        self.stats.transition_checks += 1
        decision = self.checker.check(
            state,
            Trajectory(
                np.asarray(
                    [[proposed_state.x, proposed_state.y, proposed_state.yaw]],
                    dtype=np.float64,
                ),
                dt=0.0,
            ),
        )
        if decision.safe:
            return None, 0.0
        self.stats.record_prevented_transition(decision.reason)
        return self._fallback_trajectory(
            state,
            f"next_state_{decision.reason or 'unsafe'}",
            record_intervention=False,
        )

    def transition_is_safe(
        self, state: VehicleState, proposed_state: VehicleState
    ) -> bool:
        decision = self.checker.check(
            state,
            Trajectory(
                np.asarray(
                    [[proposed_state.x, proposed_state.y, proposed_state.yaw]],
                    dtype=np.float64,
                ),
                dt=0.0,
            ),
        )
        return decision.safe

    def record_safety_stop(self) -> None:
        self.stats.safety_stops += 1

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
