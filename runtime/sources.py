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
from .trajectory_repair import GeometricFilterStats, SweptFootprintProjector


class SafetyStopError(RuntimeError):
    """门禁无法提供安全控制时请求以 safety_stop 结束当前回合。"""


def _plan_or_infeasible(planner, start: VehicleState, goal: GoalPose) -> Trajectory:
    """调用规划器，并把"无可行轨迹"归一为 ``ValueError``（回合内不可行）。

    规划器用 ``ValueError`` 表达"起点/目标与障碍冲突"，用 ``RuntimeError`` 表达
    "搜索耗尽 / 找不到可行轨迹"（见 `planner/hybrid_astar.py`）。对引擎而言两者
    都是**该回合不可行**，只有非规划类异常才是模型级故障：引擎按 ``ValueError``
    做回合内归因，按 ``RuntimeError`` 判为模型级故障并向上传播。

    不归一化时，一个难解任务会中断整批评测——实测在 V8 全场景协议上复现
    （``E1r`` 专家滚动重规划在紧场景抛 ``RuntimeError`` 导致整批中止）。
    ``HierarchicalPlanningSource`` 与 ``SafetyShieldSource`` 早已按
    ``(RuntimeError, ValueError)`` 处理同一类失败，此处与其保持一致。
    """
    try:
        return planner.plan(start, goal)
    except RuntimeError as exc:
        raise ValueError(f"专家规划无可行轨迹：{exc}") from exc


class TrajectorySource(Protocol):
    """轨迹源接口：begin 初始化回合，next_trajectory 供给参考轨迹。"""

    def begin(self, start: VehicleState, goal: GoalPose) -> None: ...

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        """返回 (全局坐标轨迹, 本次耗时 ms)。"""
        ...


class ExpertSource:
    """专家规划轨迹源：首次取轨迹时规划一次，之后复用。

    规划**延迟到** ``next_trajectory``：``begin`` 不在引擎的失败归因保护区内，
    在 ``begin`` 里规划会让"某个任务专家无解"直接中断整批评测。
    """

    def __init__(self, planner) -> None:
        self.planner = planner
        self._traj: Trajectory | None = None
        self._goal: GoalPose | None = None

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self._traj = None
        self._goal = goal

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        if self._traj is None:
            assert self._goal is not None, "begin 未调用"
            self._traj = _plan_or_infeasible(self.planner, state, self._goal)
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
        trajectory = _plan_or_infeasible(self.planner, state, self._goal)
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

    def __init__(self, network_source, local_planner, lookahead: float = 3.0, near_threshold: float = 5.0, safety_checker=None) -> None:
        self.network = network_source
        self.local_planner = local_planner
        self.lookahead = lookahead
        self.near_threshold = near_threshold
        self.safety_checker = safety_checker
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
        # 局部规划全部失败：若注入安全检查器且网络参考轨迹安全，回退纯网络
        # 保留原本能力（如 S7 平行）；否则 safety_stop（如 S3 紧 bay 纯网络本就不安全）。
        if self.safety_checker is not None and reference.horizon >= 2:
            decision = self.safety_checker.check(state, reference)
            if decision.safe:
                return reference, elapsed_ms
        raise SafetyStopError(
            f"分层局部规划无法到达任何候选子目标（{len(candidates)} 个）"
            f"{'' if self.safety_checker is None else '且网络参考不安全'}："
            f"{last_error}"
        ) from last_error

    def _subgoal_candidates(self, reference: Trajectory, state: VehicleState) -> list[GoalPose]:
        """生成候选子目标：全局目标优先，沿参考渐进点兜底。

        全局目标优先（直接规划到目标是达成"到达+位姿达标"的最可靠路径，
        避免长距离跟随参考导致航向发散振荡）；沿参考点作为中间候选；
        全部不可达时由安全检查器决定回退纯网络或 safety_stop。
        """
        pts = np.asarray(reference.points, dtype=np.float64)
        assert self._goal is not None
        candidates: list[GoalPose] = [self._goal]
        if pts.shape[0] >= 2:
            segments = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
            from_start = (
                np.concatenate([[0.0], np.cumsum(segments)])
                if segments.shape[0]
                else np.array([0.0])
            )
            state_to_start = np.hypot(pts[0, 0] - state.x, pts[0, 1] - state.y)
            for fraction in (0.15, 0.35, 0.6, 1.0):
                target_arc = state_to_start + self.lookahead * fraction
                idx = int(np.searchsorted(from_start, target_arc))
                idx = min(max(idx, 1), pts.shape[0] - 1)
                candidates.append(self._to_goal_pose(pts[idx], self._goal))
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


class GeometricFilterSource:
    """推理侧轨迹级几何过滤：只修正预测轨迹中不可行的部分。

    与 ``SafetyShieldSource``（不可行就整条切到可信回退）的区别在于干预粒度：
    本包装源把进入碰撞的部分**投影回可行域**，其余部分原样交给执行器，
    因此不牺牲网络在可行区段的能力，也不需要回退规划器。

    修不好时按两级退化，**都不放行不可行轨迹**：

    1. **计划保持**：继续跟随上一条已被验证可行的参考的尾段。实测中"修不好"
       绝大多数发生在车辆已经贴着障碍、近端净空无法再抬高的时刻，此时继续沿
       上一条可行轨迹前进比原地停车安全，也不会把碰撞换成一堆停车振荡。
    2. **安全截断**：没有可保持的参考时，截断到最后一个可行位姿（必要时原地保持）。
    """

    def __init__(self, primary, projector: SweptFootprintProjector) -> None:
        self.primary = primary
        self.projector = projector
        self.stats = GeometricFilterStats()
        self._last_feasible: Trajectory | None = None

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self.stats = GeometricFilterStats()
        self._last_feasible = None
        self.primary.begin(start, goal)

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        trajectory, elapsed_ms = self.primary.next_trajectory(state)
        start_pose = np.asarray([state.x, state.y, state.yaw], dtype=np.float64)
        try:
            outcome = self.projector.repair(
                start_pose, np.asarray(trajectory.points, dtype=np.float64)
            )
        except ValueError as exc:
            # 网络给出形状错误或非有限值：明确拒绝，不静默放行。
            self.stats.record_unusable_trajectory("unusable_prediction")
            raise SafetyStopError(f"预测轨迹无法过滤：{exc}") from exc

        if outcome.truncated:
            persisted = self._persist(start_pose, trajectory)
            if persisted is not None:
                self.stats.record(outcome, persisted=True)
                self._last_feasible = persisted
                return persisted, elapsed_ms
            self.stats.record(outcome)
            return Trajectory(outcome.points, dt=trajectory.dt), elapsed_ms

        self.stats.record(outcome)
        if not outcome.modified:
            self._last_feasible = trajectory
            return trajectory, elapsed_ms
        repaired = Trajectory(outcome.points, dt=trajectory.dt)
        self._last_feasible = repaired
        return repaired, elapsed_ms

    def _persist(
        self, start_pose: np.ndarray, fallback_shape_source: Trajectory
    ) -> Trajectory | None:
        """上一条可行参考中仍然可执行的那一段；不可用返回 None。"""
        if self._last_feasible is None:
            return None
        tail = self.projector.feasible_tail(
            start_pose, np.asarray(self._last_feasible.points, dtype=np.float64)
        )
        if tail is None:
            return None
        return Trajectory(tail, dt=fallback_shape_source.dt)

    def filter_stats(self) -> dict:
        return self.stats.to_dict()

    def record_safety_stop(self) -> None:
        """与 SafetyShieldSource 接口对齐：引擎在 SafetyStopError 路径上会回调。"""
        self.stats.safety_stops += 1


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
