"""轨迹源：闭环引擎的轨迹供给策略。

ExpertSource 一次规划全程复用（M1 地基验收 / M4 经典上界基线）；
NetworkSource 每次重规划时感知 → BEV → 网络推理（端到端主线）。
输出轨迹一律为全局坐标，车辆状态与目标位姿同为全局坐标。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        #: 最近一次通过审查的参考轨迹（主或回退），供三级降级沿用尾段。
        self._last_safe: Trajectory | None = None
        #: 主源轨迹的 dt，供原地保持轨迹保持时间语义一致。
        self._reference_dt = 0.1

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self.stats = SafetyShieldStats()
        self._last_safe = None
        self.primary.begin(start, goal)
        self.fallback.begin(start, goal)

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        primary, primary_ms = self.primary.next_trajectory(state)
        self._reference_dt = float(primary.dt)
        self.stats.checks += 1
        decision = self.checker.check(state, primary)
        if decision.safe:
            self._last_safe = primary
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
        except (RuntimeError, ValueError):
            # 专家自己规划不出来：这是**专家能力缺口**，不是安全层失败。
            self.stats.expert_fallback_unavailable += 1
            return self._minimal_risk_trajectory(
                state, "fallback_planner_failed", elapsed_ms
            )
        fallback_decision = self.checker.check(state, fallback)
        if not fallback_decision.safe:
            self.stats.expert_fallback_rejected += 1
            return self._minimal_risk_trajectory(
                state, fallback_decision.reason, elapsed_ms
            )
        self._last_safe = fallback
        return fallback, elapsed_ms + fallback_ms

    def _minimal_risk_trajectory(
        self, state: VehicleState, reason: str | None, elapsed_ms: float = 0.0
    ) -> tuple[Trajectory, float]:
        """三级降级：主轨迹与可信回退都不可用时的最小风险动作（SAFE-SHIELD-003）。

        1. **计划保持**：沿用上一条已验证安全参考中仍然可行的尾段；
        2. **原地保持**：没有可沿用参考时，返回单点轨迹让车辆不动。

        为什么这两者一定安全：回合能走到这一步，说明当前位姿**没有真实接触**
        （否则上一步已按碰撞终止回合）；原地保持不产生新的扫掠，沿用尾段则每个
        候选后缀都经过同一次审查。

        这里**故意不要求当前位姿满足规划余量**：车辆贴障时该要求必然失败，
        用它把关会让安全层永远无路可走——这正是历史 `fallback_failures` 的来源。
        真实接触仍由引擎自己的碰撞判据在每一次转移上兜底。
        """
        if self._last_safe is not None:
            tail = self._safe_tail(state, self._last_safe)
            if tail is not None:
                self.stats.record_minimal_risk(reused=True, reason=reason)
                return tail, elapsed_ms
        self.stats.record_minimal_risk(reused=False, reason=reason)
        return self._hold_trajectory(state), elapsed_ms

    def _safe_tail(
        self, state: VehicleState, trajectory: Trajectory
    ) -> Trajectory | None:
        """``trajectory`` 中从离 ``state`` 最近点起、仍然安全的最长后缀。"""
        points = np.asarray(trajectory.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0:
            return None
        nearest = int(
            np.argmin(np.hypot(points[:, 0] - state.x, points[:, 1] - state.y))
        )
        for index in range(nearest, points.shape[0]):
            candidate = Trajectory(points[index:], dt=trajectory.dt)
            if self.checker.check(state, candidate).safe:
                return candidate
        return None

    def _hold_trajectory(self, state: VehicleState) -> Trajectory:
        """原地保持：单点轨迹（MPC 对单点直接返回，不插值），目标即当前位姿。

        单点而非两点同位置：`MPCController._align_dt` 对单点走早退分支，
        两点同位置且 dt=0 会进入零长度插值路径。
        """
        return Trajectory(
            np.asarray([[state.x, state.y, state.yaw]], dtype=np.float64),
            dt=self._reference_dt,
        )

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
        if decision.safe:
            return True
        # 当前位姿本身已在要求净空层内时，被否决的是**状态**而不是这次转移：
        # 车辆无法靠"不动"修复该状态，继续否决任何动作只会让回合以 safety_stop
        # 结束（历史上"回退失败"的一半来源）。真实接触由引擎自己的
        # `_check_collision`（0 余量）在每次转移上兜底，故此处放行。
        return decision.reason == "current_pose_collision"

    def record_safety_stop(self) -> None:
        self.stats.safety_stops += 1

    def minimal_risk_trajectory(self, state: VehicleState) -> Trajectory:
        """返回当前状态下的最小风险轨迹，供引擎在替换后转移仍不安全时取用。

        `guard_transition` 给出的是通过轨迹审查的替代参考，但控制器的实际推进
        仍可能离开安全集合；此时按 `SAFE-SHIELD-003` 继续退到最小风险动作，
        而不是直接放弃回合。
        """
        trajectory, _ = self._minimal_risk_trajectory(
            state, "transition_guard_exhausted", 0.0
        )
        return trajectory

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


@dataclass
class ConsistentPlanStats:
    """一致性迟滞的记账：翻转次数、保持次数、以及达到上限后被迫放行的次数。"""

    checks: int = 0
    direction_flips: int = 0
    held: int = 0
    released: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["held_rate"] = self.held / self.checks if self.checks else 0.0
        result["flip_rate"] = (
            self.direction_flips / self.checks if self.checks else 0.0
        )
        return result


class ConsistentPlanSource:
    """跨重规划一致性迟滞：方向相反时保持上一条计划，抑制抖动。

    **动机（2026-10-13 实测）**：网络单条计划已经 97–99% 单调（专家 100% 单调），
    但相邻两次重规划的"净趋近方向"有 12–24% 概率相反；闭环失败回合的前进翻转率是
    30–38%，成功回合只有 0–5%。因此震荡是**跨重规划的时间不一致**，不是单条计划的质量
    问题——这也是"计划内损失"反复无效的原因。

    **做法**：每次重规划比较新计划与上一条计划（截去已被执行的部分）的净趋近方向；
    若两者相反，且上一条仍有可执行的剩余段、且连续保持次数未超上限，则继续执行上一条。
    这是抖动系统里的经典迟滞：把"来回翻"变成"翻一次就锁一会儿"。

    与 `SafetyShieldSource` / `GeometricFilterSource` 的区别：后两者按**几何可行性**干预，
    本包装源按**时间一致性**干预，不看障碍。因此它是纯诊断/机理验证用的推理侧改动，
    不提供任何安全保证——安全性仍由引擎的真实接触判据（以及可选的门禁）负责。
    """

    def __init__(self, primary, *, hold_limit: int = 3) -> None:
        if hold_limit < 0:
            raise ValueError("hold_limit 不能为负")
        self.primary = primary
        self.hold_limit = int(hold_limit)
        self.stats = ConsistentPlanStats()
        self._goal: GoalPose | None = None
        self._last_plan: np.ndarray | None = None
        self._held = 0

    def begin(self, start: VehicleState, goal: GoalPose) -> None:
        self.stats = ConsistentPlanStats()
        self._goal = goal
        self._last_plan = None
        self._held = 0
        self.primary.begin(start, goal)

    def next_trajectory(self, state: VehicleState) -> tuple[Trajectory, float]:
        trajectory, elapsed_ms = self.primary.next_trajectory(state)
        self.stats.checks += 1
        new_plan = np.asarray(trajectory.points, dtype=np.float64)

        if self._goal is None or self._last_plan is None:
            self._last_plan = new_plan
            return trajectory, elapsed_ms

        remaining = self._remaining_plan(self._last_plan, state)
        if remaining is None:
            self._last_plan = new_plan
            self._held = 0
            return trajectory, elapsed_ms

        new_sign = self._approach_sign(new_plan, state)
        held_sign = self._approach_sign(remaining, state)
        flipped = new_sign != 0 and held_sign != 0 and new_sign != held_sign
        if flipped:
            self.stats.direction_flips += 1
        if flipped and self._held < self.hold_limit:
            self.stats.held += 1
            self._held += 1
            return Trajectory(remaining, dt=trajectory.dt), elapsed_ms
        if flipped:
            self.stats.released += 1
        self._held = 0
        self._last_plan = new_plan
        return trajectory, elapsed_ms

    def consistency_stats(self) -> dict:
        return self.stats.to_dict()

    def _remaining_plan(
        self, plan: np.ndarray, state: VehicleState
    ) -> np.ndarray | None:
        """截去已经执行过的部分：从离当前位姿最近的点起取后缀。"""
        if plan.ndim != 2 or plan.shape[0] < 3:
            return None
        distance = np.hypot(plan[:, 0] - state.x, plan[:, 1] - state.y)
        nearest = int(np.argmin(distance))
        remaining = plan[nearest:]
        return remaining if remaining.shape[0] >= 3 else None

    def _approach_sign(self, plan: np.ndarray, state: VehicleState) -> int:
        """净趋近方向：+1 表示末端比起点更靠近目标，-1 表示反而更远，0 表示无明确方向。

        死区取 0.02m，避免"基本没动"的噪声被计成方向翻转。
        """
        if self._goal is None or plan.shape[0] < 2:
            return 0
        start_distance = float(
            np.hypot(plan[0, 0] - self._goal.x, plan[0, 1] - self._goal.y)
        )
        end_distance = float(
            np.hypot(plan[-1, 0] - self._goal.x, plan[-1, 1] - self._goal.y)
        )
        net = start_distance - end_distance
        if abs(net) < 0.02:
            return 0
        return 1 if net > 0.0 else -1


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
