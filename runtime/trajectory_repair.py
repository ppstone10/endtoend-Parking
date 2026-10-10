"""推理侧轨迹级几何过滤：把预测轨迹中进入碰撞的部分投影回可行域。

与 ``SafetyShieldSource``（轨迹不可行就整条接管）不同，本模块只对**不可行的那部分**
做最小侧向平移，其余部分原样保留，因此不牺牲网络在可行区段的能力，也不触碰训练目标。

为什么是侧向平移
----------------
车辆外廓是一个远长于宽的有向矩形（矿卡 6×3m）。沿车身纵向平移一个位姿几乎不改变
它的占用区域（长边方向的扫掠近似平移不变），能改变"是否撞上"的自由度基本只有**侧向**。
因此候选位移只取车身侧轴的两个方向。纯侧向还有一个必要性质：位移不含纵向分量，
轨迹的前进单调性不会被破坏——一旦允许纵向分量，最小位移搜索就会用"把位姿沿航迹
往回滑一点"来解开擦角碰撞，单点看着可行，整条轨迹却出现倒退。

算法（斜率受限的位移场迭代）
--------------------------
每一轮：① 对每个仍不可行的位姿求"最小可行侧移"（方向离散 + 幅值二分）；
② 把位移场投影到**可达锥**上——沿航迹弧长，相邻位姿的侧移差不得超过
``max_lateral_slope`` × 弧长差，于是需要侧移的点会把邻近点一起拉过去；
③ 重复至无可消解点。
第②步是关键：只移动被挡住的那个点会得到无法跟踪的折角，而可达锥保证
"越早越少、越晚越多"的连续侧移——等价于要求车辆**提前开始避让**。
收敛后做一次完整扫掠校验；仍不可行则交给调用方退化处理，绝不放行不可行轨迹。

当前位姿是既成事实
------------------
车辆已经在哪儿不能由过滤器改变。若它已落在要求的净空层内，过滤器不会要求它
"先退出去"，只要求它不再真正接触障碍（``contact``），并从第一个预测点起恢复要求的净空。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np


class FootprintFreeSpace(Protocol):
    """过滤器所需的最小几何接口（与场景、车辆参数解耦）。"""

    def pose_free(self, x: float, y: float, yaw: float) -> bool: ...

    def swept_segment_free(self, start_pose: np.ndarray, end_pose: np.ndarray) -> bool: ...


@dataclass(frozen=True)
class RepairOutcome:
    """一次轨迹过滤的结果。

    ``points`` 为过滤后的轨迹点；``modified`` 表示相对输入确有改动；
    ``truncated`` 表示投影失败、退化为截断；``held`` 表示连第一步都不可行，
    只能原地保持（单点轨迹，车辆不动）。
    """

    points: np.ndarray
    modified: bool
    truncated: bool
    held: bool
    max_offset_m: float
    blocked_poses: int
    residual_blocked: int
    reason: str | None


@dataclass
class GeometricFilterStats:
    """轨迹过滤的回合统计（供报告核验干预强度与失败模式）。"""

    checks: int = 0
    unmodified: int = 0
    repaired: int = 0
    persisted: int = 0
    truncated: int = 0
    held: int = 0
    safety_stops: int = 0
    blocked_poses: int = 0
    residual_blocked_poses: int = 0
    offset_m_sum: float = 0.0
    max_offset_m: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)

    def record(self, outcome: RepairOutcome, *, persisted: bool = False) -> None:
        self.checks += 1
        self.blocked_poses += int(outcome.blocked_poses)
        self.residual_blocked_poses += int(outcome.residual_blocked)
        self.offset_m_sum += float(outcome.max_offset_m)
        self.max_offset_m = max(self.max_offset_m, float(outcome.max_offset_m))
        if persisted:
            self.persisted += 1
        elif outcome.held:
            self.held += 1
        elif outcome.truncated:
            self.truncated += 1
        elif outcome.modified:
            self.repaired += 1
        else:
            self.unmodified += 1
        if outcome.reason:
            self.reasons[outcome.reason] = self.reasons.get(outcome.reason, 0) + 1

    def record_unusable_trajectory(self, reason: str) -> None:
        """输入轨迹本身不可用（形状/非有限值），过滤无法进行。"""
        self.checks += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reasons"] = dict(sorted(self.reasons.items()))
        result["intervention_rate"] = (
            (self.checks - self.unmodified) / self.checks if self.checks else 0.0
        )
        result["mean_offset_m"] = (
            self.offset_m_sum / self.checks if self.checks else 0.0
        )
        return result


class SweptFootprintProjector:
    """把不可行的预测轨迹投影回可行域的迭代弹性带。

    ``required`` 为"要求的净空"自由空间（与规划器同一膨胀量）；
    ``contact`` 为"真实接触"自由空间（膨胀量为 0），只用于当前位姿及其第一段扫掠。
    ``max_offset_m`` 是单次搜索的幅值上限，也是整条轨迹的位移上限：
    超出即认为"这条轨迹已经不在网络意图附近"，宁可截断也不硬拉。
    ``max_lateral_slope`` 为可达锥的斜率上限（侧移米 / 航迹弧长米），
    决定"最多提前多久开始避让"。
    """

    #: 候选位移方向相对车身侧轴的角度（度）。只取纯侧向：
    #: 一旦允许纵向分量，最小位移搜索就会用"把位姿沿航迹往回滑一点"来解开
    #: 擦角碰撞——单点看着可行，整条轨迹却出现倒退，MPC 会跟着来回抽。
    #: 纵向的调整属于时间参数化，不属于几何投影。
    OFFSET_ANGLES_DEG: tuple[float, ...] = (0.0,)

    def __init__(
        self,
        required: FootprintFreeSpace,
        *,
        contact: FootprintFreeSpace | None = None,
        max_offset_m: float = 1.0,
        max_lateral_slope: float = 0.5,
        bisection_steps: int = 6,
        max_rounds: int = 6,
        cone_passes: int = 8,
    ) -> None:
        if max_offset_m <= 0.0:
            raise ValueError("max_offset_m 必须为正")
        if max_lateral_slope <= 0.0:
            raise ValueError("max_lateral_slope 必须为正")
        if bisection_steps < 1:
            raise ValueError("bisection_steps 至少为 1")
        if max_rounds < 1:
            raise ValueError("max_rounds 至少为 1")
        if cone_passes < 1:
            raise ValueError("cone_passes 至少为 1")
        self.required = required
        self.contact = contact if contact is not None else required
        self.max_offset_m = float(max_offset_m)
        self.max_lateral_slope = float(max_lateral_slope)
        self.bisection_steps = int(bisection_steps)
        self.max_rounds = int(max_rounds)
        self.cone_passes = int(cone_passes)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def repair(self, start_pose: np.ndarray, points: np.ndarray) -> RepairOutcome:
        """返回把 ``points`` 投影到可行域后的轨迹（失败时退化为安全截断）。"""
        base = np.asarray(points, dtype=np.float64)
        start = np.asarray(start_pose, dtype=np.float64)
        if base.ndim != 2 or base.shape[1] != 3 or base.shape[0] == 0:
            raise ValueError(f"待过滤轨迹形状必须为 (N,3) 且非空，实际 {base.shape}")
        if not np.isfinite(base).all() or not np.isfinite(start).all():
            raise ValueError("待过滤轨迹与起始位姿必须为有限值")

        blocked_poses = self._count_blocked(start, base)
        if blocked_poses == 0:
            return RepairOutcome(
                points=base,
                modified=False,
                truncated=False,
                held=False,
                max_offset_m=0.0,
                blocked_poses=0,
                residual_blocked=0,
                reason=None,
            )

        offsets = np.zeros((base.shape[0], 2), dtype=np.float64)
        arcs = self._arc_lengths(start, base)
        reason: str | None = None
        for _ in range(self.max_rounds):
            poses = base.copy()
            poses[:, :2] += offsets
            resolved, unresolved = self._resolve_blocked(start, poses, offsets)
            if not resolved:
                # 本轮一个点都没能推进：可达锥已无助于事，交给截断兜底。
                reason = "unreachable" if unresolved else None
                break
            # 每轮结束都做可达锥投影，保证交出去的位移场始终是**可跟踪**的：
            # 只在本轮解完就退出会留下"单点硬折"，因此这一步不能省。
            offsets = self._reachable_cone(offsets, arcs, base[:, 2])
            if float(np.abs(offsets).max()) > self.max_offset_m:
                reason = "offset_cap"
                break

        poses = base.copy()
        poses[:, :2] += offsets
        residual = self._count_blocked(start, poses)
        max_offset = float(np.abs(offsets).max()) if offsets.size else 0.0
        # 判定只看最终校验：可达锥跨轮次抬高上游位姿，中途"某点暂时无解"是正常的。
        if residual == 0 and max_offset <= self.max_offset_m:
            return RepairOutcome(
                points=poses,
                modified=max_offset > 0.0,
                truncated=False,
                held=False,
                max_offset_m=max_offset,
                blocked_poses=blocked_poses,
                residual_blocked=0,
                reason=None,
            )
        return self._truncate(
            start, base, blocked_poses, reason or "residual_blocked", residual, max_offset
        )

    def is_feasible(self, start_pose: np.ndarray, points: np.ndarray) -> bool:
        """按与过滤同一口径判定整条轨迹是否可行（供测试与诊断复用）。"""
        start = np.asarray(start_pose, dtype=np.float64)
        poses = np.asarray(points, dtype=np.float64)
        return self._count_blocked(start, poses) == 0

    def feasible_tail(
        self, start_pose: np.ndarray, points: np.ndarray
    ) -> np.ndarray | None:
        """返回从 ``start_pose`` 之后仍然可行的轨迹尾段；不可行返回 None。

        用途是"计划保持"：新的预测轨迹修不好时，退回去继续跟随**上一条已被
        验证可行**的参考，而不是原地停车。尾段整体可行即安全，无需重新求解。
        """
        poses = np.asarray(points, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
            return None
        start = np.asarray(start_pose, dtype=np.float64)
        nearest = int(
            np.argmin(np.hypot(poses[:, 0] - start[0], poses[:, 1] - start[1]))
        )
        for index in range(nearest, poses.shape[0]):
            tail = poses[index:]
            if tail.shape[0] == 0:
                return None
            if self.is_feasible(start, tail):
                return tail
        return None

    # ------------------------------------------------------------------
    # 迭代求解
    # ------------------------------------------------------------------

    def _resolve_blocked(
        self, start: np.ndarray, poses: np.ndarray, offsets: np.ndarray
    ) -> tuple[bool, bool]:
        """逐个消解仍不可行的位姿。

        返回 ``(是否有改动, 是否残留无解点)``。单点本轮无解不算失败：求出的部分侧移
        会经可达锥抬高上游位姿，下一轮该点往往就能解开，故先继续推进其余点。
        """
        changed = False
        unresolved = False
        for index in range(poses.shape[0]):
            previous = start if index == 0 else poses[index - 1]
            relax = index == 0
            if self._cleared(previous, poses[index], relax=relax):
                continue
            delta = self._clearing_offset(previous, poses[index], relax=relax)
            if delta is None:
                unresolved = True
                continue
            offsets[index] += delta
            poses[index] = self._shifted(poses[index], delta)
            changed = True
        return changed, unresolved

    def _clearing_offset(
        self, previous: np.ndarray, pose: np.ndarray, *, relax: bool
    ) -> np.ndarray | None:
        """求让 ``previous → pose`` 恢复可行的最小侧移；无解返回 None。

        先按完整口径（位姿净空 + 入段扫掠）求解；整条入段在幅值上限内无解时，
        退一步只按**位姿净空**求解——它给不出可执行结果，但给出正确的侧移方向与
        缺口大小，供可达锥把上游位姿一起拉过来，下一轮再按完整口径收敛。
        """
        delta = self._search_offset(previous, pose, relax=relax, with_segment=True)
        if delta is not None:
            return delta
        return self._search_offset(previous, pose, relax=relax, with_segment=False)

    def _search_offset(
        self,
        previous: np.ndarray,
        pose: np.ndarray,
        *,
        relax: bool,
        with_segment: bool,
    ) -> np.ndarray | None:
        """在候选方向上二分最小可行幅值；全都无解返回 None。

        二分假定"幅值越大越可能可行"，自由空间非凸时这条假定不严格成立，
        因此它只是求一个**够用**的侧移量，最终以整条轨迹的完整扫掠校验为准。
        """
        best_magnitude: float | None = None
        best_direction: np.ndarray | None = None
        for direction in self._offset_directions(pose[2]):
            if not self._cleared(
                previous,
                self._shifted(pose, direction * self.max_offset_m),
                relax=relax,
                with_segment=with_segment,
            ):
                continue
            low, high = 0.0, self.max_offset_m
            for _ in range(self.bisection_steps):
                middle = 0.5 * (low + high)
                if self._cleared(
                    previous,
                    self._shifted(pose, direction * middle),
                    relax=relax,
                    with_segment=with_segment,
                ):
                    high = middle
                else:
                    low = middle
            if best_magnitude is None or high < best_magnitude:
                best_magnitude, best_direction = high, direction
        if best_magnitude is None or best_direction is None:
            return None
        return best_direction * best_magnitude

    @staticmethod
    def _shifted(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """位姿侧移：只改位置，不改航向（航向由网络意图决定，过滤器不代它决定朝向）。"""
        shifted = np.array(pose, dtype=np.float64, copy=True)
        shifted[0] += float(delta[0])
        shifted[1] += float(delta[1])
        return shifted

    def _offset_directions(self, yaw: float) -> np.ndarray:
        """车身侧轴按 ``OFFSET_ANGLES_DEG`` 旋转后的候选位移方向（单位向量，正负对称）。"""
        lateral = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
        angles = np.radians(np.asarray(self.OFFSET_ANGLES_DEG, dtype=np.float64))
        rotation = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        base = np.concatenate([rotation, -rotation], axis=0)
        return base @ np.array(
            [[lateral[0], lateral[1]], [-lateral[1], lateral[0]]], dtype=np.float64
        )

    @staticmethod
    def _arc_lengths(start: np.ndarray, poses: np.ndarray) -> np.ndarray:
        """各位姿沿航迹到当前位置的累计弧长（含当前位置到首点的一段）。"""
        stacked = np.vstack([start[None, :2], poses[:, :2]])
        steps = np.hypot(*np.diff(stacked, axis=0).T)
        return np.cumsum(steps)

    def _reachable_cone(
        self, offsets: np.ndarray, arcs: np.ndarray, yaws: np.ndarray
    ) -> np.ndarray:
        """把位移场投影到斜率受限的可达锥上。

        侧移只沿各点自身的车身侧轴，故先把位移场折算成**带符号的侧移量**再做一维
        投影：相邻点的侧移差不得超过 ``max_lateral_slope`` × 弧长差。投影用成对
        约束的迭代修正（修正量在两点间对半分），因此需要侧移的点会把上游航迹
        一起拉过去，形成"提前开始避让"的连续侧移，而不是单点硬折。
        """
        lateral = np.stack([-np.sin(yaws), np.cos(yaws)], axis=1)
        magnitudes = np.einsum("ij,ij->i", offsets, lateral)
        limit = self.max_lateral_slope * np.diff(arcs)
        for _ in range(self.cone_passes):
            gap = magnitudes[1:] - magnitudes[:-1]
            excess = np.maximum(np.abs(gap) - limit, 0.0)
            if not np.any(excess > 0.0):
                break
            correction = 0.5 * np.sign(gap) * excess
            magnitudes[1:] -= correction
            magnitudes[:-1] += correction
        return magnitudes[:, None] * lateral

    # ------------------------------------------------------------------
    # 可行性判定与退化
    # ------------------------------------------------------------------

    def _cleared(
        self,
        previous: np.ndarray,
        pose: np.ndarray,
        *,
        relax: bool,
        with_segment: bool = True,
    ) -> bool:
        """位姿满足要求净空，且（可选）上一位置到它的扫掠段可行。"""
        if not self.required.pose_free(float(pose[0]), float(pose[1]), float(pose[2])):
            return False
        if not with_segment:
            return True
        return self._segment_cleared(previous, pose, relax=relax)

    def _segment_cleared(
        self, previous: np.ndarray, pose: np.ndarray, *, relax: bool
    ) -> bool:
        space = self.contact if relax else self.required
        return bool(space.swept_segment_free(previous, pose))

    def _count_blocked(self, start: np.ndarray, poses: np.ndarray) -> int:
        blocked = 0
        for index in range(poses.shape[0]):
            previous = start if index == 0 else poses[index - 1]
            if not self._cleared(previous, poses[index], relax=index == 0):
                blocked += 1
        return blocked

    def _truncate(
        self,
        start: np.ndarray,
        base: np.ndarray,
        blocked_poses: int,
        reason: str,
        residual: int,
        attempted_offset_m: float = 0.0,
    ) -> RepairOutcome:
        """截断到最后一个可行位姿；连第一个位姿都不可行时原地保持。"""
        last = -1
        for index in range(base.shape[0]):
            previous = start if index == 0 else base[index - 1]
            if not self._cleared(previous, base[index], relax=index == 0):
                break
            last = index
        if last >= 0:
            points = base[: last + 1]
            held = False
        else:
            points = start.reshape(1, 3)
            held = True
        return RepairOutcome(
            points=points,
            modified=True,
            truncated=True,
            held=held,
            max_offset_m=attempted_offset_m,
            blocked_poses=blocked_poses,
            residual_blocked=residual,
            reason=reason,
        )


__all__ = [
    "FootprintFreeSpace",
    "GeometricFilterStats",
    "RepairOutcome",
    "SweptFootprintProjector",
]
