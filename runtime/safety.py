"""与具体场景解耦的轨迹安全门禁。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from interfaces import Trajectory, VehicleState


@dataclass(frozen=True)
class SafetyDecision:
    safe: bool
    reason: str | None = None


class TrajectorySafetyChecker(Protocol):
    def check(self, state: VehicleState, trajectory: Trajectory) -> SafetyDecision: ...


class FootprintTrajectorySafetyChecker:
    """把规划器完整矩形检查器适配为运行时安全接口。"""

    def __init__(self, footprint_checker) -> None:
        self.footprint_checker = footprint_checker

    def check(self, state: VehicleState, trajectory: Trajectory) -> SafetyDecision:
        safe, reason = self.footprint_checker.check_trajectory(state, trajectory)
        return SafetyDecision(bool(safe), reason)


@dataclass
class SafetyShieldStats:
    checks: int = 0
    transition_checks: int = 0
    interventions: int = 0
    prevented_transitions: int = 0
    #: 安全层**没有任何安全动作可用**的回合数。加入三级最小风险降级（SAFE-SHIELD-003）
    #: 后该情形在构造上不会发生（原地保持一定安全），因此本项应恒为 0；保留字段是为了
    #: 让"安全层是否曾无路可走"这件事始终可核验，而不是从报告里消失。
    fallback_failures: int = 0
    #: 可信回退规划器本身失败（抛异常）的次数。
    expert_fallback_unavailable: int = 0
    #: 可信回退轨迹经审查仍不安全的次数——这是**专家能力缺口**，不等于安全层失败。
    expert_fallback_rejected: int = 0
    #: 三级降级：沿用上一条已验证安全参考的尾段（计划保持）。
    minimal_risk_reuses: int = 0
    #: 三级降级：无可沿用参考，原地保持。
    minimal_risk_holds: int = 0
    safety_stops: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record_reason(self, key: str) -> None:
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def record_intervention(self, reason: str | None) -> None:
        self.interventions += 1
        self.record_reason(reason or "unspecified")

    def record_prevented_transition(self, reason: str | None) -> None:
        self.prevented_transitions += 1
        self.record_reason(f"next_state_{reason or 'unsafe'}")

    def record_minimal_risk(self, *, reused: bool, reason: str | None) -> None:
        kind = "reuse" if reused else "hold"
        if reused:
            self.minimal_risk_reuses += 1
        else:
            self.minimal_risk_holds += 1
        self.record_reason(f"minimal_risk_{kind}_{reason or 'unsafe'}")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["intervention_rate"] = (
            self.interventions / self.checks if self.checks else 0.0
        )
        result["transition_prevention_rate"] = (
            self.prevented_transitions / self.transition_checks
            if self.transition_checks
            else 0.0
        )
        result["minimal_risk_rate"] = (
            (self.minimal_risk_reuses + self.minimal_risk_holds) / self.checks
            if self.checks
            else 0.0
        )
        return result


__all__ = [
    "FootprintTrajectorySafetyChecker",
    "SafetyDecision",
    "SafetyShieldStats",
    "TrajectorySafetyChecker",
]
