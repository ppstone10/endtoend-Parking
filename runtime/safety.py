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
    fallback_failures: int = 0
    safety_stops: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record_intervention(self, reason: str | None) -> None:
        self.interventions += 1
        key = reason or "unspecified"
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def record_prevented_transition(self, reason: str | None) -> None:
        self.prevented_transitions += 1
        key = f"next_state_{reason or 'unsafe'}"
        self.reasons[key] = self.reasons.get(key, 0) + 1

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
        return result


__all__ = [
    "FootprintTrajectorySafetyChecker",
    "SafetyDecision",
    "SafetyShieldStats",
    "TrajectorySafetyChecker",
]
