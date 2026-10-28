"""严格参考审查；与受约束MPC共享注入的同一净空检查器。"""
import numpy as np

from interfaces import Trajectory
from .sources import SafetyShieldSource, SafetyStopError


class StrictFootprintSource(SafetyShieldSource):
    def _require_safe(self, state, trajectory):
        decision = self.checker.check(state, trajectory)
        if not decision.safe:
            raise SafetyStopError(f'严格参考约束不可满足：{decision.reason}')

    def next_trajectory(self, state):
        # 当前姿态已违反约束时，不能沿用宽松模式的current_pose_collision例外。
        hold = self._hold_trajectory(state)
        self._require_safe(state, hold)
        trajectory, elapsed = super().next_trajectory(state)
        self._require_safe(state, trajectory)
        return trajectory, elapsed

    def _fallback_trajectory(self, state, reason, elapsed_ms=0., *, record_intervention=True):
        trajectory, elapsed = super()._fallback_trajectory(
            state, reason, elapsed_ms, record_intervention=record_intervention)
        self._require_safe(state, trajectory)
        return trajectory, elapsed

    def transition_is_safe(self, state, proposed_state):
        trajectory = Trajectory(np.asarray([[proposed_state.x, proposed_state.y, proposed_state.yaw]]), dt=0.)
        return self.checker.check(state, trajectory).safe
