"""只发出原样安全前缀；近端网络参考还需原联合目标支撑。"""
from __future__ import annotations

import numpy as np
from interfaces import Trajectory
from .sources import SafetyStopError
from .strict_footprint import StrictFootprintSource


class SafePrefixReferenceSource(StrictFootprintSource):
    def __init__(self, primary, fallback, checker, *, terminal, min_duration_s):
        super().__init__(primary, fallback, checker)
        if not np.isfinite(min_duration_s) or min_duration_s <= 0:
            raise ValueError('安全前缀时长必须由正的完整控制时域给出')
        self.terminal = terminal
        self.min_duration_s = float(min_duration_s)
        self._goal = None
        self.admission_records = []
        self._admission_counts = {}

    def begin(self, start, goal):
        super().begin(start, goal)
        self._goal = goal
        self.admission_records = []
        self._admission_counts = {}

    def reference_admission_stats(self):
        return dict(self._admission_counts)

    def _near_goal(self, state):
        return (not self.terminal.reached(state, self._goal)
                and self.terminal.pos_err(state, self._goal) <= 3*self.terminal.tol_pos)

    def _supports_goal(self, trajectory):
        points = trajectory.points
        distance = np.linalg.norm(points[:, :2]-[self._goal.x, self._goal.y], axis=1)
        yaw = np.abs(np.arctan2(np.sin(points[:, 2]-self._goal.yaw), np.cos(points[:, 2]-self._goal.yaw)))
        return bool(np.any((distance < self.terminal.tol_pos) & (yaw < self.terminal.tol_yaw)))

    def _first_blocked_index(self, state, trajectory):
        checker = self.checker.footprint_checker
        previous = state.to_array()[:3]
        if not np.isfinite(trajectory.points).all():
            return 0
        for index, point in enumerate(trajectory.points):
            if not checker.swept_segment_free(previous, point):
                return index
            previous = point
        return None

    def _record(self, mode, raw, emitted, blocked, reason):
        self._admission_counts[mode] = self._admission_counts.get(mode, 0) + 1
        if reason:
            self._admission_counts[reason] = self._admission_counts.get(reason, 0) + 1
        self.admission_records.append({'mode': mode, 'raw_points': raw.horizon,
            'emitted_points': emitted.horizon, 'first_blocked_index': blocked, 'reason': reason,
            'raw_dt': raw.dt, 'emitted_dt': emitted.dt,
            'min_duration_s': self.min_duration_s})

    def next_trajectory(self, state):
        self._require_safe(state, self._hold_trajectory(state))
        raw, elapsed = self.primary.next_trajectory(state)
        self._reference_dt = raw.dt
        self.stats.checks += 1
        decision = self.checker.check(state, raw)
        blocked = None
        candidate, reason, mode = raw, None, 'unchanged'
        if not decision.safe:
            blocked = self._first_blocked_index(state, raw)
            if blocked is not None and blocked > 0 and np.isfinite(raw.dt) and raw.dt > 0:
                candidate = Trajectory(raw.points[:blocked].copy(), dt=raw.dt)
                if (candidate.horizon-1)*candidate.dt >= self.min_duration_s:
                    mode = 'safe_prefix'
                else:
                    reason = 'prefix_too_short'
            else:
                reason = 'unsafe_first_bridge'
        if reason is None and self._near_goal(state) and not self._supports_goal(candidate):
            reason = 'near_goal_unsupported'
        if reason is None:
            self._require_safe(state, candidate)
            if mode == 'safe_prefix':
                self.stats.record_intervention('safe_prefix')
            self._last_safe = candidate
            self._record(mode, raw, candidate, blocked, None)
            return candidate, elapsed
        emitted, elapsed = self._fallback_trajectory(state, reason, elapsed)
        if self._near_goal(state) and not self._supports_goal(emitted):
            raise SafetyStopError('近端严格回退也没有原联合目标支撑')
        self._record('fallback', raw, emitted, blocked, reason)
        return emitted, elapsed
