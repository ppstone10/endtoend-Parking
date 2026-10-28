"""保留原CEM搜索，只在最终输出危险时改选完整车体可行序列。"""
from __future__ import annotations

import numpy as np

from interfaces import ControlCmd
from sim import DifferentialDriveModel
from .mpc import MPCController


class ControlSafetyError(RuntimeError):
    """没有满足注入车体约束的控制序列；执行器必须停止而非执行危险动作。"""


class ConstrainedMPCController(MPCController):
    def __init__(self, *, footprint_checker, vehicle_model=None, max_candidate_checks=16, **kwargs):
        super().__init__(**kwargs)
        if isinstance(max_candidate_checks, bool) or not isinstance(max_candidate_checks, int) or max_candidate_checks < 1:
            raise ValueError('max_candidate_checks必须为正整数')
        self.footprint_checker = footprint_checker
        self.vehicle_model = vehicle_model if vehicle_model is not None else DifferentialDriveModel(
            max_v=self.max_v, max_omega=self.max_omega)
        self.max_candidate_checks = max_candidate_checks
        self._sample_bank = None
        self._audit = {}
        self._counts = {}

    def reset(self):
        super().reset()
        self._audit = {}
        self._counts = {}

    def safety_audit(self):
        return dict(self._audit)

    def safety_stats(self):
        return dict(self._counts)

    def _batch_cost(self, samples, ref, state):
        costs = super()._batch_cost(samples, ref, state)
        if self._sample_bank is not None:
            self._sample_bank.append((samples.copy(), costs.copy()))
        return costs

    def sequence_safe(self, sequence, state):
        sequence = np.asarray(sequence)
        if sequence.size != self.horizon * 2 or not np.isfinite(sequence).all():
            return False
        controls = sequence.reshape(self.horizon, 2)
        if (np.any(np.abs(controls[:, 0]) > self.max_v)
                or np.any(np.abs(controls[:, 1]) > self.max_omega)):
            return False
        if not np.isfinite(state.to_array()).all() or not self.footprint_checker.pose_free(state.x, state.y, state.yaw):
            return False
        for v, omega in controls:
            after = self.vehicle_model.step(state, ControlCmd(float(v), float(omega)), self.dt)
            if (not np.isfinite(after.to_array()).all()
                    or not self.footprint_checker.swept_segment_free(state.to_array()[:3], after.to_array()[:3])):
                return False
            state = after
        return True

    def _solve(self, u_mean, ref, state):
        if not np.isfinite(ref).all() or not np.isfinite(state.to_array()).all():
            raise ValueError('参考和车辆状态必须有限')
        if not self.footprint_checker.pose_free(state.x, state.y, state.yaw):
            raise ControlSafetyError('当前车体已违反要求净空，不能伪装成安全保持')
        self._sample_bank = []
        try:
            legacy = super()._solve(u_mean, ref, state)
            bank = self._sample_bank
        finally:
            self._sample_bank = None
        legacy_cost = float(super()._batch_cost(legacy[None, :], ref, state)[0])
        safe = self.sequence_safe(legacy, state) and np.isfinite(legacy_cost)
        audit = {'legacy_safe': bool(safe), 'legacy_cost': legacy_cost if np.isfinite(legacy_cost) else None,
                 'checked_samples': 0, 'selected': 'legacy_mean',
                 'selected_cost': legacy_cost if np.isfinite(legacy_cost) else None, 'intervened': not safe}
        selected = legacy
        if not safe:
            candidates = []
            if bank:
                samples = np.concatenate([s for s, _ in bank])
                costs = np.concatenate([c for _, c in bank])
                for index in np.argsort(costs, kind='stable')[:self.max_candidate_checks]:
                    audit['checked_samples'] += 1
                    if np.isfinite(costs[index]) and self.sequence_safe(samples[index], state):
                        candidates.append(('safe_sample', samples[index].copy(), float(costs[index])))
                        break
            for label, sequence in (('shifted_warm_start', u_mean), ('zero_hold', np.zeros_like(u_mean))):
                if self.sequence_safe(sequence, state):
                    cost = float(super()._batch_cost(sequence[None, :], ref, state)[0])
                    if np.isfinite(cost):
                        candidates.append((label, sequence.copy(), cost))
            if not candidates:
                raise ControlSafetyError('未找到完整时域安全控制，含全零保持；禁止执行危险首动作')
            label, selected, cost = min(candidates, key=lambda candidate: candidate[2])
            audit.update(selected=label, selected_cost=cost)
        audit.update(selected_horizon_safe=True, selected_sequence=selected.reshape(-1, 2).tolist())
        self._audit = audit
        self._counts['checks'] = self._counts.get('checks', 0) + 1
        self._counts['interventions'] = self._counts.get('interventions', 0) + int(audit['intervened'])
        label = audit['selected']
        self._counts[label] = self._counts.get(label, 0) + 1
        return selected
