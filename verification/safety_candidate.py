"""原CEM候选的完整矩形扫掠约束；只在最终均值危险时改选。"""
from __future__ import annotations

import numpy as np
from controller import MPCController
from interfaces import ControlCmd
from sim import DifferentialDriveModel


class FootprintConstrainedMPC(MPCController):
    """显式离线候选；碰撞约束独立于跟踪成本，没有专家/目标偏置。

    只检验预测时域，不声称递归可行或闭环可到达。zero_hold是安全停车，
    由原生终止判据决定是否到达，不能当作成功替代。
    """
    def __init__(self, *, contact_checker, max_candidate_checks=16, **kwargs):
        super().__init__(**kwargs)
        if max_candidate_checks < 1:
            raise ValueError('max_candidate_checks必须为正')
        self.contact_checker = contact_checker
        self.max_candidate_checks = int(max_candidate_checks)
        self.vehicle_model = DifferentialDriveModel(max_v=self.max_v, max_omega=self.max_omega)
        self._sample_bank = None
        self.last_safety_audit = {}

    def _batch_cost(self, samples, ref, state):
        costs = super()._batch_cost(samples, ref, state)
        if self._sample_bank is not None:
            self._sample_bank.append((samples.copy(), costs.copy()))
        return costs

    def sequence_safe(self, sequence, state):
        sequence = np.asarray(sequence).reshape(-1, 2)
        if sequence.shape != (self.horizon, 2) or not np.isfinite(sequence).all():
            return False
        if not self.contact_checker.pose_free(state.x, state.y, state.yaw):
            return False
        for v, omega in sequence:
            after = self.vehicle_model.step(state, ControlCmd(float(v), float(omega)), self.dt)
            if not self.contact_checker.swept_segment_free(state.to_array()[:3], after.to_array()[:3]):
                return False
            state = after
        return True

    def _solve(self, u_mean, ref, state):
        if not self.contact_checker.pose_free(state.x, state.y, state.yaw):
            raise ValueError('初始车体已接触，不能伪装成安全停车')
        self._sample_bank = []
        try:
            legacy = super()._solve(u_mean, ref, state)
            bank = self._sample_bank
        finally:
            self._sample_bank = None
        legacy_cost = float(super()._batch_cost(legacy[None, :], ref, state)[0])
        legacy_safe = self.sequence_safe(legacy, state)
        audit = {'legacy_safe': legacy_safe, 'legacy_cost': legacy_cost, 'checked_samples': 0,
                 'sampled_sequences': sum(len(c) for _, c in bank), 'selected': 'legacy_mean',
                 'selected_cost': legacy_cost, 'intervened': not legacy_safe,
                 'selected_horizon_safe': legacy_safe}
        if legacy_safe:
            self.last_safety_audit = audit
            return legacy

        candidates = []
        if bank:
            samples = np.concatenate([s for s, _ in bank])
            costs = np.concatenate([c for _, c in bank])
            order = np.argsort(costs, kind='stable')[:self.max_candidate_checks]
            # 代价稳定排序后找到首个安全样本即可；不额外采样或变更原随机流。
            for index in order:
                audit['checked_samples'] += 1
                if np.isfinite(costs[index]) and self.sequence_safe(samples[index], state):
                    candidates.append(('safe_sample', samples[index].copy(), float(costs[index])))
                    break
        for label, sequence in [('shifted_warm_start', u_mean), ('zero_hold', np.zeros_like(u_mean))]:
            if self.sequence_safe(sequence, state):
                cost = float(super()._batch_cost(sequence[None, :], ref, state)[0])
                candidates.append((label, sequence.copy(), cost))
        if not candidates:
            raise ValueError('未找到完整时域安全动作，包括原地保持；停止候选运行')
        label, selected, cost = min(candidates, key=lambda item: item[2])
        audit.update(selected=label, selected_cost=cost,
                     selected_sequence=selected.reshape(-1, 2).tolist(), selected_horizon_safe=True)
        self.last_safety_audit = audit
        return selected
