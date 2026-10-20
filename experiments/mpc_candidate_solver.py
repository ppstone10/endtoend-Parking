"""离线候选保留试验；默认controller及生产入口不引用本模块。"""
from __future__ import annotations

import numpy as np

from controller import MPCController
from interfaces import VehicleState


class CandidatePreservingMPC(MPCController):
    """保留原CEM搜索，仅对最终均值、移位热启动、零序列按原代价择优。

    不读取障碍、目标地图或安全检查器；低代价不是安全或到达保证。
    """

    def _solve(self, u_mean: np.ndarray, ref: np.ndarray, state: VehicleState) -> np.ndarray:
        if (u_mean.shape != (2*self.horizon,) or ref.shape != (self.horizon, 3)
                or not np.isfinite(u_mean).all() or not np.isfinite(ref).all()
                or not np.isfinite(state.to_array()).all()):
            raise ValueError('候选求解输入形状或有限值无效')
        warm = u_mean.copy()
        warm[::2] = np.clip(warm[::2], -self.max_v, self.max_v)
        warm[1::2] = np.clip(warm[1::2], -self.max_omega, self.max_omega)
        # 搜索仍使用原输入，不改变随机采样、精英拟合或RNG消耗。
        legacy = super()._solve(u_mean, ref, state)
        candidates = np.stack([legacy, warm, np.zeros_like(warm)])
        costs = self._batch_cost(candidates, ref, state)
        if not np.isfinite(candidates).all() or not np.isfinite(costs).all():
            raise ValueError('候选解或代价不是有限值')
        selected = int(np.argmin(costs))
        self.last_candidate_audit = {
            'selected': ('legacy_mean', 'shifted_warm_start', 'zero_sequence')[selected],
            'candidate_costs': costs.tolist(), 'selected_cost': float(costs[selected]),
            'extra_cost_sequences': 3,
        }
        return candidates[selected].copy()
