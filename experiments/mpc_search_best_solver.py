"""仅供离线试验：在当前CEM采样中保留最佳序列，不直接返回旧热启动。"""
from __future__ import annotations

import numpy as np

from controller import MPCController
from interfaces import VehicleState


class SearchBestMPC(MPCController):
    """原搜索/原代价，比较本次最佳样本与原最终均值；不保证安全。"""

    def _solve(self, u_mean: np.ndarray, ref: np.ndarray, state: VehicleState) -> np.ndarray:
        if (self.iterations < 1 or u_mean.shape != (2*self.horizon,) or ref.shape != (self.horizon,3)
                or not np.isfinite(u_mean).all() or not np.isfinite(ref).all()
                or not np.isfinite(state.to_array()).all()):
            raise ValueError('当前搜索求解输入无效')
        n = 2*self.horizon
        std = np.tile([self.max_v*.5,self.max_omega*.8],self.horizon)
        n_elite = max(2,int(round(self.population*self.elite_frac)))
        best_cost = float('inf')
        best = None
        best_iteration = best_index = None
        # 采样、限幅、排序及拟合与原MPC逐代一致；仅旁路记录本次最佳样本。
        for iteration in range(self.iterations):
            samples = self._rng.normal(size=(self.population,n))*std+u_mean
            samples[:,0::2] = np.clip(samples[:,0::2],-self.max_v,self.max_v)
            samples[:,1::2] = np.clip(samples[:,1::2],-self.max_omega,self.max_omega)
            costs = self._batch_cost(samples,ref,state)
            if not np.isfinite(samples).all() or not np.isfinite(costs).all():
                raise ValueError('当前搜索样本或代价不是有限值')
            index = int(np.argmin(costs))
            if float(costs[index]) < best_cost:
                best_cost = float(costs[index])
                best = samples[index].copy()
                best_iteration,best_index = iteration,index
            elite_idx = np.argsort(costs)[:n_elite]
            elites = samples[elite_idx]
            u_mean = elites.mean(axis=0)
            std = elites.std(axis=0)+1e-3
        mean_cost = float(self._batch_cost(u_mean[None],ref,state)[0])
        if not np.isfinite(mean_cost) or not np.isfinite(u_mean).all():
            raise ValueError('最终均值或代价不是有限值')
        use_sample = best_cost < mean_cost
        self.last_search_audit = {
            'selected':'search_sample' if use_sample else 'legacy_mean',
            'legacy_mean_cost':mean_cost, 'best_sample_cost':best_cost,
            'selected_cost':best_cost if use_sample else mean_cost,
            'best_sample_iteration':best_iteration, 'best_sample_index':best_index,
            'extra_cost_sequences':1,
        }
        return best.copy() if use_sample else u_mean.copy()
