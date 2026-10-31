"""FEAS-CEM-001/002: opt-in hard-feasible CEM, outside frozen controllers.

The inherited sequence checker uses the injected execution model and full swept
footprint. This experiment changes elite fitting, unlike the existing final-only
ConstrainedMPCController. Nothing registers this class in production factories.
"""
from __future__ import annotations

import time

import numpy as np

from controller.constrained_mpc import ConstrainedMPCController, ControlSafetyError
from controller.mpc import MPCController


class FeasibleCEMController(ConstrainedMPCController):
    """Fit elites only from feasible samples; validate nonconvex means separately."""

    def _solve(self, u_mean, ref, state):
        started = time.perf_counter()
        u_mean = np.asarray(u_mean, dtype=np.float64).copy()
        ref = np.asarray(ref, dtype=np.float64)
        if (u_mean.shape != (2 * self.horizon,) or ref.shape != (self.horizon, 3)
                or not np.isfinite(u_mean).all() or not np.isfinite(ref).all()
                or not np.isfinite(state.to_array()).all()):
            raise ValueError('finite reference/state and correctly sized warm start required')
        if not self.footprint_checker.pose_free(state.x, state.y, state.yaw):
            raise ControlSafetyError('current footprint violates required clearance')

        warm_start = u_mean.copy()
        std = np.tile([self.max_v * .5, self.max_omega * .8], self.horizon)
        original_elite_count = max(2, int(round(self.population * self.elite_frac)))
        best_sample = None
        best_sample_cost = np.inf
        generations = []
        for generation in range(self.iterations):
            samples = self._rng.normal(size=(self.population, 2 * self.horizon)) * std + u_mean
            samples[:, 0::2] = np.clip(samples[:, 0::2], -self.max_v, self.max_v)
            samples[:, 1::2] = np.clip(samples[:, 1::2], -self.max_omega, self.max_omega)
            costs = MPCController._batch_cost(self, samples, ref, state)
            if not np.isfinite(costs).all():
                raise ValueError('nonfinite CEM tracking cost')
            feasible = np.array([self.sequence_safe(s, state) for s in samples], dtype=bool)
            eligible = np.flatnonzero(feasible)
            # NumPy's original argsort path is retained when every sample is valid.
            ranked = eligible[np.argsort(costs[eligible])]
            elites = ranked[:original_elite_count]
            if len(elites):
                best = int(ranked[0])
                if costs[best] < best_sample_cost:
                    best_sample, best_sample_cost = samples[best].copy(), float(costs[best])
                u_mean = samples[elites].mean(axis=0)
                std = samples[elites].std(axis=0) + 1e-3
            # No feasible sample: do not fit unsafe data, but consume the fixed RNG budget.
            generations.append(dict(generation=generation, feasible_samples=int(feasible.sum()),
                rejected_samples=int((~feasible).sum()), feasible_mask=feasible.tolist(),
                elite_indices=elites.tolist()))

        mean_safe = self.sequence_safe(u_mean, state)
        mean_cost = float(MPCController._batch_cost(self, u_mean[None], ref, state)[0])
        if not np.isfinite(mean_cost):
            raise ValueError('nonfinite final mean cost')
        if mean_safe:
            label, selected, cost = 'feasible_mean', u_mean, mean_cost
        else:
            candidates = []
            if best_sample is not None:
                candidates.append(('safe_sample', best_sample, best_sample_cost))
            for label, sequence in [('shifted_warm_start', warm_start),
                                    ('zero_hold', np.zeros_like(warm_start))]:
                if self.sequence_safe(sequence, state):
                    cost = float(MPCController._batch_cost(self, sequence[None], ref, state)[0])
                    if not np.isfinite(cost):
                        raise ValueError('nonfinite fallback tracking cost')
                    candidates.append((label, sequence, cost))
            if not candidates:
                raise ControlSafetyError('no full-horizon feasible sequence, including zero hold')
            label, selected, cost = min(candidates, key=lambda candidate: candidate[2])
        # Feasible samples can have an infeasible average; recheck the actual output.
        if not self.sequence_safe(selected, state):
            raise ControlSafetyError('final sequence failed independent feasibility recheck')
        is_zero = bool(np.all(selected == 0.))
        self._audit = dict(generations=generations, final_mean_safe=bool(mean_safe),
            selected=label, selected_cost=cost, selected_horizon_safe=True,
            selected_sequence=selected.reshape(-1, 2).tolist(), is_zero_hold=is_zero,
            solve_ms=1000. * (time.perf_counter() - started))
        self._counts['checks'] = self._counts.get('checks', 0) + 1
        self._counts['rejected_samples'] = self._counts.get('rejected_samples', 0) + sum(
            g['rejected_samples'] for g in generations)
        self._counts['zero_hold'] = self._counts.get('zero_hold', 0) + int(is_zero)
        self._counts[label + '_selected'] = self._counts.get(label + '_selected', 0) + 1
        return selected.copy()
