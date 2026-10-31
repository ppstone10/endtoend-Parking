"""RUN-REFINE-002..005: isolated CEM, progress and goal-rotation switches.

Only the experiment entry point constructs this controller. Native decoding,
original cost, limits, sweep resolution and scalar final validation are retained.
"""
from __future__ import annotations

import time
import numpy as np

from controller.constrained_mpc import ControlSafetyError
from controller.mpc import MPCController
from interfaces import ControlCmd
from scripts.batched_footprint_check import BatchedSafety
from scripts.feasible_cem_mpc import FeasibleCEMController


PROFILES = {
    'fast': dict(variance=False, selection=False, progress=False, terminal_rotation=False),
    'variance': dict(variance=True, selection=False, progress=False, terminal_rotation=False),
    'selection': dict(variance=False, selection=True, progress=False, terminal_rotation=False),
    'progress': dict(variance=False, selection=False, progress=True, terminal_rotation=False),
    'terminal': dict(variance=False, selection=False, progress=False, terminal_rotation=True),
    'combined': dict(variance=True, selection=True, progress=True, terminal_rotation=True),
}


class RefinedCEMController(FeasibleCEMController):
    def __init__(self, *, profile, goal=None, terminal=None, **kwargs):
        if profile not in PROFILES:
            raise ValueError('unknown frozen refinement profile')
        super().__init__(**kwargs)
        self.profile, self.options = profile, PROFILES[profile].copy()
        self.goal, self.terminal = goal, terminal
        if self.options['terminal_rotation']:
            if goal is None or terminal is None:
                raise ValueError('terminal rotation requires original task goal/tolerances')
            if not np.isfinite([goal.x, goal.y, goal.yaw, terminal.tol_pos, terminal.tol_yaw]).all() or min(terminal.tol_pos, terminal.tol_yaw) <= 0.:
                raise ValueError('finite original goal and positive tolerances required')
        self.batch_safety = BatchedSafety(self.footprint_checker, self.vehicle_model,
                                         self.dt, self.max_v, self.max_omega)
        self._progress_audit = {}

    def reset(self):
        super().reset()
        self._progress_audit = {}

    def _advance_progress(self, pts, state):
        if not self.options['progress']:
            return super()._advance_progress(pts, state)
        previous = self._progress_idx
        if previous is None:
            # Retain the original initialization when a new prediction arrives.
            self._progress_idx = int(np.argmin((pts[:, 0]-state.x)**2+(pts[:, 1]-state.y)**2))
        else:
            for _ in range(self._catchup_window):
                pose = pts[self._progress_idx]
                position = float(np.hypot(pose[0]-state.x, pose[1]-state.y))
                heading = float(abs(np.arctan2(np.sin(pose[2]-state.yaw), np.cos(pose[2]-state.yaw))))
                if position > .15 or heading > .1 or self._progress_idx == len(pts)-1:
                    break
                self._progress_idx += 1
        pose = pts[self._progress_idx]
        self._progress_audit = dict(previous=previous, current=self._progress_idx,
            position_error=float(np.hypot(pose[0]-state.x, pose[1]-state.y)),
            yaw_error=float(abs(np.arctan2(np.sin(pose[2]-state.yaw), np.cos(pose[2]-state.yaw)))),
            position_gate=.15, yaw_gate=.1)
        return self._progress_idx

    def _solve(self, u_mean, ref, state):
        started = time.perf_counter()
        u_mean, ref = np.asarray(u_mean, dtype=np.float64).copy(), np.asarray(ref, dtype=np.float64)
        if (u_mean.shape != (2*self.horizon,) or ref.shape != (self.horizon, 3)
                or not np.isfinite(u_mean).all() or not np.isfinite(ref).all()
                or not np.isfinite(state.to_array()).all()):
            raise ValueError('finite reference/state and correctly sized warm start required')
        if not self.footprint_checker.pose_free(state.x, state.y, state.yaw):
            raise ControlSafetyError('current footprint violates required clearance')
        warm_start = u_mean.copy()
        initial_std = np.tile([self.max_v*.5, self.max_omega*.8], self.horizon)
        std = initial_std.copy()
        elite_count = max(2, int(round(self.population*self.elite_frac)))
        best_sample, best_cost, generations = None, np.inf, []
        for generation in range(self.iterations):
            before_mean, before_std = u_mean.copy(), std.copy()
            samples = self._rng.normal(size=(self.population, 2*self.horizon))*std+u_mean
            samples[:, 0::2] = np.clip(samples[:, 0::2], -self.max_v, self.max_v)
            samples[:, 1::2] = np.clip(samples[:, 1::2], -self.max_omega, self.max_omega)
            costs = MPCController._batch_cost(self, samples, ref, state)
            if not np.isfinite(costs).all():
                raise ValueError('nonfinite CEM tracking cost')
            feasible = self.batch_safety.eligible(samples, state, self.sequence_safe)
            eligible = np.flatnonzero(feasible)
            ranked = eligible[np.argsort(costs[eligible])]
            elites = ranked[:elite_count]
            guarded = bool(self.options['variance'] and len(elites) < 2)
            if len(elites):
                best = int(ranked[0])
                if costs[best] < best_cost:
                    best_sample, best_cost = samples[best].copy(), float(costs[best])
                if not guarded:
                    fitted_mean = samples[elites].mean(axis=0)
                    fitted_std = samples[elites].std(axis=0)+1e-3
                    if self.options['variance']:
                        u_mean = .1*u_mean+.9*fitted_mean
                        std = np.maximum(.1*std+.9*fitted_std, .1*initial_std)
                    else:
                        u_mean, std = fitted_mean, fitted_std
            generations.append(dict(generation=generation, feasible_samples=int(feasible.sum()),
                rejected_samples=int((~feasible).sum()), feasible_mask=feasible.tolist(),
                elite_indices=elites.tolist(), sparse_elite_guard=guarded,
                mean_before=before_mean.tolist(), std_before=before_std.tolist(),
                mean_after=u_mean.tolist(), std_after=std.tolist()))
        mean_safe = self.sequence_safe(u_mean, state)
        mean_cost = self._cost(u_mean, ref, state)
        candidates = []
        if mean_safe:
            candidates.append(('feasible_mean', u_mean, mean_cost))
        if self.options['selection'] or not mean_safe:
            if best_sample is not None:
                candidates.append(('safe_sample', best_sample, best_cost))
            if self.sequence_safe(warm_start, state):
                candidates.append(('shifted_warm_start', warm_start, self._cost(warm_start, ref, state)))
            # With selection enabled zero is a safety fallback, never a competitor
            # that defeats a valid moving candidate simply by its tracking cost.
            if (not self.options['selection'] or not candidates):
                zero = np.zeros_like(warm_start)
                if self.sequence_safe(zero, state):
                    candidates.append(('zero_hold', zero, self._cost(zero, ref, state)))
        if not candidates:
            raise ControlSafetyError('no full-horizon feasible sequence, including zero hold')
        label, selected, cost = min(candidates, key=lambda c: c[2])
        terminal_audit = dict(enabled=self.options['terminal_rotation'], attempted=False, applied=False)
        if self.options['terminal_rotation']:
            rotation, terminal_audit = self._terminal_sequence(state)
            if rotation is not None:
                terminal_audit['previous_selection'] = label
                label, selected, cost = 'terminal_rotation', rotation, self._cost(rotation, ref, state)
        # Deliberately scalar, including for pure acceleration and terminal output.
        if not self.sequence_safe(selected, state):
            raise ControlSafetyError('final sequence failed independent feasibility recheck')
        is_zero = bool(np.all(selected == 0.))
        self._audit = dict(generations=generations, final_mean_safe=bool(mean_safe),
            selected=label, selected_cost=cost, selected_horizon_safe=True,
            selected_sequence=selected.reshape(-1, 2).tolist(), is_zero_hold=is_zero,
            profile=self.profile, options=self.options.copy(), progress_gate=self._progress_audit.copy(),
            terminal_rotation=terminal_audit,
            candidate_costs=[dict(label=c[0], cost=c[2]) for c in candidates],
            solve_ms=1000.*(time.perf_counter()-started))
        for key, count in [('checks', 1), ('rejected_samples', sum(g['rejected_samples'] for g in generations)),
                           ('zero_hold', int(is_zero)), (label+'_selected', 1),
                           ('sparse_elite_guards', sum(g['sparse_elite_guard'] for g in generations)),
                           ('terminal_attempts', int(terminal_audit['attempted']))]:
            self._counts[key] = self._counts.get(key, 0)+count
        return selected.copy()

    def _cost(self, sequence, ref, state):
        cost = float(MPCController._batch_cost(self, sequence[None], ref, state)[0])
        if not np.isfinite(cost):
            raise ValueError('nonfinite candidate tracking cost')
        return cost

    def _terminal_sequence(self, state):
        pos = self.terminal.pos_err(state, self.goal)
        yaw = self.terminal.yaw_err(state, self.goal)
        audit = dict(enabled=True, attempted=False, applied=False, position_error=pos,
                     yaw_error=yaw, tol_pos=self.terminal.tol_pos, tol_yaw=self.terminal.tol_yaw)
        if pos >= self.terminal.tol_pos or yaw < self.terminal.tol_yaw:
            return None, audit
        angle = float(np.arctan2(np.sin(self.goal.yaw-state.yaw), np.cos(self.goal.yaw-state.yaw)))
        audit.update(attempted=True, signed_rotation=angle)
        end = np.array([state.x, state.y, state.yaw+angle])
        full_safe = self.footprint_checker.swept_segment_free(state.to_array()[:3], end)
        audit['full_rotation_margin_safe'] = bool(full_safe)
        if not full_safe:
            return None, audit
        sequence, current = np.zeros((self.horizon, 2)), state
        for k in range(self.horizon):
            error = float(np.arctan2(np.sin(self.goal.yaw-current.yaw), np.cos(self.goal.yaw-current.yaw)))
            omega = float(np.clip(error/self.dt, -self.max_omega, self.max_omega)) if abs(error) >= 1e-12 else 0.
            sequence[k, 1] = omega
            current = self.vehicle_model.step(current, ControlCmd(0., omega), self.dt)
        if not self.sequence_safe(sequence, state):
            audit['horizon_rejected'] = True
            return None, audit
        audit['applied'] = True
        return sequence.reshape(-1), audit
