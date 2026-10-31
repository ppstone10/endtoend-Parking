"""MPC-RESCUE-001/002: validate a fixed motion bank only after a full zero hold.

No search sample or RNG draw is added. A triggered call performs eight extra
scalar H10 candidate checks, disclosed separately from the frozen 80x5 search.
"""
from __future__ import annotations
import time
import numpy as np
from controller.mpc import MPCController
from controller.constrained_mpc import ControlSafetyError
from scripts.ungated_refinement_mpc import UngatedRefinedCEMController, OPTIONS

PROFILE = 'zero_rescue'
SPEEDS = (-2., -1., -.5, -.2, .2, .5, 1., 2.)


def motion_bank(horizon):
    bank = np.zeros((len(SPEEDS), horizon, 2), dtype=np.float64)
    bank[:, :, 0] = np.asarray(SPEEDS)[:, None]
    return bank.reshape(len(SPEEDS), 2*horizon)


class ZeroHoldMotionRescueController(UngatedRefinedCEMController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.profile = PROFILE

    def _solve(self, u_mean, ref, state):
        started = time.perf_counter()
        base = super()._solve(u_mean, ref, state)
        base_label, base_cost = self._audit['selected'], self._audit['selected_cost']
        attempted = bool(np.all(base == 0.))
        rescue = dict(attempted=attempted, intervened=False, candidate_count=0,
            base_selected=base_label, base_cost=base_cost, base_sequence=base.reshape(-1, 2).tolist(),
            speeds=[], safe_mask=[], costs=[], selected_index=None)
        selected = base
        if attempted:
            bank = motion_bank(self.horizon)
            # Deliberately independent of the parent's batched candidate kernel.
            safe = np.array([self.sequence_safe(sequence, state) for sequence in bank], dtype=bool)
            costs = MPCController._batch_cost(self, bank, ref, state)
            if not np.isfinite(costs).all():
                raise ValueError('nonfinite fixed-motion candidate cost')
            rescue.update(candidate_count=len(bank), speeds=list(SPEEDS),
                          safe_mask=safe.tolist(), costs=costs.tolist())
            eligible = np.flatnonzero(safe)
            if len(eligible):
                winner = int(eligible[np.argmin(costs[eligible])])
                if float(costs[winner]) < base_cost:
                    selected = bank[winner].copy()
                    if not self.sequence_safe(selected, state):
                        raise ControlSafetyError('fixed-motion final H10 recheck failed')
                    rescue.update(intervened=True, selected_index=winner)
                    self._audit.update(selected='primitive_rescue', selected_cost=float(costs[winner]),
                        selected_sequence=selected.reshape(-1, 2).tolist(), is_zero_hold=False,
                        selected_horizon_safe=True)
                    self._counts[base_label+'_selected'] -= 1
                    self._counts['zero_hold'] -= 1
                    self._counts['primitive_rescue_selected'] = self._counts.get('primitive_rescue_selected', 0)+1
        for key, count in [('rescue_attempts', int(attempted)),
                           ('rescue_candidates_checked', rescue['candidate_count']),
                           ('rescue_safe_candidates', sum(rescue['safe_mask'])),
                           ('rescue_interventions', int(rescue['intervened']))]:
            self._counts[key] = self._counts.get(key, 0)+count
        self._audit.update(rescue=rescue, solve_ms=1000.*(time.perf_counter()-started))
        return selected.copy()
