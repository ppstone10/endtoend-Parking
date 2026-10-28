"""仅在原安全零解且当前满足规划余量时，补查完整余量安全的已有样本。"""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np

from planner import RectangleFootprintCollisionChecker
from interfaces import ControlCmd
from verification import terminal_capture
from verification.terminal_capture import TerminalCaptureMPC
from verification.reference_retention import retention_scope


class MarginRecoveryMPC(TerminalCaptureMPC):
    def __init__(self, *, margin_checker=None, **kwargs):
        super().__init__(**kwargs)
        self._recovery_bank = None
        contact=self.contact_checker
        self.margin_checker=margin_checker if margin_checker is not None else RectangleFootprintCollisionChecker(
            contact.env,vehicle_length=2*contact.half_length,vehicle_width=2*contact.half_width,
            collision_margin=.2,resolution=contact.resolution)

    def margin_sequence_safe(self, sequence, state):
        if not self.margin_checker.pose_free(state.x,state.y,state.yaw):return False
        for v,omega in np.asarray(sequence).reshape(self.horizon,2):
            after=self.vehicle_model.step(state,ControlCmd(float(v),float(omega)),self.dt)
            if not self.margin_checker.swept_segment_free(state.to_array()[:3],after.to_array()[:3]):return False
            state=after
        return True

    def _batch_cost(self, samples, ref, state):
        costs = super()._batch_cost(samples, ref, state)
        if self._recovery_bank is not None and len(samples)>1:
            self._recovery_bank.append((samples.copy(),costs.copy()))
        return costs

    def _solve(self, u_mean, ref, state):
        rng_before = deepcopy(self._rng.bit_generator.state)
        self._recovery_bank = []
        try:
            original = super()._solve(u_mean,ref,state)
            bank = self._recovery_bank
        finally:
            self._recovery_bank = None
        if not np.all(original==0) or not self.last_safety_audit['intervened']:
            return original
        previous = dict(self.last_safety_audit)
        context = {'rng_state':rng_before,'shifted_mean':np.asarray(u_mean).tolist(),
                   'reference_window':np.asarray(ref).tolist(),'state':state.to_array().tolist(),
                   'goal':[self.goal.x,self.goal.y,self.goal.yaw],
                   'tol_pos':self.terminal.tol_pos,'tol_yaw':self.terminal.tol_yaw,
                   'original_selection':previous['selected'],'original_cost':previous['selected_cost']}
        self.last_safety_audit.update(margin_recovery_context=context,margin_recovery_checked=0,margin_recovery_rank=None,
                                      margin_recovery_skip=None)
        if not self.margin_checker.pose_free(state.x,state.y,state.yaw):
            self.last_safety_audit['margin_recovery_skip']='already_inside_margin'
            return original
        if not bank:
            return original
        samples=np.concatenate([s for s,_ in bank]);costs=np.concatenate([c for _,c in bank])
        order=np.argsort(costs,kind='stable')
        for rank in range(self.max_candidate_checks,len(order)):
            ordinal=int(order[rank])
            if not np.isfinite(costs[ordinal]) or costs[ordinal]>=previous['selected_cost']:
                break
            self.last_safety_audit['margin_recovery_checked']+=1
            if self.sequence_safe(samples[ordinal],state) and self.margin_sequence_safe(samples[ordinal],state):
                selected=samples[ordinal].copy()
                self.last_safety_audit.update(selected='expanded_margin_sample',selected_cost=float(costs[ordinal]),
                                               selected_sequence=selected.reshape(-1,2).tolist(),
                                               selected_horizon_safe=True,margin_recovery_rank=rank+1)
                return selected
        return original


@contextmanager
def margin_recovery_scope(output,vehicle,expected,budget):
    original=terminal_capture.TerminalCaptureMPC
    terminal_capture.TerminalCaptureMPC=MarginRecoveryMPC
    try:
        with retention_scope(output,vehicle,expected,budget) as result:
            yield result
    finally:
        terminal_capture.TerminalCaptureMPC=original
