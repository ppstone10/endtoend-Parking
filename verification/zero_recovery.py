"""只在原安全选择退为全零时，补查同次CEM未覆盖的已有样本。"""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np

from verification import terminal_capture
from verification.terminal_capture import TerminalCaptureMPC
from verification.reference_retention import retention_scope


class ZeroRecoveryMPC(TerminalCaptureMPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._recovery_bank = None

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
        self.last_safety_audit.update(zero_recovery_context=context,zero_recovery_checked=0,zero_recovery_rank=None)
        if not bank:
            return original
        samples=np.concatenate([s for s,_ in bank]);costs=np.concatenate([c for _,c in bank])
        order=np.argsort(costs,kind='stable')
        for rank in range(self.max_candidate_checks,len(order)):
            ordinal=int(order[rank])
            if not np.isfinite(costs[ordinal]) or costs[ordinal]>=previous['selected_cost']:
                break
            self.last_safety_audit['zero_recovery_checked']+=1
            if self.sequence_safe(samples[ordinal],state):
                selected=samples[ordinal].copy()
                self.last_safety_audit.update(selected='expanded_safe_sample',selected_cost=float(costs[ordinal]),
                                               selected_sequence=selected.reshape(-1,2).tolist(),
                                               selected_horizon_safe=True,zero_recovery_rank=rank+1)
                return selected
        return original


@contextmanager
def zero_recovery_scope(output,vehicle,expected,budget):
    original=terminal_capture.TerminalCaptureMPC
    terminal_capture.TerminalCaptureMPC=ZeroRecoveryMPC
    try:
        with retention_scope(output,vehicle,expected,budget) as result:
            yield result
    finally:
        terminal_capture.TerminalCaptureMPC=original
