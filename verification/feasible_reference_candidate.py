"""合法消费窗口内的参考控制候选；非法窗口保留完整 LT25 选择。"""
from contextlib import contextmanager
import numpy as np

from verification import terminal_capture, reference_teacher
from verification.candidate_execution import connected_safe
from verification.planning_margin import PlanningMarginMPC
from verification.reference_candidate import reference_control_sequence
from verification.reference_retention import retention_scope


class FeasibleReferenceCandidateMPC(PlanningMarginMPC):
    def _solve(self, u_mean, ref, state):
        original = super()._solve(u_mean, ref, state)
        if not np.all(original == 0) or not self.last_safety_audit['intervened']:
            return original
        window_safe = connected_safe(self.contact_checker, state.to_array(), ref)
        self.last_safety_audit['reference_window_margin_safe'] = window_safe
        if not window_safe:
            self.last_safety_audit['reference_candidate_skipped'] = 'unsafe_consumed_window'
            return original
        candidate = reference_control_sequence(ref, state, self.vehicle_model,
                                               self.dt, self.max_v, self.max_omega)
        safe = self.sequence_safe(candidate, state)
        raw_cost = float(self._batch_cost(candidate[None, :], ref, state)[0]) if safe else None
        cost = raw_cost if raw_cost is not None and np.isfinite(raw_cost) else None
        previous_cost = self.last_safety_audit['selected_cost']
        accepted = bool(safe and cost is not None and cost < previous_cost)
        self.last_safety_audit['reference_candidate'] = {
            'previous_selection': self.last_safety_audit['selected'],
            'previous_cost': previous_cost, 'candidate': candidate.reshape(-1, 2).tolist(),
            'horizon_safe': safe, 'candidate_cost': cost, 'accepted': accepted,
            'reason': ('accepted' if accepted else 'unsafe' if not safe
                       else 'nonfinite_cost' if cost is None else 'not_better')}
        if accepted:
            self.last_safety_audit.update(selected='reference_inverse_candidate', selected_cost=cost,
                                          selected_sequence=candidate.reshape(-1, 2).tolist(),
                                          selected_horizon_safe=True)
            return candidate
        return original


@contextmanager
def feasible_reference_scope(output, vehicle, expected, budget):
    original = terminal_capture.TerminalCaptureMPC
    terminal_capture.TerminalCaptureMPC = FeasibleReferenceCandidateMPC
    try:
        with retention_scope(output, vehicle, expected, budget) as result:
            yield result
    finally:
        terminal_capture.TerminalCaptureMPC = original


@contextmanager
def teacher_controller_scope():
    original = reference_teacher.ReferenceCandidateMPC
    reference_teacher.ReferenceCandidateMPC = FeasibleReferenceCandidateMPC
    try:
        yield
    finally:
        reference_teacher.ReferenceCandidateMPC = original
