"""仅在原安全零解时，补充由实际消费参考推导的可行控制。"""
from contextlib import contextmanager
import numpy as np

from interfaces import ControlCmd
from verification import terminal_capture
from verification.planning_margin import PlanningMarginMPC
from verification.reference_retention import retention_scope


def reference_control_sequence(window, state, model, dt, max_v, max_omega):
    """按原差分模型递推投影，生成一个受原执行上限约束的序列。"""
    points = np.asarray(window, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or not np.isfinite(state.to_array()).all()
            or not np.isfinite([dt, max_v, max_omega]).all()
            or min(dt, max_v, max_omega) <= 0):
        raise ValueError('非法参考窗口、状态或控制上限')
    controls = []
    current = state
    for target in points:
        heading = [np.cos(current.yaw), np.sin(current.yaw)]
        speed = float(np.clip(np.dot(target[:2] - [current.x, current.y], heading) / dt,
                              -max_v, max_v))
        error = np.arctan2(np.sin(target[2] - current.yaw), np.cos(target[2] - current.yaw))
        turn = float(np.clip(error / dt, -max_omega, max_omega))
        controls.append([speed, turn])
        current = model.step(current, ControlCmd(speed, turn), dt)
    return np.asarray(controls, dtype=np.float64).reshape(-1)


class ReferenceCandidateMPC(PlanningMarginMPC):
    def _solve(self, u_mean, ref, state):
        original = super()._solve(u_mean, ref, state)
        if not np.all(original == 0) or not self.last_safety_audit['intervened']:
            return original
        candidate = reference_control_sequence(ref, state, self.vehicle_model,
                                               self.dt, self.max_v, self.max_omega)
        safe = self.sequence_safe(candidate, state)
        raw_cost = float(self._batch_cost(candidate[None, :], ref, state)[0]) if safe else None
        cost = raw_cost if raw_cost is not None and np.isfinite(raw_cost) else None
        previous_cost = self.last_safety_audit['selected_cost']
        accepted = bool(safe and cost is not None and cost < previous_cost)
        audit = {'previous_selection': self.last_safety_audit['selected'],
                 'previous_cost': previous_cost, 'candidate': candidate.reshape(-1, 2).tolist(),
                 'horizon_safe': safe, 'candidate_cost': cost, 'accepted': accepted,
                 'reason': ('accepted' if accepted else 'unsafe' if not safe
                            else 'nonfinite_cost' if cost is None else 'not_better')}
        self.last_safety_audit['reference_candidate'] = audit
        if accepted:
            self.last_safety_audit.update(selected='reference_inverse_candidate', selected_cost=cost,
                                          selected_sequence=candidate.reshape(-1, 2).tolist(),
                                          selected_horizon_safe=True)
            return candidate
        return original


@contextmanager
def reference_candidate_scope(output, vehicle, expected, budget):
    original = terminal_capture.TerminalCaptureMPC
    terminal_capture.TerminalCaptureMPC = ReferenceCandidateMPC
    try:
        with retention_scope(output, vehicle, expected, budget) as result:
            yield result
    finally:
        terminal_capture.TerminalCaptureMPC = original
