"""原车位位置容差内，以完整余量证书接管纯原地朝向。"""
import numpy as np
from interfaces import ControlCmd
from .constrained_mpc import ConstrainedMPCController


class TerminalCaptureMPCController(ConstrainedMPCController):
    def __init__(self, *, goal, terminal, **kwargs):
        super().__init__(**kwargs)
        values = [goal.x, goal.y, goal.yaw, terminal.tol_pos, terminal.tol_yaw]
        if not np.isfinite(values).all() or min(terminal.tol_pos, terminal.tol_yaw) <= 0:
            raise ValueError('目标和原车位容差必须有限且容差为正')
        self.goal, self.terminal = goal, terminal

    def _solve(self, u_mean, ref, state):
        original = super()._solve(u_mean, ref, state)
        if (self.terminal.pos_err(state, self.goal) >= self.terminal.tol_pos
                or self.terminal.yaw_err(state, self.goal) < self.terminal.tol_yaw):
            return original
        angle = float(np.arctan2(np.sin(self.goal.yaw-state.yaw), np.cos(self.goal.yaw-state.yaw)))
        end = np.array([state.x, state.y, state.yaw+angle])
        if not self.footprint_checker.swept_segment_free(state.to_array()[:3], end):
            return original
        sequence, current = np.zeros((self.horizon, 2)), state
        for step in range(self.horizon):
            error = float(np.arctan2(np.sin(self.goal.yaw-current.yaw), np.cos(self.goal.yaw-current.yaw)))
            omega = float(np.clip(error/self.dt, -self.max_omega, self.max_omega)) if abs(error) >= 1e-12 else 0.
            sequence[step, 1] = omega
            current = self.vehicle_model.step(current, ControlCmd(0., omega), self.dt)
        if not self.sequence_safe(sequence, state):
            return original
        selected = sequence.reshape(-1)
        cost = float(self._batch_cost(selected[None, :], ref, state)[0])
        if not np.isfinite(cost):
            return original
        previous = self._audit.copy()
        self._counts[previous['selected']] -= 1
        self._counts['terminal_rotation'] = self._counts.get('terminal_rotation', 0) + 1
        self._counts['interventions'] += int(not previous['intervened'])
        self._audit.update(selected='terminal_rotation', selected_sequence=sequence.tolist(),
            selected_cost=cost, intervened=True, previous_selection=previous['selected'],
            terminal_certificate={'position_error': self.terminal.pos_err(state, self.goal),
                'yaw_error': self.terminal.yaw_err(state, self.goal), 'signed_rotation': angle,
                'tol_pos': self.terminal.tol_pos, 'tol_yaw': self.terminal.tol_yaw,
                'full_rotation_margin_safe': True})
        return selected
