"""仅在原位置容差内，用完整扫掠证书接管终端原地朝向。"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np

from interfaces import ControlCmd
from verification.safety_candidate import FootprintConstrainedMPC
from verification import candidate_execution
from experiments import validation_matrix as matrix


class TerminalCaptureMPC(FootprintConstrainedMPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goal = None
        self.terminal = None

    def bind_terminal(self, goal, terminal):
        if not np.isfinite([goal.x, goal.y, goal.yaw, terminal.tol_pos, terminal.tol_yaw]).all():
            raise ValueError('目标与终端容差必须有限')
        if terminal.tol_pos <= 0 or terminal.tol_yaw <= 0:
            raise ValueError('终端容差必须为正')
        self.goal, self.terminal = goal, terminal

    def _solve(self, u_mean, ref, state):
        if self.goal is None or self.terminal is None:
            raise ValueError('终端候选必须绑定同一引擎目标与验收器')
        legacy = super()._solve(u_mean, ref, state)
        if (self.terminal.pos_err(state, self.goal) >= self.terminal.tol_pos
                or self.terminal.yaw_err(state, self.goal) < self.terminal.tol_yaw):
            return legacy
        error = float(np.arctan2(np.sin(self.goal.yaw - state.yaw), np.cos(self.goal.yaw - state.yaw)))
        end = np.array([state.x, state.y, state.yaw + error])
        # 全角程证书不能由首步安全替代，尤其长车体转向时四角会扫过障碍。
        if not self.contact_checker.swept_segment_free(state.to_array()[:3], end):
            return legacy
        sequence = np.zeros((self.horizon, 2))
        current = state
        for step in range(self.horizon):
            remaining = float(np.arctan2(np.sin(self.goal.yaw - current.yaw), np.cos(self.goal.yaw - current.yaw)))
            omega = float(np.clip(remaining / self.dt, -self.max_omega, self.max_omega))
            if abs(remaining) < 1e-12:
                omega = 0.
            sequence[step, 1] = omega
            current = self.vehicle_model.step(current, ControlCmd(0., omega), self.dt)
        if not self.sequence_safe(sequence, state):
            return legacy
        selected = sequence.reshape(-1)
        parent = dict(self.last_safety_audit)
        self.last_safety_audit.update(
            selected='terminal_rotation', intervened=True, selected_horizon_safe=True,
            selected_sequence=sequence.tolist(), selected_cost=float(self._batch_cost(selected[None, :], ref, state)[0]),
            previous_selection=parent['selected'], terminal_certificate={
                'position_error': self.terminal.pos_err(state, self.goal),
                'yaw_error': self.terminal.yaw_err(state, self.goal),
                'signed_rotation': error, 'full_rotation_contact_safe': True,
                'goal': [self.goal.x, self.goal.y, self.goal.yaw],
                'tol_pos': self.terminal.tol_pos, 'tol_yaw': self.terminal.tol_yaw})
        return selected


@contextmanager
def terminal_scope(output, vehicle, expected, budget):
    """复用冻结录制器，只在当前显式上下文绑定目标和新控制类。"""
    original_engine = matrix.ClosedLoopEngine
    original_controller = candidate_execution.FootprintConstrainedMPC

    class GoalBindingEngine(original_engine):
        def run(self, start, goal):
            self.mpc.bind_terminal(goal, self.terminal)
            return super().run(start, goal)

    matrix.ClosedLoopEngine = GoalBindingEngine
    candidate_execution.FootprintConstrainedMPC = TerminalCaptureMPC
    try:
        with candidate_execution.candidate_scope('control', output, vehicle, expected, budget) as rows:
            yield rows
    finally:
        matrix.ClosedLoopEngine = original_engine
        candidate_execution.FootprintConstrainedMPC = original_controller
