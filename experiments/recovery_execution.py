"""实验专用上下文分叉；不改变生产engine和控制器。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np

from runtime.engine import ClosedLoopEngine
from runtime.sources import NetworkSource, ReplanningExpertSource


@dataclass
class ContinuationContext:
    step: int
    state: object
    mpc: object
    pipeline: object


class MarginAuditEngine(ClosedLoopEngine):
    """只记录原0.2m规划检查器的真实转移结果，不干预动作。"""
    def __init__(self, *args, planning_checker, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_checker = planning_checker
        self.margin_violations = []

    def _check_collision(self, previous, state):
        self.margin_violations.append(not self.planning_checker.swept_segment_free(previous.to_array()[:3], state.to_array()[:3]))
        return super()._check_collision(previous, state)


class ContextCollector(MarginAuditEngine):
    """每次推理前捕获上下文；只保留本任务的少量重规划快照。"""
    def run(self, start, goal):
        self.contexts = []
        self.margin_violations = []
        predict = self.source.next_trajectory

        def capture(state):
            step = len(self.margin_violations)
            if self.planning_checker.pose_free(state.x, state.y, state.yaw):
                self.contexts.append(ContinuationContext(step, deepcopy(state), deepcopy(self.mpc), deepcopy(self.source.sensor_pipeline)))
            return predict(state)

        self.source.next_trajectory = capture
        try:
            return super().run(start, goal)
        finally:
            self.source.next_trajectory = predict


def select_context(contexts, margin_violations, success):
    first_loss = next((i for i, hit in enumerate(margin_violations) if hit), None)
    if first_loss is not None:
        prior = [c for c in contexts if c.step <= first_loss]
        return (prior[-1], 'before_first_margin_loss') if prior else (None, 'missing_safe_context')
    if success and contexts:
        middle = len(margin_violations) / 2
        return min(contexts, key=lambda c: abs(c.step - middle)), 'success_control'
    return None, 'no_margin_loss_but_not_successful'


def execute_branch(context, mode, model, planner, goal, engine_kwargs, *, max_steps, replan_every, progress=None):
    """沿用原engine，但只在私有MPC副本上禁止起跑reset。"""
    if mode not in {'network', 'expert'} or max_steps <= 0 or context.step % replan_every:
        raise ValueError('分支模式/步数/重规划相位无效')
    mpc = deepcopy(context.mpc)
    pipeline = deepcopy(context.pipeline)
    source = NetworkSource(pipeline, model) if mode == 'network' else ReplanningExpertSource(deepcopy(planner))
    if progress is not None:
        predict = source.next_trajectory
        calls = [0]

        def report_prediction(state):
            calls[0] += 1
            tick = time.perf_counter()
            progress(calls[0], 'BEGIN', None)
            result = predict(state)
            progress(calls[0], 'END', time.perf_counter() - tick)
            return result

        source.next_trajectory = report_prediction
    engine = MarginAuditEngine(mpc=mpc, source=source, planning_checker=planner._collision_checker,
                               replan_every=replan_every, max_steps=max_steps, **engine_kwargs)
    reset = mpc.reset
    mpc.reset = lambda: None
    try:
        result = engine.run(deepcopy(context.state), goal)
    finally:
        mpc.reset = reset
    return result, engine.margin_violations


def record_arrays(result):
    states = np.asarray([s.to_array() for s in result.record.states]).reshape(-1, 5)
    commands = np.asarray([[c.v, c.omega] for c in result.record.cmds]).reshape(-1, 2)
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError('分支状态或命令出现非有限值')
    return states, commands


def verify_network_suffix(original, continued, step, *, prefix_only=False):
    states, commands = record_arrays(original)
    actual_states, actual_commands = record_arrays(continued)
    expected_states, expected_commands = states[step:], commands[step:]
    if prefix_only:
        expected_states, expected_commands = expected_states[:len(actual_states)], expected_commands[:len(actual_commands)]
    if expected_states.shape != actual_states.shape or expected_commands.shape != actual_commands.shape:
        raise ValueError('网络分支长度与原回合后缀不一致')
    if not np.allclose(expected_states, actual_states, atol=1e-9, rtol=0) or not np.allclose(expected_commands, actual_commands, atol=1e-9, rtol=0):
        raise ValueError('网络分支未精确复现原回合后缀')
    return {'matched': True, 'steps': len(actual_states), 'prefix_only': prefix_only,
            'max_state_error': float(np.abs(expected_states - actual_states).max()) if len(actual_states) else 0.,
            'max_command_error': float(np.abs(expected_commands - actual_commands).max()) if len(actual_commands) else 0.}


def execution_summary(result, margin_violations, *, truncated=False):
    return {'success': result.success, 'collision': result.collision, 'failure': result.failure,
            'steps': result.steps, 'final_pos_err': result.final_pos_err, 'final_yaw_err': result.final_yaw_err,
            'margin_violation_steps': int(sum(margin_violations)),
            'first_margin_violation_step': next((i for i, v in enumerate(margin_violations) if v), None),
            'reached_with_margin': bool(result.success and not any(margin_violations)),
            'truncated_preflight': truncated}
