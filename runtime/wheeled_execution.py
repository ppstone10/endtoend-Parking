"""保留转角的独立轮式执行；不将旧五维状态解释为轮式状态。"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from controller.wheeled_mpc import WheeledControlError
from interfaces.wheeled import WheeledVehicleState


class WheeledExecutor:
    def __init__(self, controller, model):
        if controller.model.config.identity != model.config.identity:
            raise ValueError("预测与执行车型身份不一致")
        self.controller, self.model = controller, model
        self.reset()

    def reset(self):
        self.controller.reset()
        self.last_command = None
        self.max_prediction_error = 0.

    def propose(self, state, trajectory, dt):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0 or abs(dt-self.controller.dt) > 1e-12:
            raise ValueError("执行器与控制器dt不一致")
        command = self.controller.compute(trajectory, state)
        proposed = self.model.step(state, command, dt)
        error = float(np.abs(proposed.to_array()-self.controller.last_prediction[0]).max())
        if error > 1e-12:
            raise RuntimeError("轮式首步预测与执行不一致")
        self.max_prediction_error = max(self.max_prediction_error, error)
        self.last_command = command
        return proposed


class WheeledClosedLoopEngine:
    def __init__(self, executor, *, max_steps=600, replan_every=0,
                 tol_pos=.3, tol_yaw=math.radians(10), transition_checker=None, max_seconds=None):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps必须为正整数")
        if isinstance(replan_every, bool) or not isinstance(replan_every, int) or replan_every < 0:
            raise ValueError("replan_every必须为非负整数，0表示固定参考")
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (tol_pos, tol_yaw)):
            raise ValueError("到达容差必须有限且为正")
        self.executor, self.max_steps, self.replan_every = executor, max_steps, replan_every
        self.tol_pos, self.tol_yaw, self.transition_checker = tol_pos, tol_yaw, transition_checker
        if max_seconds is not None and (isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("闭环时间预算必须有限且为正")
        self.max_seconds = max_seconds

    def run(self, start, goal, source):
        if not isinstance(start, WheeledVehicleState):
            raise TypeError("轮式闭环要求完整转角状态")
        self.executor.model.yaw_rate(start)
        if not all(math.isfinite(value) for value in (goal.x, goal.y, goal.yaw)):
            raise ValueError("轮式闭环目标必须有限")
        self.executor.reset()
        state, trajectory = start, None
        states, controls, timings = [state.to_array().tolist()], [], []
        started = perf_counter()
        failure, detail = "timeout", None
        for step in range(self.max_steps):
            pos_error = math.hypot(state.x-goal.x, state.y-goal.y)
            yaw_error = abs(math.atan2(math.sin(state.yaw-goal.yaw), math.cos(state.yaw-goal.yaw)))
            if step == 0 and self.transition_checker is not None and not self.transition_checker(state, state):
                failure = "initial_collision"
                break
            if pos_error < self.tol_pos and yaw_error < self.tol_yaw:
                failure = None
                break
            if self.max_seconds is not None and perf_counter()-started >= self.max_seconds:
                failure = "budget_exhausted"
                break
            if trajectory is None or (self.replan_every and step % self.replan_every == 0):
                try:
                    trajectory = source(state, goal)
                except ValueError as exc:
                    failure, detail = "planning_failure", str(exc)
                    break
            tick = perf_counter()
            try:
                proposed = self.executor.propose(state, trajectory, self.executor.controller.dt)
            except WheeledControlError as exc:
                failure, detail = "safety_stop", str(exc)
                break
            timings.append((perf_counter()-tick)*1000)
            if self.transition_checker is not None and not self.transition_checker(state, proposed):
                failure = "safety_stop"
                break
            state = proposed
            controls.append([self.executor.last_command.v, self.executor.last_command.steering_angle])
            states.append(state.to_array().tolist())
        pos_error = math.hypot(state.x-goal.x, state.y-goal.y)
        yaw_error = abs(math.atan2(math.sin(state.yaw-goal.yaw), math.cos(state.yaw-goal.yaw)))
        if failure == "timeout" and pos_error < self.tol_pos and yaw_error < self.tol_yaw:
            failure = None
        return dict(success=failure is None, failure=failure, detail=detail, states=states, controls=controls,
                    model_identity=self.executor.model.config.identity,
                    summary=dict(steps=len(controls), position_error=pos_error, yaw_error=yaw_error,
                                 max_prediction_error=self.executor.max_prediction_error,
                                 collision_checked=self.transition_checker is not None,
                                 compute_ms_max=max(timings, default=0.),
                                 compute_ms_mean=float(np.mean(timings)) if timings else 0.))
