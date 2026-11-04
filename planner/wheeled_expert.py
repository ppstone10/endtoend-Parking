"""有界RS专家：几何候选经停车转轮与离散速度剖面生成真实模型状态。"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from numbers import Real
from time import perf_counter

import numpy as np

from interfaces.wheeled import WheeledControlCmd, WheeledTrajectory, WheeledVehicleState
from .reeds_shepp import reeds_shepp_paths


class WheeledPlanningError(ValueError):
    """当前有限候选或预算内未生成合格参考，不代表全局无解。"""

    def __init__(self, message, *, candidate_failures=None):
        super().__init__(message)
        self.candidate_failures = dict(candidate_failures or {})


@dataclass(frozen=True)
class WheeledExpertPlan:
    trajectory: WheeledTrajectory
    states: np.ndarray
    controls: np.ndarray
    candidate_word: str


class WheeledReedsSheppPlanner:
    def __init__(self, model, checker, *, dt=.1, cruise_speed=.5,
                 max_candidates=48, max_steps=2400, max_seconds=3.):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (dt, cruise_speed, max_seconds)):
            raise ValueError("专家dt/速度/时间预算必须有限且为正")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in (max_candidates, max_steps)):
            raise ValueError("专家候选数与步数必须为正整数")
        if abs(checker.dt-dt) > 1e-12 or checker.model.config.identity != model.config.identity:
            raise ValueError("专家、模型与转移检查身份/dt不一致")
        self.model, self.checker, self.dt = model, checker, float(dt)
        self.cruise_speed, self.max_candidates = float(cruise_speed), max_candidates
        self.max_steps, self.max_seconds = max_steps, float(max_seconds)

    def _speed_profile(self, distance, speed_limit):
        """从0到0的采样梯形包络，统一缩放以严格匹配段弧长。"""
        c, dt = self.model.config, self.dt
        count = (math.ceil(distance/(speed_limit*dt))+math.ceil(speed_limit/(c.max_acceleration*dt))
                 +math.ceil(speed_limit/(c.max_deceleration*dt))+1)
        if count > self.max_steps:
            raise WheeledPlanningError("当前候选超过输出步数预算", candidate_failures={"step_budget": 1})
        indices = np.arange(1, count+1)
        envelope = np.minimum(np.minimum(c.max_acceleration*indices*dt,
                                          c.max_deceleration*(count-indices)*dt), speed_limit)
        available = float(envelope.sum()*dt)
        if available < distance-1e-12 or available <= 0:
            raise WheeledPlanningError("离散速度包络未覆盖请求弧长", candidate_failures={"speed_profile": 1})
        return envelope*(distance/available)

    def plan(self, start, goal, *, requested_maneuver=None, minimum_requested_ratio=.5):
        if requested_maneuver not in (None, "forward", "reverse"):
            raise ValueError("请求机动必须为forward/reverse或None")
        if (isinstance(minimum_requested_ratio, bool) or not isinstance(minimum_requested_ratio, Real)
                or not math.isfinite(minimum_requested_ratio) or not 0 <= minimum_requested_ratio <= 1):
            raise ValueError("请求机动最小占比必须有限且位于[0,1]")
        if not isinstance(start, WheeledVehicleState):
            raise TypeError("轮式专家要求完整转角状态")
        self.model.yaw_rate(start)
        if not all(math.isfinite(v) for v in (goal.x, goal.y, goal.yaw)):
            raise ValueError("轮式专家目标必须有限")
        if not self.checker.pose_free(start):
            raise WheeledPlanningError("起点车体碰撞或规划余量不合格", candidate_failures={"initial_collision": 1})
        target = WheeledVehicleState(goal.x, goal.y, goal.yaw)
        if not self.checker.pose_free(target):
            raise WheeledPlanningError("目标车体碰撞或规划余量不合格", candidate_failures={"goal_collision": 1})
        deadline = perf_counter()+self.max_seconds
        failures = Counter()

        def check_budget():
            if perf_counter() > deadline:
                raise WheeledPlanningError("轮式专家时间预算耗尽",
                                           candidate_failures={**failures, "time_budget": 1})

        def append(states, controls, command):
            check_budget()
            if len(controls) >= self.max_steps:
                raise WheeledPlanningError("当前候选超过输出步数预算", candidate_failures={"step_budget": 1})
            before = states[-1]
            after = self.model.step(before, command, self.dt)
            if not self.checker(before, after):
                raise WheeledPlanningError("候选真实转移违反车体规划余量", candidate_failures={"geometry": 1})
            states.append(after)
            controls.append([command.v, command.steering_angle])

        prefix_states, prefix_controls = [start], []
        while prefix_states[-1].v != 0:
            append(prefix_states, prefix_controls, WheeledControlCmd(0, start.steering_angle))
        stopped = prefix_states[-1]
        rear_start = self.model.center_to_rear_pose((stopped.x, stopped.y, stopped.yaw))
        rear_goal = self.model.center_to_rear_pose((goal.x, goal.y, goal.yaw))
        candidates = reeds_shepp_paths(rear_start, rear_goal, self.model.config.min_turning_radius)
        last_error = "没有RS候选"
        for path in candidates[:self.max_candidates]:
            check_budget()
            states, controls = list(prefix_states), list(prefix_controls)
            try:
                for mode, length in zip(path.modes, path.lengths):
                    if abs(length) < 1e-12:
                        continue
                    delta = {"S": 0., "L": self.model.config.max_steering_angle,
                             "R": -self.model.config.max_steering_angle}[mode]
                    while states[-1].steering_angle != delta:
                        append(states, controls, WheeledControlCmd(0, delta))
                    limit = min(self.cruise_speed, self.model.config.max_forward_speed if length > 0
                                else self.model.config.max_reverse_speed)
                    speeds = self._speed_profile(abs(length), limit)
                    for speed in speeds:
                        append(states, controls, WheeledControlCmd(math.copysign(float(speed), length), delta))
                end = states[-1]
                yaw_error = abs(math.atan2(math.sin(end.yaw-goal.yaw), math.cos(end.yaw-goal.yaw)))
                if math.hypot(end.x-goal.x, end.y-goal.y) > 1e-7 or yaw_error > 1e-7 or end.v != 0:
                    raise WheeledPlanningError("真实模型终点未满足专家数值验收", candidate_failures={"terminal": 1})
                array = np.asarray([state.to_array() for state in states])
                if requested_maneuver is not None:
                    speeds = array[1:, 3]
                    requested = speeds > 0 if requested_maneuver == "forward" else speeds < 0
                    total = float(np.abs(speeds).sum())
                    ratio = float(np.abs(speeds[requested]).sum()/total) if total > 0 else 1.
                    if ratio < minimum_requested_ratio:
                        raise WheeledPlanningError("候选请求机动里程占比不足",
                                                   candidate_failures={"requested_ratio": 1})
                commands = np.asarray(controls, dtype=float).reshape(-1, 2)
                array.setflags(write=False)
                commands.setflags(write=False)
                return WheeledExpertPlan(WheeledTrajectory(array[:, :3], self.dt, self.model.config.identity),
                                          array, commands, path.word.name)
            except WheeledPlanningError as exc:
                if "time_budget" in exc.candidate_failures:
                    raise
                failures.update(exc.candidate_failures)
                last_error = str(exc)
        raise WheeledPlanningError(f"有限RS候选未通过：{last_error}",
                                   candidate_failures=failures or {"no_candidate": 1})
