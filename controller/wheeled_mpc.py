"""显式轮式CEM跟踪；模型通过注入提供，预测不另写车辆方程。"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from interfaces.wheeled import WheeledControlCmd, WheeledTrajectory, WheeledVehicleState


class WheeledControlError(RuntimeError):
    """所有候选均未通过明确的执行约束，不能静默回退旧差速控制。"""


class WheeledMPCController:
    def __init__(self, model, *, dt=.1, horizon=10, population=24, iterations=3,
                 seed=0, transition_checker=None):
        for name, value in (("horizon", horizon), ("population", population), ("iterations", iterations)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name}必须为正整数")
        if population < 4:
            raise ValueError("population至少为4")
        if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("控制dt必须有限且为正")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed必须为非负整数")
        self.model, self.dt = model, float(dt)
        self.horizon, self.population, self.iterations = int(horizon), int(population), int(iterations)
        self.seed, self.transition_checker = int(seed), transition_checker
        self.reset()

    def reset(self):
        self._rng = np.random.default_rng(self.seed)
        self._trajectory = None
        self._aligned = None
        self._reference_controls = None
        self._index = None
        self._warm = None
        self.last_prediction = None
        self.last_controls = None
        self.last_cost = None

    def align_reference(self, trajectory):
        if not isinstance(trajectory, WheeledTrajectory):
            raise TypeError("控制器要求显式WheeledTrajectory")
        if trajectory.model_identity != self.model.config.identity:
            raise ValueError("参考与轮式控制器车型身份不一致")
        points = trajectory.points
        if len(points) == 1 or abs(trajectory.dt-self.dt) < 1e-12:
            return points.copy()
        total = (len(points)-1)*trajectory.dt
        times = np.arange(int(math.ceil(total/self.dt))+1)*self.dt
        source_times = np.arange(len(points))*trajectory.dt
        return np.column_stack([np.interp(times, source_times,
                                          np.unwrap(points[:, k]) if k == 2 else points[:, k])
                                for k in range(3)])

    def rollout(self, state, controls):
        if not isinstance(state, WheeledVehicleState):
            raise TypeError("控制器要求显式轮式转角状态")
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[1] != 2 or len(controls) == 0 or not np.isfinite(controls).all():
            raise ValueError("控制序列必须是非空有限N×2")
        states = []
        for speed, delta in controls:
            state = self.model.step(state, WheeledControlCmd(speed, delta), self.dt)
            states.append(state.to_array())
        return np.asarray(states)

    def _cost(self, state, controls, reference):
        predicted = self.rollout(state, controls)
        if self.transition_checker is not None:
            previous = state
            for row in predicted:
                current = WheeledVehicleState.from_array(row)
                if not self.transition_checker(previous, current):
                    return math.inf, predicted
                previous = current
        xy = predicted[:, :2]-reference[:, :2]
        yaw = predicted[:, 2]-reference[:, 2]
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        tracking = 10*np.square(xy).sum(axis=1) + 5*np.square(yaw)
        cost = float(tracking.sum()+20*tracking[-1]+.01*np.square(predicted[:, 3]).sum())
        return cost, predicted

    def _feedforward(self, poses, state):
        """由后轴几何推导候选；停车时预转下一移动段所需转角，变化率由模型审核。"""
        rear = np.asarray([self.model.center_to_rear_pose(p) for p in poses])
        controls = []
        for before, after in zip(rear[:-1], rear[1:]):
            turn = math.atan2(math.sin(after[2]-before[2]), math.cos(after[2]-before[2]))
            mid = before[2]+turn/2
            chord = (after[0]-before[0])*math.cos(mid)+(after[1]-before[1])*math.sin(mid)
            sinc = 1 if abs(turn) < 1e-10 else math.sin(turn/2)/(turn/2)
            distance = chord/sinc
            speed = distance/self.dt
            delta = math.atan(self.model.config.wheelbase*turn/distance) if abs(distance) > 1e-10 else None
            controls.append((speed, delta))
        # 位姿不变不能反推转角；从同一条完整参考预览下一次移动，避免短窗口错过转轮时序。
        next_delta = next((delta for _, delta in reversed(controls) if delta is not None),
                          state.steering_angle)
        for index in range(len(controls)-1, -1, -1):
            speed, delta = controls[index]
            if delta is not None:
                next_delta = delta
            controls[index] = (speed, next_delta)
        controls.append((0., controls[-1][1] if controls else state.steering_angle))
        return self._bounded(np.asarray(controls))

    def _bounded(self, controls):
        controls = np.array(controls, dtype=float, copy=True)
        c = self.model.config
        controls[..., 0] = np.clip(controls[..., 0], -c.max_reverse_speed, c.max_forward_speed)
        controls[..., 1] = np.clip(controls[..., 1], -c.max_steering_angle, c.max_steering_angle)
        return controls

    def compute(self, trajectory, state):
        if not isinstance(state, WheeledVehicleState):
            raise TypeError("控制器要求显式轮式转角状态")
        if not isinstance(trajectory, WheeledTrajectory):
            raise TypeError("控制器要求显式WheeledTrajectory")
        if trajectory.model_identity != self.model.config.identity:
            raise ValueError("参考与轮式控制器车型身份不一致")
        if trajectory is not self._trajectory:
            self._aligned = self.align_reference(trajectory)
            self._trajectory = trajectory
            self._reference_controls = self._feedforward(self._aligned, state)
            self._index = int(np.argmin(np.square(self._aligned[:, :2]-[state.x, state.y]).sum(axis=1)))
        else:
            self._index = min(self._index+1, len(self._aligned)-1)
        indices = np.minimum(np.arange(self._index, self._index+self.horizon+1), len(self._aligned)-1)
        poses = self._aligned[indices]
        reference = poses[1:]
        feedforward = self._reference_controls[indices[:-1]].copy()
        warm = feedforward if self._warm is None else np.vstack((self._warm[1:], self._warm[-1:]))
        hold = np.tile([0., state.steering_angle], (self.horizon, 1))
        best_cost, best_controls, best_prediction = math.inf, None, None

        def assess(candidate):
            nonlocal best_cost, best_controls, best_prediction
            cost, prediction = self._cost(state, candidate, reference)
            if math.isfinite(cost) and cost < best_cost:
                best_cost, best_controls, best_prediction = cost, candidate.copy(), prediction
            return cost

        for candidate in (feedforward, warm, hold):
            assess(candidate)
        mean = warm.copy()
        std = np.tile([.5, self.model.config.max_steering_angle*.6], (self.horizon, 1))
        for _ in range(self.iterations):
            samples = self._bounded(mean+self._rng.normal(size=(self.population, self.horizon, 2))*std)
            costs = np.asarray([assess(sample) for sample in samples])
            finite = np.flatnonzero(np.isfinite(costs))
            if len(finite):
                elite = finite[np.argsort(costs[finite])[:max(2, self.population//5)]]
                mean = samples[elite].mean(axis=0)
                std = samples[elite].std(axis=0)+1e-3
                assess(mean)
        if best_controls is None:
            raise WheeledControlError("没有通过轮式执行约束的有限代价候选")
        self._warm = best_controls.copy()
        self.last_controls = best_controls.copy()
        self.last_prediction = best_prediction.copy()
        self.last_cost = best_cost
        return WheeledControlCmd(*best_controls[0])
