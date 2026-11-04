"""独立低速单轨模型：显式转角状态、离散限速与恒曲率解析推进。"""

from __future__ import annotations

import math
from numbers import Real

from interfaces.wheeled import WheeledControlCmd, WheeledVehicleState
from .wheeled_config import WheeledVehicleConfig


def _validated_pose(pose) -> tuple[float, float, float]:
    if len(pose) != 3:
        raise ValueError("位姿必须是(x,y,yaw)三维")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
           for value in pose):
        raise ValueError("位姿必须包含有限数值")
    return tuple(float(value) for value in pose)


def _slew_towards(current: float, target: float, max_change: float) -> float:
    # 完整到达时直接返回目标，避免current+(target-current)抵消掉极小目标。
    if abs(target-current) <= max_change:
        return target
    return current + math.copysign(max_change, target-current)


class KinematicBicycleModel:
    """无隐含执行状态的轮式模型；不是旧MPC的即插即用替代品。

    每步先得到幅值/采样变化率受限的v和delta，再将它们在dt内零阶保持。
    这是离散运动约定，不模拟步内连续执行器加速/转向过程。
    """

    def __init__(self, config: WheeledVehicleConfig) -> None:
        if not isinstance(config, WheeledVehicleConfig):
            raise TypeError("轮式模型必须使用WheeledVehicleConfig")
        self.config = config

    def center_to_rear_pose(self, pose) -> tuple[float, float, float]:
        x, y, yaw = _validated_pose(pose)
        offset = self.config.center_from_rear
        return _validated_pose((x-offset*math.cos(yaw), y-offset*math.sin(yaw), yaw))

    def rear_to_center_pose(self, pose) -> tuple[float, float, float]:
        x, y, yaw = _validated_pose(pose)
        offset = self.config.center_from_rear
        return _validated_pose((x+offset*math.cos(yaw), y+offset*math.sin(yaw), yaw))

    def _validate_state(self, state: WheeledVehicleState) -> None:
        if not isinstance(state, WheeledVehicleState):
            raise TypeError("轮式模型不接受旧VehicleState，必须显式提供转角")
        if not -self.config.max_reverse_speed <= state.v <= self.config.max_forward_speed:
            raise ValueError("轮式状态速度超出配置限制")
        if abs(state.steering_angle) > self.config.max_steering_angle:
            raise ValueError("轮式状态转角超出配置限制")

    def yaw_rate(self, state: WheeledVehicleState) -> float:
        self._validate_state(state)
        return state.v * math.tan(state.steering_angle) / self.config.wheelbase

    def center_velocity(self, state: WheeledVehicleState) -> tuple[float, float]:
        """车体中心的世界系速度，保留后轴偏置产生的侧向分量。"""
        omega = self.yaw_rate(state)
        lateral = self.config.center_from_rear * omega
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        return state.v*c-lateral*s, state.v*s+lateral*c

    def _limited_speed(self, current: float, requested: float, dt: float) -> float:
        target = max(-self.config.max_reverse_speed, min(self.config.max_forward_speed, requested))
        if (current > 0 and target < 0) or (current < 0 and target > 0):
            target = 0.0
        rate = (self.config.max_acceleration if abs(target) > abs(current)
                else self.config.max_deceleration)
        return _slew_towards(current, target, rate*dt)

    def step(self, state: WheeledVehicleState, command: WheeledControlCmd,
             dt: float) -> WheeledVehicleState:
        self._validate_state(state)
        if not isinstance(command, WheeledControlCmd):
            raise TypeError("轮式模型不接受差速ControlCmd，必须显式提供转角指令")
        if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt 必须是有限正数，不能是布尔值")
        v = self._limited_speed(state.v, command.v, dt)
        target_delta = max(-self.config.max_steering_angle,
                           min(self.config.max_steering_angle, command.steering_angle))
        delta = _slew_towards(state.steering_angle, target_delta, self.config.max_steering_rate*dt)
        if v == 0:
            return WheeledVehicleState(state.x, state.y, state.yaw, v, delta)

        distance = v * dt
        turn = distance * math.tan(delta) / self.config.wheelbase
        yaw = state.yaw + turn
        if not all(math.isfinite(value) for value in (distance, turn, yaw)):
            raise ValueError("轮式推进溢出，不能生成非有限状态")
        # sinc(turn/2)避免在接近直线时对两个近等sin/cos作差。
        scale = 1 - turn*turn/24 if abs(turn) < 1e-8 else math.sin(turn/2)/(turn/2)
        rear_x, rear_y, _ = self.center_to_rear_pose((state.x, state.y, state.yaw))
        rear_x += distance * scale * math.cos(state.yaw + turn/2)
        rear_y += distance * scale * math.sin(state.yaw + turn/2)
        x, y, yaw = self.rear_to_center_pose((rear_x, rear_y, yaw))
        return WheeledVehicleState(x, y, yaw, v, delta)
