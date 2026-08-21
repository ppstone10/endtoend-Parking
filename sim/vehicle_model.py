"""差分驱动车辆运动模型。

低速泊车场景使用差分运动学模型，状态 (x, y, yaw, v, omega)。
"""

from __future__ import annotations

import numpy as np

from interfaces import ControlCmd, VehicleState


class DifferentialDriveModel:
    """差分驱动运动学模型，使用欧拉积分推进状态。"""

    def __init__(self, max_v: float = 2.0, max_omega: float = 1.0) -> None:
        self.max_v = max_v
        self.max_omega = max_omega

    def step(
        self, state: VehicleState, cmd: ControlCmd, dt: float
    ) -> VehicleState:
        """按控制指令推进状态 dt 秒，并做最大速度限幅。"""
        v = float(np.clip(cmd.v, -self.max_v, self.max_v))
        omega = float(np.clip(cmd.omega, -self.max_omega, self.max_omega))
        new_x = state.x + v * np.cos(state.yaw) * dt
        new_y = state.y + v * np.sin(state.yaw) * dt
        new_yaw = state.yaw + omega * dt
        return VehicleState(new_x, new_y, new_yaw, v, omega)