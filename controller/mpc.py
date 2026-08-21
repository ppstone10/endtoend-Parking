"""MPC 轨迹跟踪控制器。

阶段五实现：输入未来轨迹与当前状态，输出 [v_cmd, omega_cmd]。
本阶段只保留接口契约。
"""

from __future__ import annotations

from interfaces import ControlCmd, Trajectory, VehicleState


class MPCController:
    """MPC 轨迹跟踪控制器骨架。"""

    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt

    def compute(
        self, trajectory: Trajectory, state: VehicleState
    ) -> ControlCmd:
        """根据当前状态与目标轨迹计算控制指令（骨架阶段未实现）。"""
        raise NotImplementedError("MPCController.compute 将在阶段五实现")