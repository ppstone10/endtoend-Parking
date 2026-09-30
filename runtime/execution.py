"""闭环执行器：把参考轨迹转为车辆推进策略。

闭环引擎把"参考轨迹 → 下一车辆状态"这一步抽象为执行器，以便把执行质量
从网络质量中分离出来：

- ``MpcVehicleExecutor``：生产路径。MPC 滚动优化出控制量，由车辆运动模型
  推进（有跟踪误差、限幅与动力学约束）。
- ``IdealPathExecutor``：理想执行。不做优化、不加动力学，直接按预测轨迹的
  弧长推进（等价于"执行器能完美跟踪"），用于隔离网络自身的滚动行为。

两者的输入输出契约一致（``propose(state, trajectory, dt) -> VehicleState``），
因此同一份网络轨迹可以分别送入两种执行器做同索引对照。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from interfaces import ControlCmd, Trajectory, VehicleState

__all__ = [
    "ExecutionError",
    "IdealPathExecutor",
    "MpcVehicleExecutor",
    "TrajectoryExecutor",
]


class ExecutionError(RuntimeError):
    """执行器无法给出下一状态时明确失败，不静默保持原位。"""


class TrajectoryExecutor(Protocol):
    """执行器接口：给定当前状态与参考轨迹，给出下一控制周期的车辆状态。

    ``last_cmd`` 返回本次推进等价的控制指令（MPC 实际指令 / 理想执行沿路径的
    等效指令），供回合记录与振荡分类复用，避免理想执行丢失方向切换证据。
    """

    def reset(self) -> None:
        """重置回合内状态（弧长游标等）。"""

    @property
    def last_cmd(self) -> ControlCmd:
        """上一控制周期等价的控制指令。"""
        ...

    def propose(
        self, state: VehicleState, trajectory: Trajectory, dt: float
    ) -> VehicleState:
        """返回推进一个控制周期后的车辆状态。"""
        ...

    def begin_trajectory(self, trajectory: Trajectory) -> None:
        """通知执行器收到一条新参考轨迹（重规划点）。"""
        ...

    def audit(self) -> dict:
        """返回执行器自身的诊断量（可 JSON 序列化）。"""
        ...


def _as_points(trajectory: Trajectory) -> np.ndarray:
    points = np.asarray(trajectory.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ExecutionError("参考轨迹必须是 (N,>=2) 的点序列")
    return points


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class MpcVehicleExecutor:
    """生产执行器：MPC 计算控制量 + 车辆运动模型推进。"""

    def __init__(self, mpc, vehicle_model) -> None:
        self.mpc = mpc
        self.vehicle_model = vehicle_model
        self._last_cmd = ControlCmd(0.0, 0.0)

    def reset(self) -> None:
        self.mpc.reset()
        self._last_cmd = ControlCmd(0.0, 0.0)

    @property
    def last_cmd(self) -> ControlCmd:
        return self._last_cmd

    def begin_trajectory(self, trajectory: Trajectory) -> None:
        """MPC 每次 ``compute`` 直接消费轨迹，无需游标状态。"""

    def propose(
        self, state: VehicleState, trajectory: Trajectory, dt: float
    ) -> VehicleState:
        cmd = self.mpc.compute(trajectory, state)
        self._last_cmd = cmd
        return self.vehicle_model.step(state, cmd, dt)

    def audit(self) -> dict:
        return {"kind": "mpc_vehicle"}


class IdealPathExecutor:
    """理想执行器：沿参考轨迹按弧长直接推进车辆状态。

    每次前进的名义弧长取参考轨迹的"相邻点弧长中位数"（由网络 dt 决定），因此
    推进速度与轨迹自身的采样间隔一致：网络 dt=0.2s、控制周期 0.1s（
    ``control_steps_per_point=2``）时每点耗时 0.1s，等价于 1 点/米级的
    匀速前进。前进时沿轨迹折线插值位置与航向，不引入跟踪误差与控制限幅，
    代表"执行器能完美跟踪网络轨迹"的上界，用于判定网络自身是否收敛。

    重规划时（``begin_trajectory``）弧长游标清零，新轨迹从起点开始消费；
    起点即当前车辆位置，所以游标从 0 开始推进不会造成跳变。
    """

    def __init__(self, control_steps_per_point: float = 1.0) -> None:
        if control_steps_per_point <= 0.0:
            raise ValueError("control_steps_per_point 必须为正")
        self._factor = float(control_steps_per_point)
        self._points: np.ndarray | None = None
        self._arc: np.ndarray | None = None
        self._cursor_arc = 0.0
        self._executed_distance = 0.0
        self._samples = 0
        self._advance_failures = 0
        self._last_cmd = ControlCmd(0.0, 0.0)

    def reset(self) -> None:
        self._points = None
        self._arc = None
        self._cursor_arc = 0.0
        self._executed_distance = 0.0
        self._samples = 0
        self._advance_failures = 0
        self._last_cmd = ControlCmd(0.0, 0.0)

    @property
    def last_cmd(self) -> ControlCmd:
        return self._last_cmd

    @property
    def control_steps_per_point(self) -> float:
        """相邻轨迹点之间的控制周期数。"""
        return self._factor

    @control_steps_per_point.setter
    def control_steps_per_point(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("control_steps_per_point 必须为正")
        self._factor = float(value)

    @property
    def arc_per_step(self) -> float:
        """每个控制周期前进的名义弧长（点距中位数 / 每点控制周期数）。"""
        return self.arc_per_point / self._factor

    def begin_trajectory(self, trajectory: Trajectory) -> None:
        points = _as_points(trajectory)
        if points.shape[0] < 2:
            raise ExecutionError("理想执行要求参考轨迹至少 2 个点")
        segments = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
        self._points = points
        self._arc = np.concatenate([[0.0], np.cumsum(segments)])
        self._cursor_arc = 0.0

    def propose(
        self, state: VehicleState, trajectory: Trajectory, dt: float
    ) -> VehicleState:
        if self._points is None or self._arc is None:
            self.begin_trajectory(trajectory)
        assert self._points is not None and self._arc is not None
        self._cursor_arc = min(
            self._cursor_arc + self.arc_per_step, float(self._arc[-1])
        )
        target = self._interpolate(self._cursor_arc)
        executed = float(np.hypot(target[0] - state.x, target[1] - state.y))
        self._executed_distance += executed
        self._samples += 1
        if executed <= 1e-9 and self._cursor_arc < float(self._arc[-1]) - 1e-9:
            self._advance_failures += 1
        yaw_delta = _wrap(float(target[2]) - state.yaw)
        self._last_cmd = ControlCmd(
            v=executed / dt if dt > 0.0 else 0.0,
            omega=yaw_delta / dt if dt > 0.0 else 0.0,
        )
        return VehicleState(
            float(target[0]),
            float(target[1]),
            float(target[2]),
            self._last_cmd.v,
            self._last_cmd.omega,
        )

    @property
    def arc_per_point(self) -> float:
        """相邻轨迹点的名义弧长（轨迹不规则时取中位数，避免极端段主导）。"""
        assert self._points is not None and self._arc is not None
        segment = np.diff(self._arc)
        if segment.shape[0] == 0:
            raise ExecutionError("参考轨迹没有有效段长")
        return float(np.median(segment))

    def _interpolate(self, arc: float) -> np.ndarray:
        assert self._points is not None and self._arc is not None
        index = int(np.searchsorted(self._arc, arc, side="right") - 1)
        index = min(max(index, 0), self._points.shape[0] - 1)
        if index >= self._points.shape[0] - 1:
            return self._points[-1]
        span = float(self._arc[index + 1] - self._arc[index])
        ratio = 0.0 if span <= 1e-12 else (arc - float(self._arc[index])) / span
        start = self._points[index]
        end = self._points[index + 1]
        yaw = start[2] + ratio * _wrap(float(end[2]) - float(start[2]))
        return np.array(
            [
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
                yaw,
            ],
            dtype=np.float64,
        )

    def audit(self) -> dict:
        """理想执行诊断（无预测-执行偏差量词：理想执行的偏差恒为跟踪口径）。"""
        total_arc = float(self._arc[-1]) if self._arc is not None else 0.0
        return {
            "kind": "ideal_path",
            "control_steps_per_point": self._factor,
            "samples": self._samples,
            "executed_distance_m": round(self._executed_distance, 4),
            "total_arc_m": round(total_arc, 4),
            "advance_failures": self._advance_failures,
        }
