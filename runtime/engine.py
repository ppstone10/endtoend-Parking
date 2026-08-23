"""滚动闭环引擎。

每个回合：轨迹源供给全局参考轨迹 → MPC 逐控制周期跟踪 → 车辆模型推进；
每 replan_every 个周期向轨迹源重新取轨迹（K=1 即逐周期重感知）。
终止：到达（位置+航向双阈值）/ 碰撞 / 超时 / 振荡，失败自动分类。
"""

from __future__ import annotations

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from metrics import EpisodeResult
from .recorder import EpisodeRecord
from .sources import TrajectorySource
from .termination import (
    FAILURE_COLLISION,
    FAILURE_OSCILLATION,
    FAILURE_POSE_ERROR,
    FAILURE_TIMEOUT,
    TerminalChecker,
    classify_oscillation,
)


def vehicle_corners(state: VehicleState, length: float, width: float) -> np.ndarray:
    """车辆矩形四角全局坐标 (4, 2)。"""
    half_l, half_w = length / 2.0, width / 2.0
    cos_yaw, sin_yaw = np.cos(state.yaw), np.sin(state.yaw)
    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return local @ rot.T + np.array([state.x, state.y])


class ClosedLoopEngine:
    """滚动闭环泊车引擎。

    vehicle_model 提供 step(state, cmd, dt)；mpc 为 MPCController；
    source 为轨迹源；terminal 为到达判定；env 提供碰撞检测（可选）；
    vehicle_length/vehicle_width 用于碰撞矩形；replan_every 为重规划周期
    （控制周期数，1 为逐周期）；max_steps 为回合步数上限。
    """

    def __init__(
        self,
        vehicle_model,
        mpc,
        source: TrajectorySource,
        terminal: TerminalChecker | None = None,
        env=None,
        vehicle_length: float = 4.0,
        vehicle_width: float = 2.0,
        replan_every: int = 1,
        max_steps: int = 600,
        meta: dict | None = None,
    ) -> None:
        if replan_every < 1:
            raise ValueError("replan_every 至少为 1")
        self.vehicle_model = vehicle_model
        self.mpc = mpc
        self.source = source
        self.terminal = terminal or TerminalChecker()
        self.env = env
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.replan_every = replan_every
        self.max_steps = max_steps
        self.meta = meta or {}

    def run(self, start: VehicleState, goal: GoalPose) -> EpisodeResult:
        """执行一次闭环泊车回合，返回完整指标。"""
        state = VehicleState(start.x, start.y, start.yaw, start.v, start.omega)
        record = EpisodeRecord()
        self.mpc.reset()
        self.source.begin(state, goal)
        traj, infer_ms = self.source.next_trajectory(state)
        inference_times = [infer_ms]

        collision = False
        for step in range(self.max_steps):
            if step > 0 and step % self.replan_every == 0:
                traj, infer_ms = self.source.next_trajectory(state)
                inference_times.append(infer_ms)
            cmd = self.mpc.compute(traj, state)
            state = self.vehicle_model.step(state, cmd, self.mpc.dt)
            collision = self._check_collision(state)
            record.log(state, cmd, traj, traj, collision)
            if collision:
                break
            if self.terminal.reached(state, goal):
                break

        return self._build_result(state, goal, traj, record, inference_times, collision)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _check_collision(self, state: VehicleState) -> bool:
        if self.env is None:
            return False
        corners = vehicle_corners(state, self.vehicle_length, self.vehicle_width)
        return any(not self.env.is_free(float(cx), float(cy)) for cx, cy in corners)

    def _build_result(
        self,
        state: VehicleState,
        goal: GoalPose,
        traj: Trajectory,
        record: EpisodeRecord,
        inference_times: list[float],
        collision: bool,
    ) -> EpisodeResult:
        pos_err = self.terminal.pos_err(state, goal)
        yaw_err = self.terminal.yaw_err(state, goal)
        success = (not collision) and self.terminal.reached(state, goal)
        failure = None
        if collision:
            failure = FAILURE_COLLISION
        elif success:
            failure = None
        elif classify_oscillation(np.array([c.v for c in record.cmds]), self._ref_flips(traj)):
            failure = FAILURE_OSCILLATION
        elif pos_err < self.terminal.tol_pos * 2.0 and yaw_err < self.terminal.tol_yaw * 2.0:
            failure = FAILURE_POSE_ERROR  # 接近但未达标
        else:
            failure = FAILURE_TIMEOUT
        return EpisodeResult(
            success=success,
            failure=failure,
            steps=record.n_steps,
            final_pos_err=pos_err,
            final_yaw_err=yaw_err,
            path_length=record.path_length(),
            parking_time=record.n_steps * self.mpc.dt,
            tracking_rms=record.tracking_rms(),
            inference_ms=float(np.mean(inference_times)) if inference_times else 0.0,
            collision=collision,
            record=record,
            meta=dict(self.meta),
        )

    @staticmethod
    def _ref_flips(traj: Trajectory) -> int | None:
        """参考轨迹的方向切换次数（供振荡阈值参考）。"""
        pts = np.asarray(traj.points, dtype=np.float64)
        if pts.shape[0] < 3:
            return None
        seg = np.diff(pts[:, :2], axis=0)
        heading = np.arctan2(seg[:, 1], seg[:, 0])
        if heading.shape[0] < 2:
            return None
        return int(np.sum(np.abs(np.diff(heading)) > np.pi / 2.0))
