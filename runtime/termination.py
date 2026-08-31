"""闭环终止判定与失败分类。

到达判定为位置+航向双阈值（阈值来自目标车位容差，见 REQUIREMENTS §3）；
失败分类：碰撞 / 超时 / 位姿超差 / 振荡。
"""

from __future__ import annotations

import numpy as np

from interfaces import GoalPose, VehicleState

FAILURE_COLLISION = "collision"
FAILURE_TIMEOUT = "timeout"
FAILURE_POSE_ERROR = "pose_error"
FAILURE_OSCILLATION = "oscillation"
FAILURE_SAFETY_STOP = "safety_stop"


class TerminalChecker:
    """按位置与航向双阈值判定到达目标。"""

    def __init__(self, tol_pos: float = 0.3, tol_yaw: float = np.deg2rad(10.0)) -> None:
        if tol_pos <= 0.0 or tol_yaw <= 0.0:
            raise ValueError("终止阈值必须为正")
        self.tol_pos = tol_pos
        self.tol_yaw = tol_yaw

    @staticmethod
    def pos_err(state: VehicleState, goal: GoalPose) -> float:
        return float(np.hypot(state.x - goal.x, state.y - goal.y))

    @staticmethod
    def yaw_err(state: VehicleState, goal: GoalPose) -> float:
        return float(abs(np.arctan2(np.sin(state.yaw - goal.yaw), np.cos(state.yaw - goal.yaw))))

    def reached(self, state: VehicleState, goal: GoalPose) -> bool:
        return self.pos_err(state, goal) < self.tol_pos and self.yaw_err(state, goal) < self.tol_yaw


def classify_oscillation(
    cmd_vs: np.ndarray,
    ref_flips: int | None = None,
    base_flips: int = 16,
    flip_margin: int = 8,
    min_speed: float = 0.1,
) -> bool:
    """根据速度符号翻转次数判定振荡。

    参考轨迹本身的方向切换是合法机动，阈值取 2 倍参考切换数加余量；
    无参考时用 base_flips。|v| < min_speed 的抖动不计。
    """
    if cmd_vs.shape[0] < 2:
        return False
    v = np.asarray(cmd_vs, dtype=np.float64)
    active = np.abs(v) > min_speed
    signs = np.sign(v[active])
    if signs.shape[0] < 2:
        return False
    flips = int(np.sum(signs[1:] != signs[:-1]))
    threshold = base_flips if ref_flips is None else 2 * int(ref_flips) + flip_margin
    return flips > threshold
