"""MPC 轨迹跟踪控制器。

对差分驱动模型做滚动时域优化：每个控制周期求解一段预测时域内的控制序列
[v, omega]，使预测轨迹尽可能贴合参考轨迹并抑制控制量，输出序列首项。

求解使用数值梯度下降（纯 numpy，无 scipy 依赖），控制量按最大速度限幅。
"""

from __future__ import annotations

import numpy as np

from interfaces import ControlCmd, Trajectory, VehicleState


class MPCController:
    """MPC 轨迹跟踪控制器。

    dt 为控制周期；horizon 为预测时域（控制步数）；max_v/max_omega 为控制量上限；
    w_yaw 为航向误差权重，w_v/w_omega 为控制量惩罚权重；
    iterations 为梯度下降迭代次数，lr 为学习率。
    """

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 10,
        max_v: float = 2.0,
        max_omega: float = 1.0,
        w_yaw: float = 0.5,
        w_v: float = 0.05,
        w_omega: float = 0.05,
        iterations: int = 60,
        lr: float = 0.1,
    ) -> None:
        self.dt = dt
        self.horizon = horizon
        self.max_v = max_v
        self.max_omega = max_omega
        self.w_yaw = w_yaw
        self.w_v = w_v
        self.w_omega = w_omega
        self.iterations = iterations
        self.lr = lr
        self._last_u = np.zeros(2 * horizon)

    def compute(self, trajectory: Trajectory, state: VehicleState) -> ControlCmd:
        """根据当前状态与目标轨迹计算控制指令。"""
        ref = self._reference_points(trajectory, state)
        u = self._last_u.copy()
        u = self._optimize(u, ref, state)
        self._last_u = u
        return ControlCmd(
            v=float(np.clip(u[0], -self.max_v, self.max_v)),
            omega=float(np.clip(u[1], -self.max_omega, self.max_omega)),
        )

    # ------------------------------------------------------------------
    # 参考轨迹
    # ------------------------------------------------------------------

    def _reference_points(
        self, trajectory: Trajectory, state: VehicleState
    ) -> np.ndarray:
        """从距离当前状态最近的轨迹点起，取预测时域内的参考点，不足时填充。"""
        pts = trajectory.points.astype(np.float64)
        n = pts.shape[0]
        horizon = self.horizon
        # 找欧氏距离最近的轨迹索引作为当前跟踪点。
        dx = pts[:, 0] - state.x
        dy = pts[:, 1] - state.y
        start_idx = int(np.argmin(dx * dx + dy * dy))
        end_idx = min(n, start_idx + horizon)
        ref = pts[start_idx:end_idx]
        if ref.shape[0] < horizon:
            pad = np.tile(pts[-1], (horizon - ref.shape[0], 1))
            ref = np.concatenate([ref, pad], axis=0)
        return ref

    # ------------------------------------------------------------------
    # 模型预测与代价
    # ------------------------------------------------------------------

    def _predict(self, state: VehicleState, u: np.ndarray) -> np.ndarray:
        """按差分运动学预测 horizon 步状态序列 (H, 3)。"""
        x, y, yaw = state.x, state.y, state.yaw
        dt = self.dt
        pred = np.empty((self.horizon, 3))
        for k in range(self.horizon):
            v = float(np.clip(u[2 * k], -self.max_v, self.max_v))
            omega = float(np.clip(u[2 * k + 1], -self.max_omega, self.max_omega))
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += omega * dt
            pred[k] = (x, y, yaw)
        return pred

    def _cost(self, pred: np.ndarray, ref: np.ndarray, u: np.ndarray) -> float:
        """跟踪误差 + 控制量惩罚。"""
        err = pred - ref
        xy_cost = float(np.sum(err[:, :2] ** 2))
        yaw_err = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
        yaw_cost = float(np.sum(yaw_err**2)) * self.w_yaw
        v = u[0::2]
        omega = u[1::2]
        u_cost = float(np.sum(v**2) * self.w_v + np.sum(omega**2) * self.w_omega)
        return xy_cost + yaw_cost + u_cost

    def _optimize(self, u: np.ndarray, ref: np.ndarray, state: VehicleState) -> np.ndarray:
        """数值梯度下降求解控制序列。"""
        u = u.astype(np.float64)
        n = len(u)
        step = 1e-4
        for _ in range(self.iterations):
            grad = np.zeros(n)
            base_cost = self._cost(self._predict(state, u), ref, u)
            for j in range(n):
                u_plus = u.copy()
                u_plus[j] += step
                cost_plus = self._cost(self._predict(state, u_plus), ref, u_plus)
                u_minus = u.copy()
                u_minus[j] -= step
                cost_minus = self._cost(self._predict(state, u_minus), ref, u_minus)
                grad[j] = (cost_plus - cost_minus) / (2.0 * step)
            u = u - self.lr * grad
            # 限幅，避免发散。
            u[0::2] = np.clip(u[0::2], -self.max_v, self.max_v)
            u[1::2] = np.clip(u[1::2], -self.max_omega, self.max_omega)
            _ = base_cost
        return u