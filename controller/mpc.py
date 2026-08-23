"""MPC 轨迹跟踪控制器。

对差分驱动模型做滚动时域优化：每个控制周期用 CEM（交叉熵方法）在预测时域内
搜索控制序列 [v, omega]，使预测轨迹贴合参考轨迹并对终态（终点位置+航向）强惩罚，
输出序列首项。上一周期的解平移一步作为热启动。

求解器纯 numpy 向量化：population 采样 → 批量 rollout → 精英拟合 → 方差收缩迭代。
控制量按 max_v/max_omega 限幅；参考轨迹 dt 与控制周期不一致时按时间插值对齐。
"""

from __future__ import annotations

import numpy as np

from interfaces import ControlCmd, Trajectory, VehicleState


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """将角度差归一化到 [-pi, pi)。"""
    return np.arctan2(np.sin(angle), np.cos(angle))


class MPCController:
    """MPC 轨迹跟踪控制器（CEM 求解）。

    dt 为控制周期；horizon 为预测时域（控制步数）；max_v/max_omega 为控制量上限；
    w_xy/w_yaw 为逐步跟踪权重；w_terminal 为终态权重（终点位置与航向的强惩罚，
    保证停准而非仅路过）；w_v/w_omega 为控制量惩罚（兼实现终点减速停稳）；
    population/elite_frac/iterations 为 CEM 参数；catchup_window 为进度超前
    纠正的前向窗口（点数）；seed 保证可复现。
    """

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 10,
        max_v: float = 2.0,
        max_omega: float = 1.0,
        w_xy: float = 1.0,
        w_yaw: float = 0.5,
        w_terminal: float = 20.0,
        w_v: float = 0.02,
        w_omega: float = 0.02,
        population: int = 80,
        elite_frac: float = 0.25,
        iterations: int = 5,
        catchup_window: int = 4,
        seed: int = 0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("控制周期 dt 必须为正")
        if horizon < 1:
            raise ValueError("预测时域 horizon 至少为 1")
        if not 0.0 < elite_frac <= 1.0:
            raise ValueError("elite_frac 必须在 (0, 1] 区间")
        self.dt = dt
        self.horizon = horizon
        self.max_v = max_v
        self.max_omega = max_omega
        self.w_xy = w_xy
        self.w_yaw = w_yaw
        self.w_terminal = w_terminal
        self.w_v = w_v
        self.w_omega = w_omega
        self.population = population
        self.elite_frac = elite_frac
        self.iterations = iterations
        self._catchup_window = catchup_window
        self._rng = np.random.default_rng(seed)
        self._last_u = np.zeros(2 * horizon)
        self._progress_traj: Trajectory | None = None
        self._progress_idx: int | None = None
        self._aligned_cache: np.ndarray | None = None

    def compute(self, trajectory: Trajectory, state: VehicleState) -> ControlCmd:
        """根据当前状态与参考轨迹计算控制指令。"""
        if trajectory is not self._progress_traj:
            # 新轨迹：进度索引重新初始化（支持滚动重规划时替换轨迹）。
            self._progress_traj = trajectory
            self._progress_idx = None
            self._aligned_cache = None
        aligned = self._aligned_cache
        if aligned is None:
            aligned = self._align_dt(trajectory)
            self._aligned_cache = aligned
        start_idx = self._advance_progress(aligned, state)
        ref = self._reference_points(aligned, start_idx)
        u_mean = self._shift(self._last_u)
        u_mean = self._solve(u_mean, ref, state)
        self._last_u = u_mean
        return ControlCmd(
            v=float(np.clip(u_mean[0], -self.max_v, self.max_v)),
            omega=float(np.clip(u_mean[1], -self.max_omega, self.max_omega)),
        )

    def reset(self) -> None:
        """清除进度与热启动状态（新一回合开始时调用）。"""
        self._progress_traj = None
        self._progress_idx = None
        self._aligned_cache = None
        self._last_u = np.zeros(2 * self.horizon)

    # ------------------------------------------------------------------
    # 参考轨迹预处理
    # ------------------------------------------------------------------

    def _align_dt(self, trajectory: Trajectory) -> np.ndarray:
        """将轨迹点按时间插值重采样到控制周期 dt 的网格上。

        轨迹 dt 与控制周期一致时原样返回；yaw 用 sin/cos 分量插值避免跨 ±pi 跳变。
        """
        pts = np.asarray(trajectory.points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("参考轨迹不能为空")
        if pts.shape[0] == 1 or abs(trajectory.dt - self.dt) < 1e-9:
            return pts
        n = pts.shape[0]
        t_src = np.arange(n) * float(trajectory.dt)
        total = t_src[-1]
        n_out = max(2, int(round(total / self.dt)) + 1)
        t_dst = np.linspace(0.0, total, n_out)
        out = np.empty((n_out, 3), dtype=np.float64)
        out[:, 0] = np.interp(t_dst, t_src, pts[:, 0])
        out[:, 1] = np.interp(t_dst, t_src, pts[:, 1])
        yaw_sin = np.interp(t_dst, t_src, np.sin(pts[:, 2]))
        yaw_cos = np.interp(t_dst, t_src, np.cos(pts[:, 2]))
        out[:, 2] = np.arctan2(yaw_sin, yaw_cos)
        return out

    def _advance_progress(self, pts: np.ndarray, state: VehicleState) -> int:
        """时间锚定的单调进度索引。

        每次 compute 前进一个对齐点（对齐后点距 = 控制周期），滞后时不回退
        （窗口先行形成追赶压力），并在前方小窗内按最近点纠正超前。
        以车辆当前位置为锚会让代价面在进度方向变平（快慢无差），故必须按
        轨迹时间轴推进。
        """
        n = pts.shape[0]
        if self._progress_idx is None:
            d = (pts[:, 0] - state.x) ** 2 + (pts[:, 1] - state.y) ** 2
            self._progress_idx = int(np.argmin(d))
            return self._progress_idx
        base = min(self._progress_idx + 1, n - 1)
        hi = min(n, base + self._catchup_window)
        seg = pts[base:hi]
        d = (seg[:, 0] - state.x) ** 2 + (seg[:, 1] - state.y) ** 2
        self._progress_idx = base + int(np.argmin(d))
        return self._progress_idx

    def _reference_points(self, pts: np.ndarray, start_idx: int) -> np.ndarray:
        """从进度索引起取预测时域参考点，不足时用末点填充。"""
        n = pts.shape[0]
        end_idx = min(n, start_idx + self.horizon)
        ref = pts[start_idx:end_idx]
        if ref.shape[0] < self.horizon:
            pad = np.tile(pts[-1], (self.horizon - ref.shape[0], 1))
            ref = np.concatenate([ref, pad], axis=0)
        return ref

    def _shift(self, u: np.ndarray) -> np.ndarray:
        """上一周期解平移一步作为热启动（末步重复）。"""
        out = np.empty_like(u)
        out[:-2] = u[2:]
        out[-2:] = u[-2:]
        return out

    # ------------------------------------------------------------------
    # CEM 求解
    # ------------------------------------------------------------------

    def _solve(self, u_mean: np.ndarray, ref: np.ndarray, state: VehicleState) -> np.ndarray:
        """交叉熵方法迭代搜索最优控制序列，返回精英均值。"""
        n = 2 * self.horizon
        std = np.tile(
            [self.max_v * 0.5, self.max_omega * 0.8], self.horizon
        )
        n_elite = max(2, int(round(self.population * self.elite_frac)))
        for _ in range(self.iterations):
            samples = self._rng.normal(size=(self.population, n)) * std + u_mean
            samples[:, 0::2] = np.clip(samples[:, 0::2], -self.max_v, self.max_v)
            samples[:, 1::2] = np.clip(samples[:, 1::2], -self.max_omega, self.max_omega)
            costs = self._batch_cost(samples, ref, state)
            elite_idx = np.argsort(costs)[:n_elite]
            elites = samples[elite_idx]
            u_mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-3
        return u_mean

    def _batch_cost(self, samples: np.ndarray, ref: np.ndarray, state: VehicleState) -> np.ndarray:
        """批量 rollout 全体采样并评估代价，返回 (population,) 代价。"""
        p = samples.shape[0]
        x = np.full(p, state.x)
        y = np.full(p, state.y)
        yaw = np.full(p, state.yaw)
        cost = np.zeros(p)
        for k in range(self.horizon):
            v = samples[:, 2 * k]
            omega = samples[:, 2 * k + 1]
            x = x + v * np.cos(yaw) * self.dt
            y = y + v * np.sin(yaw) * self.dt
            yaw = yaw + omega * self.dt
            dx = x - ref[k, 0]
            dy = y - ref[k, 1]
            yaw_err = _wrap_angle(yaw - ref[k, 2])
            cost += self.w_xy * (dx * dx + dy * dy) + self.w_yaw * yaw_err * yaw_err
        # 终态强惩罚：预测终点须同时贴合窗口末参考点的位置与航向。
        dx_end = x - ref[-1, 0]
        dy_end = y - ref[-1, 1]
        yaw_end_err = _wrap_angle(yaw - ref[-1, 2])
        cost += self.w_terminal * (
            dx_end * dx_end + dy_end * dy_end + self.w_yaw * yaw_end_err * yaw_end_err
        )
        v_all = samples[:, 0::2]
        omega_all = samples[:, 1::2]
        cost += self.w_v * np.sum(v_all * v_all, axis=1)
        cost += self.w_omega * np.sum(omega_all * omega_all, axis=1)
        return cost
