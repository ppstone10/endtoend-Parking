"""滚动闭环一致性指标（E2/E3 专用）。

既有 ``metrics.evaluation`` 只给回合级终态指标（成功率/终点误差/跟踪 RMS）。
判断"网络自己连续滚动会不会跑偏"需要过程级证据，本模块按控制周期采集
状态、距目标距离、路径长度与当前参考轨迹，输出：

- ``divergence_events``：一个重规划周期内"距目标距离增加超过阈值"的次数
  （跑偏的直接计数）；
- ``goal_approach_monotonic_ratio``：重规划序列中距目标距离下降（净前进为正）
  的比例；
- ``progress_eff_median``：净前进距离 / 实际位移（≤0 表示原地或倒退）；
- ``drift_slope``：按重规划序号对距目标距离做最小二乘拟合的斜率
  （>0 表示越跑越远）；
- ``pred_dist_to_traj_m``：实际状态到当前网络预测轨迹的最近距离
  （网络说走这条线、实际走到离它多远；理想执行下应接近 0）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DIVERGENCE_TOLERANCE_M",
    "CycleSample",
    "CycleTracker",
    "analyze_cycle_samples",
    "linear_slope",
]


DIVERGENCE_TOLERANCE_M = 0.1
HEADING_DIVERGENCE_TOLERANCE_RAD = np.deg2rad(5.0)


@dataclass
class CycleSample:
    """一个控制周期的过程记录。"""

    step: int
    cycle: int
    time_s: float
    x: float
    y: float
    yaw: float
    d_goal_m: float
    yaw_err_rad: float
    path_length_m: float
    distance_to_trajectory_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "cycle": self.cycle,
            "time_s": round(self.time_s, 4),
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(self.yaw, 6),
            "d_goal_m": round(self.d_goal_m, 4),
            "yaw_err_deg": round(float(np.degrees(self.yaw_err_rad)), 3),
            "path_length_m": round(self.path_length_m, 4),
            "dist_to_traj_m": (
                None
                if self.distance_to_trajectory_m is None
                else round(self.distance_to_trajectory_m, 4)
            ),
        }


@dataclass
class CycleTracker:
    """按控制周期采集过程证据，供滚动一致性分析使用。

    ``begin`` 在回合开始时调用；``start_cycle`` 在每次重规划（收到新参考轨迹）
    时调用；``sample`` 每个控制周期调用一次。
    """

    goal: np.ndarray | None = None
    samples: list[CycleSample] = field(default_factory=list)
    _cycle: int = 0
    _cycle_start_d_goal: float = 0.0
    _reference_points: np.ndarray | None = None

    def begin(self, goal, state) -> None:
        """回合开始时初始化目标与周期计数。"""
        self.goal = np.asarray([goal.x, goal.y, goal.yaw], dtype=np.float64)
        self.samples = []
        self._cycle = 0
        self._cycle_start_d_goal = self._distance_to_goal(state)
        self._reference_points = None

    def start_cycle(self, state, trajectory=None) -> None:
        """进入新的重规划周期，并记录本次参考轨迹。"""
        self._cycle += 1
        self._cycle_start_d_goal = self._distance_to_goal(state)
        if trajectory is not None:
            self._reference_points = np.asarray(trajectory.points, dtype=np.float64)

    def sample(self, state, *, path_length: float, time_s: float) -> None:
        """记录一个控制周期的状态。"""
        self.samples.append(
            CycleSample(
                step=len(self.samples),
                cycle=self._cycle,
                time_s=float(time_s),
                x=float(state.x),
                y=float(state.y),
                yaw=float(state.yaw),
                d_goal_m=self._distance_to_goal(state),
                yaw_err_rad=self._heading_error(state),
                path_length_m=float(path_length),
                distance_to_trajectory_m=self._distance_to_trajectory(state),
            )
        )

    def _distance_to_goal(self, state) -> float:
        if self.goal is None:
            raise ValueError("CycleTracker.begin 未调用")
        return float(np.hypot(state.x - self.goal[0], state.y - self.goal[1]))

    def _heading_error(self, state) -> float:
        """当前航向相对目标航向的环绕误差（弧度，取绝对值）。

        冒烟实测：闭环失败常表现为"位置到位但航向不收敛"（E2 终点航向
        48°–77° 而距目标距离始终下降），只看距离会漏判，故此处单独记录。
        """
        if self.goal is None:
            raise ValueError("CycleTracker.begin 未调用")
        return float(
            abs(np.arctan2(np.sin(state.yaw - self.goal[2]), np.cos(state.yaw - self.goal[2])))
        )

    def _distance_to_trajectory(self, state) -> float | None:
        """实际状态到当前参考轨迹（网络预测）的最近距离。

        取最近点而非时间对齐点：网络 ``dt`` 与 MPC ``dt`` 不一致（0.2s vs 0.1s），
        时间索引会引入口径误差；最近点距离只回答"实际走的地方离预测轨迹多远"，
        不依赖时间语义。
        """
        points = self._reference_points
        if points is None or points.shape[0] == 0:
            return None
        distance = np.hypot(points[:, 0] - state.x, points[:, 1] - state.y)
        return float(np.min(distance))

    def cycles(self) -> list[dict[str, Any]]:
        """按重规划周期聚合：净前进、位移、进度效率、位置/航向发散。"""
        if self.goal is None:
            raise ValueError("CycleTracker.begin 未调用")
        grouped: dict[int, list[CycleSample]] = {}
        for item in self.samples:
            grouped.setdefault(item.cycle, []).append(item)
        cycles: list[dict[str, Any]] = []
        for cycle in sorted(grouped):
            items = grouped[cycle]
            first, last = items[0], items[-1]
            d_start, d_end = first.d_goal_m, last.d_goal_m
            yaw_start, yaw_end = first.yaw_err_rad, last.yaw_err_rad
            displacement = last.path_length_m - first.path_length_m
            net_forward = d_start - d_end
            net_heading = yaw_start - yaw_end
            cycles.append(
                {
                    "cycle": cycle,
                    "steps": len(items),
                    "d_goal_start_m": round(d_start, 4),
                    "d_goal_end_m": round(d_end, 4),
                    "yaw_err_start_deg": round(float(np.degrees(yaw_start)), 3),
                    "yaw_err_end_deg": round(float(np.degrees(yaw_end)), 3),
                    "net_forward_m": round(net_forward, 4),
                    "net_heading_deg": round(float(np.degrees(net_heading)), 3),
                    "displacement_m": round(displacement, 4),
                    "progress_eff": (
                        None
                        if displacement <= 1e-9
                        else round(net_forward / displacement, 4)
                    ),
                    "divergence": bool(d_end - d_start > DIVERGENCE_TOLERANCE_M),
                    "heading_divergence": bool(
                        yaw_end - yaw_start > HEADING_DIVERGENCE_TOLERANCE_RAD
                    ),
                }
            )
        return cycles


def linear_slope(values: list[float]) -> float | None:
    """对序列按序号做最小二乘拟合，返回斜率；样本不足 2 个返回 None。"""
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator <= 1e-12:
        return None
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denominator)


def analyze_cycle_samples(tracker: CycleTracker) -> dict[str, Any]:
    """汇总过程证据为滚动一致性报告。"""
    cycles = tracker.cycles()
    gaps = [
        item.distance_to_trajectory_m
        for item in tracker.samples
        if item.distance_to_trajectory_m is not None
    ]
    if not cycles:
        return {
            "cycles": 0,
            "samples": len(tracker.samples),
            "pred_dist_to_traj_m": (
                round(float(np.mean(gaps)), 4) if gaps else None
            ),
        }
    divergences = [c for c in cycles if c["divergence"]]
    heading_divergences = [c for c in cycles if c["heading_divergence"]]
    net_forwards = [c["net_forward_m"] for c in cycles]
    monotonic = sum(1 for value in net_forwards if value > 0.0)
    net_headings = [c["net_heading_deg"] for c in cycles]
    heading_monotonic = sum(1 for value in net_headings if value > 0.0)
    progress = [c["progress_eff"] for c in cycles if c["progress_eff"] is not None]
    d_goal_series = [tracker.samples[0].d_goal_m] + [c["d_goal_end_m"] for c in cycles]
    yaw_series = [round(float(np.degrees(tracker.samples[0].yaw_err_rad)), 3)] + [
        c["yaw_err_end_deg"] for c in cycles
    ]
    slope = linear_slope(d_goal_series)
    yaw_slope = linear_slope(yaw_series)
    return {
        "cycles": len(cycles),
        "samples": len(tracker.samples),
        "d_goal_start_m": round(d_goal_series[0], 4),
        "d_goal_end_m": round(d_goal_series[-1], 4),
        "d_goal_min_m": round(min(d_goal_series), 4),
        "yaw_err_start_deg": yaw_series[0],
        "yaw_err_end_deg": yaw_series[-1],
        "yaw_err_min_deg": min(yaw_series),
        "divergence_events": len(divergences),
        "divergence_cycles": [c["cycle"] for c in divergences],
        "goal_approach_monotonic_ratio": round(monotonic / len(cycles), 4),
        "heading_divergence_events": len(heading_divergences),
        "heading_approach_monotonic_ratio": round(heading_monotonic / len(cycles), 4),
        "progress_eff_median": (
            round(float(np.median(progress)), 4) if progress else None
        ),
        "drift_slope": None if slope is None else round(slope, 5),
        "heading_drift_slope_deg": None if yaw_slope is None else round(yaw_slope, 4),
        "pred_dist_to_traj_m": round(float(np.mean(gaps)), 4) if gaps else None,
        "pred_dist_to_traj_max_m": round(float(np.max(gaps)), 4) if gaps else None,
    }
