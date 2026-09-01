"""闭环回合指标定义与聚合。

EpisodeResult 对应论文 8 项指标中单回合可度量的部分：
泊车成功、碰撞、最终位置误差、最终航向误差、路径长度、泊车时间、
MPC 跟踪误差（横向偏差 RMS）、网络/规划推理时间。summarize 聚合多回合。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeResult:
    """单个泊车回合的完整指标。"""

    success: bool
    failure: str | None  # None | "collision" | "timeout" | "pose_error" | "oscillation"
    steps: int
    final_pos_err: float  # 米
    final_yaw_err: float  # 弧度
    path_length: float  # 米
    parking_time: float  # 秒
    tracking_rms: float  # 对参考轨迹的横向偏差 RMS（米）
    inference_ms: float  # 轨迹源单次推理/规划平均耗时（毫秒）
    collision: bool = False
    record: Any = None  # EpisodeRecord，供可视化回放
    meta: dict = field(default_factory=dict)  # 任务元数据（场景/任务类型等）

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "failure": self.failure,
            "steps": self.steps,
            "final_pos_err": self.final_pos_err,
            "final_yaw_err": self.final_yaw_err,
            "path_length": self.path_length,
            "parking_time": self.parking_time,
            "tracking_rms": self.tracking_rms,
            "inference_ms": self.inference_ms,
            "collision": self.collision,
            "meta": self.meta,
        }


def summarize(results: list[EpisodeResult]) -> dict:
    """聚合多回合指标：成功率、碰撞率与各项均值±标准差。

    除八项论文指标外，附加两条顶层异常检测指标：
    - time_dist_ratio：泊车时间 / 实际路径长度（s/m），异常高说明振荡/空转；
    - winding：实际路径长度 / 起终点直线距离，异常高说明绕路。
    winding 依赖回合 meta 中的 start_xy（起终点直线距离），缺失时跳过。
    """

    def _ms(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    n = len(results)
    if n == 0:
        return {"episodes": 0}
    out: dict = {
        "episodes": n,
        "success_rate": sum(r.success for r in results) / n,
        "collision_rate": sum(r.collision for r in results) / n,
    }
    for key in ("final_pos_err", "final_yaw_err", "path_length", "parking_time", "tracking_rms", "inference_ms"):
        mean, std = _ms([getattr(r, key) for r in results])
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std

    # 时间-距离效率比：时间与路径同量级衡量空转。
    ratios = [r.parking_time / r.path_length for r in results if r.path_length > 0.0]
    if ratios:
        mean, std = _ms(ratios)
        out["time_dist_ratio_mean"] = mean
        out["time_dist_ratio_std"] = std

    # 蜿蜒度：路径长度 / 起终点直线距离（需要 meta 提供 start_xy）。
    winding = [
        r.path_length / np.hypot(
            float(r.meta["goal_x"] - r.meta["start_x"]),
            float(r.meta["goal_y"] - r.meta["start_y"]),
        )
        for r in results
        if r.path_length > 0.0
        and {"start_x", "start_y", "goal_x", "goal_y"}.issubset(r.meta)
    ]
    if winding:
        mean, std = _ms(winding)
        out["winding_mean"] = mean
        out["winding_std"] = std

    failures: dict[str, int] = {}
    for r in results:
        if r.failure is not None:
            failures[r.failure] = failures.get(r.failure, 0) + 1
    out["failures"] = failures
    return out
