"""闭环访问状态轨迹质量指标（验证阶梯 L2）。

在闭环滚动过程中记录每个重规划状态，并在该状态同时采集网络预测轨迹与
专家重规划轨迹（同一状态、同一目标），度量网络在"闭环诱导状态分布"上的
轨迹质量。与开环（数据集起点一次性预测）不同，这里的误差直接对应
闭环失败机制：

- 访问状态开环误差：网络预测 vs 专家重规划 的 ADE/FDE/终点航向差；
- 近端质量：车辆距目标 < near_threshold_m 时，预测长度/终点距目标/终点
  航向误差/方向切换（方向 B 定论的振荡直接度量）；
- 跨重规划一致性：相邻两次重规划的终点航向跳变与长度跳变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

NEAR_THRESHOLD_M = 3.0
FLIP_ANGLE_RAD = np.pi / 2.0
CONSISTENCY_YAW_DEG = 30.0


@dataclass
class VisitedStateRecord:
    """单次重规划在闭环访问状态上的轨迹样本。

    state 为当前车辆状态 [x, y, yaw]（全局）；network_points 为网络预测轨迹
    （全局）；expert_points 为同一状态专家重规划轨迹（全局，可为 None）；
    goal 为目标位姿 [x, y, yaw]；meta 携带任务分组字段。
    """

    step: int
    state: np.ndarray
    network_points: np.ndarray
    expert_points: np.ndarray | None
    goal: np.ndarray
    meta: dict[str, Any]

    @property
    def state_xy(self) -> np.ndarray:
        return np.asarray(self.state, dtype=np.float64)[:2]

    @property
    def goal_xy(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=np.float64)[:2]

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.state_xy - self.goal_xy))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _traj_stats(points: np.ndarray, goal: np.ndarray) -> dict:
    """轨迹的点数、长度、终点距目标、终点航向误差与方向切换。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return {
            "n": 0, "length_m": 0.0, "end_to_goal_m": None,
            "end_yaw_err_deg": None, "flips": 0,
        }
    seg = np.diff(pts[:, :2], axis=0)
    length_m = float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))) if seg.shape[0] else 0.0
    end = pts[-1]
    end_to_goal = float(np.linalg.norm(end[:2] - goal[:2]))
    yaw_err = _wrap(end[2] - goal[2])
    headings = np.arctan2(seg[:, 1], seg[:, 0]) if seg.shape[0] else np.zeros(0)
    flips = (
        int(np.sum(np.abs(np.diff(headings)) > FLIP_ANGLE_RAD))
        if headings.shape[0] >= 2
        else 0
    )
    return {
        "n": int(pts.shape[0]),
        "length_m": round(length_m, 3),
        "end_to_goal_m": round(end_to_goal, 3),
        "end_yaw_err_deg": round(float(np.degrees(yaw_err)), 2),
        "flips": flips,
    }


def _pair_error(network: np.ndarray, expert: np.ndarray) -> dict | None:
    """网络预测 vs 专家重规划 在共同前缀上的 ADE/FDE/终点航向差。"""
    net = np.asarray(network, dtype=np.float64)
    exp = np.asarray(expert, dtype=np.float64)
    if net.shape[0] == 0 or exp.shape[0] == 0:
        return None
    n = min(net.shape[0], exp.shape[0])
    xy_err = np.linalg.norm(net[:n, :2] - exp[:n, :2], axis=1)
    yaw_err = np.abs(
        np.arctan2(
            np.sin(net[:n, 2] - exp[:n, 2]),
            np.cos(net[:n, 2] - exp[:n, 2]),
        )
    )
    return {
        "ade_m": round(float(xy_err.mean()), 3),
        "fde_m": round(float(np.linalg.norm(net[n - 1, :2] - exp[n - 1, :2])), 3),
        "yaw_mae_deg": round(float(np.degrees(yaw_err.mean())), 2),
        "prefix_n": n,
    }


def analyze_visited_state_records(
    records: Iterable[VisitedStateRecord],
    *,
    near_threshold_m: float = NEAR_THRESHOLD_M,
    consistency_yaw_deg: float = CONSISTENCY_YAW_DEG,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """计算逐重规划行与总体/分组聚合。

    返回 (report, rows)：report 含 overall 与 groups（按
    scene/task_type/maneuver/noise_level）；rows 为逐重规划可 JSON 序列化记录。
    """
    row_list = list(records)
    rows: list[dict[str, Any]] = []
    expert_pair_rows: list[dict[str, Any]] = []
    for record in row_list:
        goal = np.asarray(record.goal, dtype=np.float64)
        d_goal = record.distance_to_goal
        net_stats = _traj_stats(record.network_points, goal)
        row: dict[str, Any] = {
            "step": record.step,
            "d_goal_m": round(d_goal, 2),
            "near": bool(d_goal < near_threshold_m),
            **{"net_" + key: value for key, value in net_stats.items()},
        }
        if record.expert_points is not None:
            exp_stats = _traj_stats(record.expert_points, goal)
            row.update({f"exp_{key}": value for key, value in exp_stats.items()})
            pair = _pair_error(record.network_points, record.expert_points)
            if pair is not None:
                row.update({f"vs_{key}": value for key, value in pair.items()})
                expert_pair_rows.append(row)
        row.update(_meta_fields(record.meta))
        rows.append(row)

    # 跨重规划一致性：按样本聚合相邻行。
    consistency_rows = _consistency_per_sample(rows, consistency_yaw_deg)
    near_rows = [row for row in rows if row["near"]]

    overall: dict[str, Any] = {}
    if expert_pair_rows:
        overall.update(_aggregate_pair(expert_pair_rows))
    if near_rows:
        overall.update(_aggregate_near(near_rows))
    if consistency_rows:
        overall.update(_aggregate_consistency(consistency_rows))
    overall["samples"] = len(_unique_task_ids(rows))
    overall["replans"] = len(rows)
    overall["near_replans"] = len(near_rows)

    groups: dict[str, dict[str, Any]] = {}
    for dimension in ("scene", "task_type", "maneuver", "noise_level"):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(dimension, "unknown"))
            buckets.setdefault(key, []).append(row)
        groups[dimension] = {}
        for key, bucket in sorted(buckets.items()):
            near_bucket = [r for r in bucket if r["near"]]
            pair_bucket = [r for r in bucket if "vs_ade_m" in r]
            agg: dict[str, Any] = {"replans": len(bucket)}
            if pair_bucket:
                agg.update(_aggregate_pair(pair_bucket))
            if near_bucket:
                agg.update(_aggregate_near(near_bucket))
            if near_bucket:
                agg["samples"] = len({r.get("task_id") for r in bucket})
            groups[dimension][key] = agg

    report = {"overall": overall, "groups": groups}
    return report, rows


def _meta_fields(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(meta.get("task_id", "unknown")),
        "scene": str(meta.get("scene_name", "unknown")),
        "task_type": str(meta.get("task_type", "unknown")),
        "maneuver": str(meta.get("maneuver", "unknown")),
        "noise_level": str(meta.get("noise_level", "unknown")),
    }


def _unique_task_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("task_id", "")) for row in rows}


def _aggregate_pair(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "vs_ade_m": round(float(np.mean([r["vs_ade_m"] for r in rows])), 3),
        "vs_fde_m": round(float(np.mean([r["vs_fde_m"] for r in rows])), 3),
        "vs_yaw_mae_deg": round(
            float(np.mean([r["vs_yaw_mae_deg"] for r in rows])), 2
        ),
        "pair_replans": len(rows),
    }


def _aggregate_near(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "near_len_m": round(float(np.mean([r["net_length_m"] for r in rows])), 3),
        "near_end_to_goal_m": round(
            float(np.mean([r["net_end_to_goal_m"] for r in rows])), 3
        ),
        "near_end_yaw_err_deg": round(
            float(np.mean([abs(r["net_end_yaw_err_deg"]) for r in rows])), 2
        ),
        "near_flips": round(float(np.mean([r["net_flips"] for r in rows])), 3),
    }


def _consistency_per_sample(rows: list[dict[str, Any]], threshold_deg: float) -> list[dict[str, Any]]:
    """按任务 ID 分组，对相邻重规划计算终点航向/长度跳变次数。"""
    consistency: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("task_id", "")), []).append(row)
    threshold = float(np.deg2rad(threshold_deg))
    for task_id, sample_rows in buckets.items():
        sample_rows.sort(key=lambda r: int(r.get("step", 0)))
        yaw_jumps = 0
        length_jumps = 0
        for prev, curr in zip(sample_rows[:-1], sample_rows[1:]):
            if prev.get("net_end_yaw_err_deg") is None or curr.get("net_end_yaw_err_deg") is None:
                continue
            yaw_delta = abs(_wrap(np.deg2rad(curr["net_end_yaw_err_deg"] - prev["net_end_yaw_err_deg"])))
            if yaw_delta > threshold:
                yaw_jumps += 1
            prev_len = prev.get("net_length_m")
            curr_len = curr.get("net_length_m")
            if prev_len and curr_len:
                ratio = curr_len / prev_len if prev_len else 0.0
                if ratio > 3.0 or ratio < 1.0 / 3.0:
                    length_jumps += 1
        consistency.append({
            "task_id": task_id,
            "replans": len(sample_rows),
            "near_replans": len([r for r in sample_rows if r["near"]]),
            "yaw_jumps": yaw_jumps,
            "length_jumps": length_jumps,
        })
    return consistency


def _aggregate_consistency(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "consistency_yaw_jumps_per_sample": round(
            float(np.mean([r["yaw_jumps"] for r in rows])), 3
        ),
        "consistency_length_jumps_per_sample": round(
            float(np.mean([r["length_jumps"] for r in rows])), 3
        ),
    }


__all__ = [
    "CONSISTENCY_YAW_DEG",
    "FLIP_ANGLE_RAD",
    "NEAR_THRESHOLD_M",
    "VisitedStateRecord",
    "analyze_visited_state_records",
]