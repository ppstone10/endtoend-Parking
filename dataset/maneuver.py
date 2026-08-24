"""专家轨迹实际行驶方向与 Task 机动要求的一致性分析。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.tasks import Maneuver


DEFAULT_MIN_REQUESTED_DISTANCE_RATIO = 0.5
_DISTANCE_EPSILON = 1e-9


@dataclass(frozen=True)
class ManeuverAudit:
    """一条专家轨迹相对请求机动方向的可序列化审计结果。"""

    requested_maneuver: Maneuver
    forward_distance_m: float
    reverse_distance_m: float
    requested_distance_ratio: float
    minimum_requested_distance_ratio: float
    direction_changes: int
    consistent: bool

    @property
    def total_distance_m(self) -> float:
        return self.forward_distance_m + self.reverse_distance_m

    @property
    def forward_distance_ratio(self) -> float:
        return self.forward_distance_m / self.total_distance_m

    @property
    def reverse_distance_ratio(self) -> float:
        return self.reverse_distance_m / self.total_distance_m

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_maneuver": self.requested_maneuver.value,
            "forward_distance_m": self.forward_distance_m,
            "reverse_distance_m": self.reverse_distance_m,
            "total_distance_m": self.total_distance_m,
            "forward_distance_ratio": self.forward_distance_ratio,
            "reverse_distance_ratio": self.reverse_distance_ratio,
            "requested_distance_ratio": self.requested_distance_ratio,
            "minimum_requested_distance_ratio": self.minimum_requested_distance_ratio,
            "direction_changes": self.direction_changes,
            "consistent": self.consistent,
        }


def audit_maneuver_consistency(
    points: np.ndarray,
    requested_maneuver: Maneuver | str,
    *,
    minimum_requested_distance_ratio: float = DEFAULT_MIN_REQUESTED_DISTANCE_RATIO,
) -> ManeuverAudit:
    """按轨迹切向投影统计方向距离，并判定请求方向是否占主导。"""
    threshold = validate_minimum_requested_distance_ratio(
        minimum_requested_distance_ratio
    )
    maneuver = Maneuver(requested_maneuver)
    directions, segment_lengths = trajectory_segment_directions(points)
    total_distance = float(segment_lengths.sum())
    if total_distance <= _DISTANCE_EPSILON:
        raise ValueError("专家轨迹总行驶距离必须为正")

    forward_distance = float(segment_lengths[directions > 0].sum())
    reverse_distance = float(segment_lengths[directions < 0].sum())
    requested_distance = (
        forward_distance if maneuver == Maneuver.FORWARD else reverse_distance
    )
    requested_ratio = requested_distance / total_distance
    moving_directions = directions[segment_lengths > _DISTANCE_EPSILON]
    direction_changes = int(
        np.count_nonzero(moving_directions[1:] != moving_directions[:-1])
    )
    return ManeuverAudit(
        requested_maneuver=maneuver,
        forward_distance_m=forward_distance,
        reverse_distance_m=reverse_distance,
        requested_distance_ratio=requested_ratio,
        minimum_requested_distance_ratio=threshold,
        direction_changes=direction_changes,
        consistent=requested_ratio + _DISTANCE_EPSILON >= threshold,
    )


def trajectory_segment_directions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每段实际行驶方向（1 前进/-1 倒车）与平面距离。"""
    trajectory = np.asarray(points, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 2:
        raise ValueError("专家轨迹必须是至少两个点的 (N, 3) 数组")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("专家轨迹必须只包含有限值")

    delta = np.diff(trajectory[:, :2], axis=0)
    segment_lengths = np.linalg.norm(delta, axis=1)
    heading = trajectory[:-1, 2]
    signed_progress = delta[:, 0] * np.cos(heading) + delta[:, 1] * np.sin(heading)
    directions = np.where(signed_progress < -_DISTANCE_EPSILON, -1, 1).astype(
        np.int8
    )
    return directions, segment_lengths


def summarize_maneuver_consistency(
    trajs: np.ndarray,
    masks: np.ndarray,
    metadata: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """独立复算归档内机动一致性并按请求方向和任务单元汇总。"""
    trajectories = np.asarray(trajs)
    trajectory_masks = np.asarray(masks)
    if trajectories.shape[:2] != trajectory_masks.shape:
        raise ValueError("trajs 与 masks 形状不一致")
    if metadata is not None and (
        not isinstance(metadata, list) or len(metadata) != len(trajectories)
    ):
        raise ValueError("task_meta 数量必须与样本数量一致")

    result: dict[str, Any] = {
        "minimum_requested_distance_ratio": DEFAULT_MIN_REQUESTED_DISTANCE_RATIO,
        "audited_sample_count": 0,
        "consistent_count": 0,
        "inconsistent_count": 0,
        "invalid_trajectory_count": 0,
        "missing_maneuver_count": 0,
        "consistency_rate": 0.0,
        "requested_maneuver_counts": {},
        "inconsistent_scene_task_counts": {},
    }
    if metadata is None:
        result["missing_maneuver_count"] = int(len(trajectories))
        return result

    requested_counts: dict[str, Counter[str]] = {}
    inconsistent_strata: Counter[str] = Counter()
    for trajectory, mask, item in zip(trajectories, trajectory_masks, metadata):
        maneuver = item.get("difficulty", {}).get("maneuver")
        if maneuver is None:
            result["missing_maneuver_count"] += 1
            continue
        count = int(np.count_nonzero(mask))
        try:
            audit = audit_maneuver_consistency(trajectory[:count], maneuver)
        except (TypeError, ValueError):
            result["invalid_trajectory_count"] += 1
            continue

        result["audited_sample_count"] += 1
        outcome = "consistent" if audit.consistent else "inconsistent"
        result[f"{outcome}_count"] += 1
        requested_counts.setdefault(audit.requested_maneuver.value, Counter())[outcome] += 1
        if not audit.consistent:
            stratum = (
                f"{item.get('scene_name', 'unknown')}/"
                f"{item.get('task_type', 'unknown')}"
            )
            inconsistent_strata[stratum] += 1

    audited = int(result["audited_sample_count"])
    if audited:
        result["consistency_rate"] = result["consistent_count"] / audited
    result["requested_maneuver_counts"] = {
        maneuver: {
            "consistent": counts["consistent"],
            "inconsistent": counts["inconsistent"],
        }
        for maneuver, counts in sorted(requested_counts.items())
    }
    result["inconsistent_scene_task_counts"] = dict(
        sorted(inconsistent_strata.items())
    )
    return result


def require_maneuver_consistency(summary: dict[str, Any]) -> None:
    """严格数据门禁：存在不一致、无效轨迹或缺失声明时拒绝归档。"""
    audit = summary.get("maneuver_consistency")
    if not isinstance(audit, dict):
        raise ValueError("机动一致性门禁失败：统计中缺少 maneuver_consistency")
    inconsistent = int(audit.get("inconsistent_count", 0))
    invalid = int(audit.get("invalid_trajectory_count", 0))
    missing = int(audit.get("missing_maneuver_count", 0))
    if inconsistent or invalid or missing:
        raise ValueError(
            "机动一致性门禁失败："
            f"不一致 {inconsistent}、无效轨迹 {invalid}、缺失声明 {missing}"
        )


def validate_minimum_requested_distance_ratio(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.5 <= threshold <= 1.0:
        raise ValueError("最低请求方向占比必须位于 [0.5, 1.0]")
    return threshold
