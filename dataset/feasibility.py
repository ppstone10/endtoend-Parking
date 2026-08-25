"""履带钻机专家轨迹的运动学、扫掠碰撞与模型版本审计。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from sim.vehicle_config import VehicleConfig


_POSITION_EPSILON = 1e-6
_YAW_EPSILON = 1e-6
_LIMIT_EPSILON = 1e-6


@dataclass(frozen=True)
class TrajectoryFeasibilityAudit:
    """一条中心位姿轨迹相对履带理论模型的可序列化审计。"""

    model_name: str
    model_version: str
    segment_count: int
    moving_segment_count: int
    pivot_segment_count: int
    stationary_segment_count: int
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    max_lateral_residual_m: float
    start_position_error_m: float
    start_yaw_error_rad: float
    goal_position_error_m: float
    goal_yaw_error_rad: float
    endpoints_match: bool
    collision_free: bool
    kinematically_feasible: bool
    feasible: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "segment_count": self.segment_count,
            "moving_segment_count": self.moving_segment_count,
            "pivot_segment_count": self.pivot_segment_count,
            "stationary_segment_count": self.stationary_segment_count,
            "max_linear_speed_mps": self.max_linear_speed_mps,
            "max_angular_speed_radps": self.max_angular_speed_radps,
            "max_lateral_residual_m": self.max_lateral_residual_m,
            "start_position_error_m": self.start_position_error_m,
            "start_yaw_error_rad": self.start_yaw_error_rad,
            "goal_position_error_m": self.goal_position_error_m,
            "goal_yaw_error_rad": self.goal_yaw_error_rad,
            "endpoints_match": self.endpoints_match,
            "collision_free": self.collision_free,
            "kinematically_feasible": self.kinematically_feasible,
            "feasible": self.feasible,
        }


def audit_trajectory_feasibility(
    points: np.ndarray,
    *,
    dt: float,
    max_v: float,
    max_omega: float,
    pose_free: Callable[[float, float, float], bool] | None,
    model_metadata: dict[str, Any],
    swept_segment_free: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    lateral_tolerance_m: float = 0.02,
    expected_start_pose: np.ndarray | tuple[float, float, float] | None = None,
    expected_goal_pose: np.ndarray | tuple[float, float, float] | None = None,
    endpoint_position_tolerance_m: float = 1e-3,
    endpoint_yaw_tolerance_rad: float = 1e-3,
) -> TrajectoryFeasibilityAudit:
    """从中心位姿序列复算线/角速度、横向残差和扫掠碰撞。"""
    source = np.asarray(points)
    trajectory = np.asarray(source, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 2:
        raise ValueError("专家轨迹必须是至少两个点的 (N, 3) 数组")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("专家轨迹必须只包含有限值")
    limits = (
        dt,
        max_v,
        max_omega,
        lateral_tolerance_m,
        endpoint_position_tolerance_m,
        endpoint_yaw_tolerance_rad,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("dt、速度上限与横向容差必须为有限正数")

    delta_xy = np.diff(trajectory[:, :2], axis=0)
    distance = np.linalg.norm(delta_xy, axis=1)
    delta_yaw = _wrap_angles(np.diff(trajectory[:, 2]))
    moving = distance > _POSITION_EPSILON
    pivot = ~moving & (np.abs(delta_yaw) > _YAW_EPSILON)
    stationary = ~moving & ~pivot

    linear_speed = distance / dt
    angular_speed = np.abs(delta_yaw) / dt
    midpoint_yaw = trajectory[:-1, 2] + delta_yaw / 2.0
    lateral_residual = np.abs(
        -np.sin(midpoint_yaw) * delta_xy[:, 0]
        + np.cos(midpoint_yaw) * delta_xy[:, 1]
    )
    max_linear = float(linear_speed.max(initial=0.0))
    max_angular = float(angular_speed.max(initial=0.0))
    max_lateral = float(lateral_residual[moving].max(initial=0.0))
    # 轨迹归档为 float32；在较大的世界坐标上相邻点相减后再除以 dt，
    # 量化误差会被放大到数微米/秒。余量按源精度与坐标尺度推导，
    # 避免把严格贴合速度上限的合法规划段误判为超速。
    source_epsilon = (
        float(np.finfo(source.dtype).eps)
        if np.issubdtype(source.dtype, np.floating)
        else 0.0
    )
    position_scale = max(1.0, float(np.abs(trajectory[:, :2]).max(initial=0.0)))
    yaw_scale = max(1.0, float(np.abs(trajectory[:, 2]).max(initial=0.0)))
    linear_limit_epsilon = max(
        _LIMIT_EPSILON, 4.0 * source_epsilon * position_scale / dt
    )
    angular_limit_epsilon = max(
        _LIMIT_EPSILON, 4.0 * source_epsilon * yaw_scale / dt
    )
    kinematically_feasible = bool(
        max_linear <= max_v + linear_limit_epsilon
        and max_angular <= max_omega + angular_limit_epsilon
        and max_lateral <= lateral_tolerance_m + _LIMIT_EPSILON
    )

    collision_free = True
    if pose_free is not None:
        collision_free = all(
            pose_free(float(x), float(y), float(yaw)) for x, y, yaw in trajectory
        )
        if collision_free and swept_segment_free is not None:
            collision_free = all(
                swept_segment_free(trajectory[index], trajectory[index + 1])
                for index in range(len(trajectory) - 1)
            )

    start_position_error, start_yaw_error = _endpoint_error(
        trajectory[0], expected_start_pose
    )
    goal_position_error, goal_yaw_error = _endpoint_error(
        trajectory[-1], expected_goal_pose
    )
    endpoints_match = bool(
        start_position_error <= endpoint_position_tolerance_m + _LIMIT_EPSILON
        and start_yaw_error <= endpoint_yaw_tolerance_rad + _LIMIT_EPSILON
        and goal_position_error <= endpoint_position_tolerance_m + _LIMIT_EPSILON
        and goal_yaw_error <= endpoint_yaw_tolerance_rad + _LIMIT_EPSILON
    )
    model_name = str(model_metadata.get("name", "unknown"))
    model_version = str(model_metadata.get("model_version", "unknown"))
    return TrajectoryFeasibilityAudit(
        model_name=model_name,
        model_version=model_version,
        segment_count=len(trajectory) - 1,
        moving_segment_count=int(np.count_nonzero(moving)),
        pivot_segment_count=int(np.count_nonzero(pivot)),
        stationary_segment_count=int(np.count_nonzero(stationary)),
        max_linear_speed_mps=max_linear,
        max_angular_speed_radps=max_angular,
        max_lateral_residual_m=max_lateral,
        start_position_error_m=start_position_error,
        start_yaw_error_rad=start_yaw_error,
        goal_position_error_m=goal_position_error,
        goal_yaw_error_rad=goal_yaw_error,
        endpoints_match=endpoints_match,
        collision_free=bool(collision_free),
        kinematically_feasible=kinematically_feasible,
        feasible=bool(collision_free and kinematically_feasible and endpoints_match),
    )


def summarize_trajectory_feasibility(
    trajs: np.ndarray,
    masks: np.ndarray,
    *,
    dt: float | np.ndarray,
    metadata: list[dict[str, Any]] | None,
    vehicle_config: VehicleConfig,
    states: np.ndarray | None = None,
    goals: np.ndarray | None = None,
) -> dict[str, Any]:
    """从归档独立复算运动学，并核对生成期碰撞证据与模型版本。"""
    trajectories = np.asarray(trajs)
    trajectory_masks = np.asarray(masks)
    if trajectories.shape[:2] != trajectory_masks.shape:
        raise ValueError("trajs 与 masks 形状不一致")
    if metadata is not None and (
        not isinstance(metadata, list) or len(metadata) != len(trajectories)
    ):
        raise ValueError("task_meta 数量必须与样本数量一致")
    archive_dt = float(np.asarray(dt).reshape(-1)[0])
    expected_model = vehicle_config.to_metadata()
    state_values = None if states is None else np.asarray(states)
    goal_values = None if goals is None else np.asarray(goals)
    if state_values is not None and len(state_values) != len(trajectories):
        raise ValueError("states 数量必须与样本数量一致")
    if goal_values is not None and len(goal_values) != len(trajectories):
        raise ValueError("goals 数量必须与样本数量一致")
    result: dict[str, Any] = {
        "model_name": vehicle_config.name,
        "model_version": vehicle_config.model_version,
        "audited_sample_count": 0,
        "feasible_count": 0,
        "infeasible_count": 0,
        "invalid_trajectory_count": 0,
        "missing_collision_evidence_count": 0,
        "model_mismatch_count": 0,
        "pivot_segment_count": 0,
        "moving_segment_count": 0,
        "max_linear_speed_mps": 0.0,
        "max_angular_speed_radps": 0.0,
        "max_lateral_residual_m": 0.0,
        "feasibility_rate": 0.0,
    }
    if metadata is None:
        result["missing_collision_evidence_count"] = int(len(trajectories))
        result["model_mismatch_count"] = int(len(trajectories))
        return result

    for sample_index, (trajectory, mask, item) in enumerate(
        zip(trajectories, trajectory_masks, metadata)
    ):
        count = int(np.count_nonzero(mask))
        try:
            audit = audit_trajectory_feasibility(
                trajectory[:count],
                dt=archive_dt,
                max_v=vehicle_config.plan_v,
                max_omega=vehicle_config.plan_max_omega,
                pose_free=None,
                model_metadata=expected_model,
                expected_start_pose=(
                    None
                    if state_values is None
                    else state_values[sample_index, :3]
                ),
                expected_goal_pose=(
                    None if goal_values is None else goal_values[sample_index, :3]
                ),
            )
        except (TypeError, ValueError):
            result["invalid_trajectory_count"] += 1
            continue
        result["audited_sample_count"] += 1
        dataset_meta = item.get("dataset", {}) if isinstance(item, dict) else {}
        stored_audit = dataset_meta.get("feasibility_audit", {})
        collision_evidence = (
            isinstance(stored_audit, dict)
            and stored_audit.get("collision_free") is True
            and stored_audit.get("feasible") is True
        )
        model_matches = dataset_meta.get("vehicle_model") == expected_model
        if not collision_evidence:
            result["missing_collision_evidence_count"] += 1
        if not model_matches:
            result["model_mismatch_count"] += 1
        feasible = bool(
            audit.kinematically_feasible
            and audit.endpoints_match
            and collision_evidence
            and model_matches
        )
        audit = replace(
            audit,
            collision_free=collision_evidence,
            feasible=feasible,
        )
        result["feasible_count" if feasible else "infeasible_count"] += 1
        result["pivot_segment_count"] += audit.pivot_segment_count
        result["moving_segment_count"] += audit.moving_segment_count
        for key in (
            "max_linear_speed_mps",
            "max_angular_speed_radps",
            "max_lateral_residual_m",
        ):
            result[key] = max(float(result[key]), float(getattr(audit, key)))

    audited = int(result["audited_sample_count"])
    if audited:
        result["feasibility_rate"] = result["feasible_count"] / audited
    return result


def require_trajectory_feasibility(summary: dict[str, Any]) -> None:
    """严格数据门禁：轨迹、碰撞证据和模型版本必须全部通过。"""
    audit = summary.get("trajectory_feasibility")
    if not isinstance(audit, dict):
        raise ValueError("轨迹可行性门禁失败：统计中缺少 trajectory_feasibility")
    infeasible = int(audit.get("infeasible_count", 0))
    invalid = int(audit.get("invalid_trajectory_count", 0))
    missing = int(audit.get("missing_collision_evidence_count", 0))
    mismatch = int(audit.get("model_mismatch_count", 0))
    if infeasible or invalid or missing or mismatch:
        raise ValueError(
            "轨迹可行性门禁失败："
            f"不可行 {infeasible}、无效 {invalid}、"
            f"缺少碰撞证据 {missing}、模型不匹配 {mismatch}"
        )


def _wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def _endpoint_error(
    actual: np.ndarray,
    expected: np.ndarray | tuple[float, float, float] | None,
) -> tuple[float, float]:
    if expected is None:
        return 0.0, 0.0
    target = np.asarray(expected, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("期望起终位姿必须是有限 (3,) 数组")
    position_error = float(np.linalg.norm(actual[:2] - target[:2]))
    yaw_error = abs(float(_wrap_angles(np.asarray([actual[2] - target[2]]))[0]))
    return position_error, yaw_error


__all__ = [
    "TrajectoryFeasibilityAudit",
    "audit_trajectory_feasibility",
    "summarize_trajectory_feasibility",
    "require_trajectory_feasibility",
]
