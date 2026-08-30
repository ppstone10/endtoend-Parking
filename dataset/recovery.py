"""从学习器闭环访问状态生成专家恢复监督。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from interfaces import GoalPose, VehicleState
from sim import Maneuver

from .generator import DatasetGenerator, TrainingSample
from .maneuver import audit_maneuver_consistency


@dataclass(frozen=True)
class RecoveryCandidate:
    rollout_step: int
    state: VehicleState
    deviation_m: float
    position_deviation_m: float
    yaw_deviation_rad: float
    trigger: str = "stride"


def select_recovery_candidates(
    states: Iterable[VehicleState],
    collisions: Iterable[bool],
    expert_points: np.ndarray,
    *,
    stride: int,
    min_deviation_m: float,
    min_yaw_deviation_rad: float | None = None,
    yaw_radius_m: float = 1.0,
) -> list[RecoveryCandidate]:
    """确定性选择偏离专家分布且尚未碰撞的闭环状态。"""
    if (
        stride <= 0
        or not np.isfinite(min_deviation_m)
        or min_deviation_m < 0.0
        or (
            min_yaw_deviation_rad is not None
            and (
                not np.isfinite(min_yaw_deviation_rad)
                or min_yaw_deviation_rad < 0.0
            )
        )
        or not np.isfinite(yaw_radius_m)
        or yaw_radius_m <= 0.0
    ):
        raise ValueError("stride/位置航向阈值/yaw_radius_m 参数无效")
    expert = np.asarray(expert_points, dtype=np.float64)
    if expert.ndim != 2 or expert.shape[0] == 0 or expert.shape[1] < 3:
        raise ValueError("expert_points 必须为非空 (N,>=3)")
    state_values = list(states)
    collision_values = list(collisions)
    if len(state_values) != len(collision_values):
        raise ValueError("states 与 collisions 必须逐步对齐")
    candidates: list[RecoveryCandidate] = []
    for offset, (state, collision) in enumerate(
        zip(state_values, collision_values)
    ):
        step = offset + 1
        pre_collision = (
            offset + 1 < len(collision_values) and collision_values[offset + 1]
        )
        if collision or (step % stride != 0 and not pre_collision):
            continue
        position_errors = np.hypot(expert[:, 0] - state.x, expert[:, 1] - state.y)
        yaw_errors = np.abs(
            np.arctan2(
                np.sin(expert[:, 2] - state.yaw),
                np.cos(expert[:, 2] - state.yaw),
            )
        )
        combined = np.hypot(position_errors, yaw_radius_m * yaw_errors)
        match = int(np.argmin(combined))
        position_deviation = float(position_errors[match])
        yaw_deviation = float(yaw_errors[match])
        if (
            pre_collision
            or position_deviation >= min_deviation_m
            or (
                min_yaw_deviation_rad is not None
                and yaw_deviation >= min_yaw_deviation_rad
            )
        ):
            candidates.append(
                RecoveryCandidate(
                    step,
                    state,
                    float(combined[match]),
                    position_deviation,
                    yaw_deviation,
                    "pre_collision" if pre_collision else "stride",
                )
            )
    return sorted(
        candidates,
        key=lambda item: (
            0 if item.trigger == "pre_collision" else 1,
            -item.deviation_m,
            item.rollout_step,
        ),
    )


def build_recovery_sample(
    candidate: RecoveryCandidate,
    *,
    source_index: int,
    source_metadata: dict[str, Any],
    goal: GoalPose,
    planner,
    pipeline,
    checkpoint_identity: str,
) -> TrainingSample:
    """从学习器状态规划、感知并生成一条可审计恢复样本。"""
    trajectory = planner.plan(candidate.state, goal)
    forward = audit_maneuver_consistency(trajectory.points, Maneuver.FORWARD)
    reverse = audit_maneuver_consistency(trajectory.points, Maneuver.REVERSE)
    maneuver = (
        Maneuver.FORWARD
        if forward.requested_distance_ratio >= reverse.requested_distance_ratio
        else Maneuver.REVERSE
    )
    maneuver_audit = forward if maneuver is Maneuver.FORWARD else reverse
    feasibility, vehicle_model = DatasetGenerator._audit_feasibility(
        planner, trajectory, candidate.state, goal
    )
    if not maneuver_audit.consistent:
        raise ValueError("恢复专家轨迹没有稳定的主导机动方向")
    if not feasibility.feasible:
        raise ValueError("恢复专家轨迹未通过独立可行性审计")
    set_target_goals = getattr(pipeline, "set_target_goals", None)
    if callable(set_target_goals):
        set_target_goals([goal])
    bev = pipeline.capture_bev(
        candidate.state.x, candidate.state.y, candidate.state.yaw
    )

    metadata = deepcopy(source_metadata)
    difficulty = metadata.setdefault("difficulty", {})
    difficulty["maneuver"] = maneuver.value
    dataset_meta = metadata.setdefault("dataset", {})
    dataset_meta.update(
        {
            "source": "closed_loop_recovery",
            "goal_policy": "original_selected_goal",
            "maneuver_audit": maneuver_audit.to_metadata(),
            "vehicle_model": vehicle_model,
            "feasibility_audit": feasibility.to_metadata(),
        }
    )
    metadata["recovery"] = {
        "schema_version": 1,
        "source_dataset_index": int(source_index),
        "source_task_id": str(source_metadata.get("task_id", "")),
        "rollout_step": candidate.rollout_step,
        "deviation_m": candidate.deviation_m,
        "position_deviation_m": candidate.position_deviation_m,
        "yaw_deviation_rad": candidate.yaw_deviation_rad,
        "trigger": candidate.trigger,
        "policy_checkpoint": checkpoint_identity,
        "label_policy": "expert_replan_from_learner_state",
    }
    return TrainingSample(
        bev=bev,
        goal=goal,
        state=candidate.state,
        expert_trajectory=trajectory,
        task_meta=metadata,
    )


__all__ = [
    "RecoveryCandidate",
    "build_recovery_sample",
    "select_recovery_candidates",
]
