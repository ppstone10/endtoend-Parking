"""只读旧轨迹的轮式必要条件诊断，不授予标签复用资格。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np


def load_legacy_source(path):
    """故意不读取bevs；完整几何/闭环校准另行执行。"""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["schema_version"]) != 2:
            raise ValueError("旧数据审核要求schema v2")
        data = {key: archive[key] for key in ("trajs", "masks", "states", "task_meta", "dt")}
    data["task_meta"] = [json.loads(str(item)) for item in data["task_meta"]]
    n = len(data["trajs"])
    if (data["trajs"].ndim != 3 or data["trajs"].shape[2] != 3
            or data["masks"].shape != data["trajs"].shape[:2]
            or data["states"].shape != (n, 5) or len(data["task_meta"]) != n):
        raise ValueError("旧数据轨迹/mask/状态/元数据分母或形状不一致")
    data["dt"] = float(np.asarray(data["dt"]).reshape(-1)[0])
    if not np.isfinite(data["dt"]) or data["dt"] <= 0:
        raise ValueError("旧数据dt必须有限且为正")
    return data


def audit_wheeled_necessary_conditions(points, dt, config, *, initial_v=0.):
    source = np.asarray(points)
    poses = np.asarray(source, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 1 or not np.isfinite(poses).all():
        raise ValueError("审核要求非空有限N×3轨迹")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(initial_v):
        raise ValueError("审核dt/初始速度无效")
    epsilon = np.finfo(source.dtype).eps if np.issubdtype(source.dtype, np.floating) else 0.
    position_tolerance = float(max(1e-7, 16*epsilon*max(1., float(np.abs(poses[:, :2]).max()))))
    angle_tolerance = float(max(1e-7, 16*epsilon*max(1., float(np.abs(poses[:, 2]).max()))))
    reasons = {}
    if len(poses) > 1:
        turn = np.arctan2(np.sin(np.diff(poses[:, 2])), np.cos(np.diff(poses[:, 2])))
        body_distance = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        reasons["body_pivot"] = int(np.count_nonzero((body_distance <= position_tolerance) &
                                                    (np.abs(turn) > angle_tolerance)))
        rear = poses[:, :2]-config.center_from_rear*np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2])))
        displacement = np.diff(rear, axis=0)
        midpoint = poses[:-1, 2]+turn/2
        projected = displacement[:, 0]*np.cos(midpoint)+displacement[:, 1]*np.sin(midpoint)
        lateral = -displacement[:, 0]*np.sin(midpoint)+displacement[:, 1]*np.cos(midpoint)
        distance = projected/np.sinc(turn/(2*np.pi))
        moving = np.abs(distance) > position_tolerance
        speed = distance/dt
        speed_tolerance = position_tolerance/dt
        reasons["rear_lateral_residual"] = int(np.count_nonzero(np.abs(lateral) > 3*position_tolerance))
        reasons["curvature_excess"] = int(np.count_nonzero(np.abs(turn) >
                                        np.abs(distance)/config.min_turning_radius+angle_tolerance
                                        +3*position_tolerance/config.min_turning_radius))
        reasons["speed_excess"] = int(np.count_nonzero((speed > config.max_forward_speed+speed_tolerance) |
                                                       (speed < -config.max_reverse_speed-speed_tolerance)))
        previous_speed = np.r_[initial_v, speed[:-1]]
        rate = np.where(np.abs(speed) > np.abs(previous_speed), config.max_acceleration, config.max_deceleration)
        reasons["speed_slew_excess"] = int(np.count_nonzero(np.abs(speed-previous_speed) >
                                                           rate*dt+3*speed_tolerance))
        reasons["reversal_without_stop"] = int(np.count_nonzero((speed*previous_speed < 0) &
                                             (np.abs(speed) > speed_tolerance) & (np.abs(previous_speed) > speed_tolerance)))
        indices = np.flatnonzero(moving)
        delta = np.arctan(config.wheelbase*turn[moving]/distance[moving])
        # float32位姿经差分/除以短弧长后会放大误差，不能只给转角一个固定epsilon。
        magnitudes = np.abs(distance[moving])
        delta_uncertainty = config.wheelbase*(angle_tolerance/magnitudes
                              +3*position_tolerance*np.abs(turn[moving])/magnitudes**2)
        # 停留段未记录转角：只用两个可辨认转角间可用时间判断必要条件。
        reasons["steering_slew_excess"] = int(np.count_nonzero(np.abs(np.diff(delta)) >
                                                config.max_steering_rate*np.diff(indices)*dt
                                                +delta_uncertainty[:-1]+delta_uncertainty[1:]))
    reasons = {key: value for key, value in reasons.items() if value}
    return dict(classification="incompatible" if reasons else "necessary_conditions_only",
                reasons=reasons, direct_reuse_qualified=False,
                position_tolerance=position_tolerance, angle_tolerance=angle_tolerance,
                steering_and_geometry_certified=False)


def audit_archive(path, config):
    data = load_legacy_source(path)
    rows, reason_tasks, classifications = [], Counter(), Counter()
    for index, (points, mask, metadata) in enumerate(zip(data["trajs"], data["masks"], data["task_meta"])):
        if not np.logical_or(mask == 0, mask == 1).all() or np.any(mask[1:] > mask[:-1]):
            raise ValueError(f"第{index}条mask不是连续有效前缀")
        length = int(mask.sum())
        row = audit_wheeled_necessary_conditions(points[:length], data["dt"], config,
                                                 initial_v=float(data["states"][index, 3]))
        row.update(index=index, task_id=metadata["task_id"], scene_name=metadata["scene_name"])
        classifications[row["classification"]] += 1
        reason_tasks.update(row["reasons"].keys())
        rows.append(row)
    return dict(source=str(Path(path).resolve()), samples=len(rows), dt=data["dt"],
                model_identity=config.identity, classifications=dict(classifications),
                reason_task_counts=dict(reason_tasks), direct_reuse_qualified=False, rows=rows)
