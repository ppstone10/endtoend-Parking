"""实际发出参考的静态末点/目标支撑诊断，不代表车辆闭环结果。"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


def emitted_terminal_metrics(
    points: np.ndarray, local_goal: np.ndarray, emitted_length: int,
    target_length: int, metadata: dict[str, Any],
) -> dict[str, Any]:
    """只检查实际发出前缀；车位容差缺失时仍报告几何误差但不授予联合通过。"""
    if (points.ndim != 2 or points.shape[1] != 3 or local_goal.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(local_goal).all()
            or not 0 < emitted_length <= len(points) or not 0 < target_length <= len(points)):
        raise ValueError('实际终点诊断需要有限轨迹/局部目标与有效发出长度')
    emitted = points[:emitted_length]
    position = np.linalg.norm(emitted[:, :2] - local_goal[:2], axis=1)
    delta_yaw = emitted[:, 2] - local_goal[2]
    yaw = np.abs(np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw)))
    result = {
        'emitted_endpoint_position_error_m': float(position[-1]),
        'emitted_endpoint_yaw_error_rad': float(yaw[-1]),
        'emitted_suffix_points': max(0, emitted_length-target_length),
        'emitted_endpoint_joint_passed': None,
        'emitted_any_joint_passed': None,
        'terminal_status': 'not_observed',
        'terminal_missing_reason': 'goal_tolerance_missing',
    }
    dataset = metadata.get('dataset')
    goal = dataset.get('selected_goal') if isinstance(dataset, dict) else None
    if not isinstance(goal, dict) or 'tol_pos' not in goal or 'tol_yaw' not in goal:
        return result
    tolerances = [goal['tol_pos'], goal['tol_yaw']]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in tolerances):
        raise ValueError('selected_goal容差必须为正有限数')
    joint = (position < tolerances[0]) & (yaw < tolerances[1])
    result.update(terminal_status='observed', terminal_missing_reason=None,
                  emitted_endpoint_joint_passed=bool(joint[-1]),
                  emitted_any_joint_passed=bool(np.any(joint)))
    return result


def aggregate_emitted_terminal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """保留真实观察分母，缺失目标/容差不混入通过率。"""
    observed = [r for r in rows if r.get('terminal_status') == 'observed']
    geometric = [r for r in rows if 'emitted_endpoint_position_error_m' in r]
    n = len(observed)
    joint = sum(bool(r['emitted_endpoint_joint_passed']) for r in observed)
    return {
        'terminal_status': 'observed' if n == len(rows) else 'partial' if n else 'not_observed',
        'terminal_tolerance_missing_samples': sum(r.get('terminal_missing_reason') == 'goal_tolerance_missing' for r in rows),
        'terminal_goal_missing_samples': sum(r.get('terminal_missing_reason') == 'local_goal_missing' for r in rows),
        'emitted_endpoint_geometry_evaluated': len(geometric),
        'emitted_endpoint_joint_evaluated': n,
        'emitted_endpoint_joint_passed': joint,
        'emitted_endpoint_joint_rate': joint/n if n else None,
        'emitted_any_joint_passed': sum(bool(r['emitted_any_joint_passed']) for r in observed),
        'emitted_suffix_samples': sum(r['emitted_suffix_points'] > 0 for r in rows),
        'emitted_suffix_points': sum(r['emitted_suffix_points'] for r in rows),
        'emitted_endpoint_position_error_mean_m': float(np.mean([r['emitted_endpoint_position_error_m'] for r in geometric])) if geometric else None,
        'emitted_endpoint_yaw_error_mean_rad': float(np.mean([r['emitted_endpoint_yaw_error_rad'] for r in geometric])) if geometric else None,
    }
