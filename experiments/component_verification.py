"""E5 分层验收规则；局部证据与整任务资格分别表达。"""
from __future__ import annotations

from collections import Counter
import numpy as np

SCHEMA = 'production_component_verification_v1'


def pose_errors(pose, target):
    delta = np.asarray(pose)[:3] - np.asarray(target)[:3]
    return float(np.linalg.norm(delta[:2])), float(abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))))


def locate_failure(*, success, collision, boundary_errors, reference_unsafe,
                   connection_unsafe, execution_unsafe):
    """返回观测签名而非唯一因果；无反证时不以排除法认定链接故障。"""
    if boundary_errors:
        return 'boundary_mismatch'
    if success:
        return 'task_reached_with_observed_risks' if reference_unsafe or execution_unsafe else 'task_reached'
    if collision:
        if reference_unsafe:
            return 'unsafe_consumed_reference'
        if connection_unsafe:
            return 'unsafe_state_reference_bridge'
        if execution_unsafe:
            return 'execution_left_safe_reference'
        return 'unresolved_collision'
    return 'unresolved_nonarrival'


def acceptance(rows, *, full_coverage, checks_ok):
    count = len(rows)
    successes = sum(bool(r['success']) for r in rows)
    collisions = sum(bool(r['collision']) for r in rows)
    metrics_pass = bool(count and successes / count >= .7 and collisions / count <= .1)
    return {'evaluated_samples': count, 'successes': successes, 'collisions': collisions,
            'success_rate': successes / count if count else None,
            'collision_rate': collisions / count if count else None,
            'required_success_rate': .7, 'allowed_collision_rate': .1,
            'task_metrics_pass': metrics_pass, 'checks_ok': bool(checks_ok),
            'full_300_coverage': bool(full_coverage and count == 300),
            'production_qualified': bool(full_coverage and count == 300 and checks_ok and metrics_pass)}


def validate_baseline(report, indices):
    expected = {'name': 'E5', 'bev_source': 'sensor', 'trajectory_source': 'network',
                'executor': 'mpc_vehicle', 'safety_mode': 'none'}
    if report.get('status') != 'completed' or any(report['experiment'].get(k) != v for k, v in expected.items()):
        raise ValueError('要求已完成、无干预的原生E5基线；GT/ideal/回退不提供生产资格')
    available = [r['meta']['dataset_index'] for r in report['episodes']]
    protocol = report['protocol']
    if (not indices or len(indices) != len(set(indices)) or len(available) != len(set(available))
            or not set(indices) <= set(available)
            or len(protocol['evaluated_indices']) != len(set(protocol['evaluated_indices']))
            or len(protocol['selected_indices']) != len(set(protocol['selected_indices']))
            or set(available) != set(protocol['evaluated_indices'])
            or set(available) != set(protocol['selected_indices'])
            or report.get('reconstruct_failures') or report['overall'].get('reconstruct_failures', 0)):
        raise ValueError('基线/所选索引缺失、重复或分母不完整')
    if (protocol['mpc_dt'] != .1 or protocol['max_steps'] != 600 or protocol['replan_every'] != 10
            or protocol['control_seed'] != 0):
        raise ValueError('v1验证要求原600步/10步重规划/dt=.1/seed=0协议')
    return {r['meta']['dataset_index']: r for r in report['episodes']}


def trajectory_demands(points, dt, *, max_v, max_omega):
    """参考点间运动需求仅作诊断，不把离散差分当动力学可行性证明。"""
    points = np.asarray(points)
    delta = np.diff(points.astype(np.float64), axis=0)
    speed = np.linalg.norm(delta[:, :2], axis=1) / dt
    yaw_rate = np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))) / dt
    return {'max_segment_speed_mps': float(speed.max(initial=0)),
            'max_segment_yaw_rate_radps': float(yaw_rate.max(initial=0)),
            'segments_over_vehicle_speed': int(np.sum(speed > max_v)),
            'segments_over_vehicle_yaw_rate': int(np.sum(yaw_rate > max_omega))}


def layer_status(errors, *, quality=None):
    return {'contract_status': 'failed' if errors else 'passed', 'errors': errors,
            'quality_status': quality or 'diagnostic_only', 'module_qualified': False}


def aggregate_layers(episodes):
    names = ('L1', 'L2', 'L3', 'L1_L2', 'L2_L3')
    return {name: {'contract_status': 'failed' if any(e['layers'][name]['errors'] for e in episodes) else 'passed',
                   'contract_failed_episodes': [e['dataset_index'] for e in episodes if e['layers'][name]['errors']],
                   'quality_status_counts': dict(Counter(e['layers'][name]['quality_status'] for e in episodes)),
                   'module_qualified': False,
                   'qualification_note': '结构一致/局部重放不等于模块已证明安全与收敛；质量证据见逐回合与整任务联合门槛。'}
            for name in names}
