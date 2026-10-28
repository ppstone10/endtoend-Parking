"""只读归档分诊：真实末态与网络最后一条参考的联合终端支撑分开。"""
from pathlib import Path
from collections import Counter
import numpy as np

from verification.candidate_study import read
from verification.terminal_study import SOURCE
from scripts.inspect_footprint_execution import inspect
from scripts.diagnose_closed_loop_failures import _sha256
from training.reporting import atomic_write_json


def diagnose(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('归档分诊使用新目录')
    inspect(SOURCE)
    report = read(SOURCE/'report.json')
    rows = []
    for result in report['rows']['control']:
        if result['success']:
            continue
        index = result['dataset_index']
        with np.load(SOURCE/'control'/f'episode-{index:04d}.npz', allow_pickle=False) as tape:
            points = tape[f'plan_{len(tape["plan_issue_steps"])-1:04d}']; goal = tape['goal']
            pos = np.linalg.norm(points[:, :2]-goal[:2], axis=1)
            yaw = np.abs(np.arctan2(np.sin(points[:, 2]-goal[2]), np.cos(points[:, 2]-goal[2])))
            rows.append({'dataset_index': index, 'scene_name': result['scene_name'],
                         'actual_final_pos_err': result['final_pos_err'], 'actual_final_yaw_err': result['final_yaw_err'],
                         'actual_position_in_tolerance': result['final_pos_err'] < .3,
                         'last_reference_points': len(points), 'last_reference_joint_goal_points': int(np.sum((pos < .3)&(yaw < np.deg2rad(10)))),
                         'last_reference_endpoint_pos_err': float(pos[-1]), 'last_reference_endpoint_yaw_err': float(yaw[-1]),
                         'intervened_steps': result['intervened_steps']})
    original = read('runs/verification/component-v2-migrated-20261023-v2/report.json')
    nonarrival = {e['dataset_index'] for e in original['episodes'] if e['axes']['contact_at_failure']=='unresolved_nonarrival'}
    s6 = [r for r in rows if r['dataset_index'] in nonarrival]
    payload = {'status': 'completed', 'source_report_sha256': _sha256(SOURCE/'report.json'), 'failed_episodes': len(rows),
               'failure_scenes': dict(Counter(r['scene_name'] for r in rows)),
               'failures_with_actual_position_in_tolerance': sum(r['actual_position_in_tolerance'] for r in rows),
               'failures_without_joint_goal_point_in_last_reference': sum(r['last_reference_joint_goal_points']==0 for r in rows),
               'original_18_nonarrival': {'samples':len(s6), 'no_control_intervention':sum(r['intervened_steps']==0 for r in s6),
                                        'no_joint_goal_point_in_last_reference':sum(r['last_reference_joint_goal_points']==0 for r in s6)},
               'episodes': rows, 'limits': '已停止裁剪后最后一条参考的支撑诊断；不代表未裁剪网络输出无合格点，不证明每条失败唯一由末参考导致；无新网络推理、无训练标签或val回灌。'}
    output.mkdir(parents=True, exist_ok=False); atomic_write_json(output/'report.json', payload)
    return payload
