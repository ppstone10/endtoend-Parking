"""独立复算归档动作和终端转向证书，不调用终端候选求解器。"""
from pathlib import Path
import time
import numpy as np

from dataset import DatasetGenerator
from experiments import validation_matrix as matrix
from interfaces import VehicleState, ControlCmd
from planner import RectangleFootprintCollisionChecker
from sim import VehicleConfig, DifferentialDriveModel
from training.reporting import atomic_write_json
from scripts.diagnose_closed_loop_failures import _sha256
from verification.identity import execution_identity
from verification.terminal_study import read, inspect, FILES


def audit(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('独立审计必须使用新目录')
    report = inspect(source)
    if report['mode'] != 'smoke':
        raise ValueError('当前独立证书审计只接受固定短检，正式不得暗中扩成批量重算')
    config = read(source/'config.json')
    data = DatasetGenerator.load(config['data']); manifest = matrix.load_dataset_manifest(Path(config['data']))
    vehicle = VehicleConfig(**manifest['vehicle_model']); model = DifferentialDriveModel(**vehicle.vehicle_model_kwargs())
    started = time.perf_counter(); summaries = []
    for index in config['indices']:
        if time.perf_counter()-started > 90:
            raise TimeoutError('独立短审计预算耗尽')
        restored = matrix.reconstruct_dataset_task(data['task_meta'][index], root_seed=manifest['seed'], vehicle=vehicle)
        params = dict(vehicle_length=vehicle.length, vehicle_width=vehicle.width, resolution=vehicle.collision_check_resolution)
        checker = RectangleFootprintCollisionChecker(restored.task.scene.env, collision_margin=0., **params)
        margin = RectangleFootprintCollisionChecker(restored.task.scene.env, collision_margin=.2, **params)
        record = read(source/'control'/f'episode-{index:04d}.json')
        terminal_steps = 0; violations = 0
        with np.load(source/'control'/f'episode-{index:04d}.npz', allow_pickle=False) as tape:
            state = VehicleState.from_array(tape['initial_state'])
            for before, after, cmd, sequence, frame in zip(tape['states_before'], tape['states_after'], tape['commands'],
                                                          tape['selected_control_sequences'], record['frames'], strict=True):
                if not np.array_equal(before, state.to_array()) or not np.array_equal(cmd, sequence[0]):
                    raise ValueError('状态或首动作不连续')
                if frame['selected'] == 'terminal_rotation':
                    terminal_steps += 1
                    certificate = frame['terminal_certificate']
                    goal = tape['goal']
                    pos = float(np.hypot(state.x-goal[0], state.y-goal[1]))
                    delta = float(np.arctan2(np.sin(goal[2]-state.yaw), np.cos(goal[2]-state.yaw)))
                    if not (pos < .3 and abs(delta) >= np.deg2rad(10)):
                        raise ValueError('终端接管违反原位置/航向资格')
                    if certificate['goal'] != goal.tolist() or certificate['tol_pos'] != .3 or certificate['tol_yaw'] != np.deg2rad(10):
                        raise ValueError('终端目标/容差不同于原生')
                    if not checker.swept_segment_free(before[:3], np.array([state.x, state.y, state.yaw+delta])):
                        raise ValueError('完整转向角程并不安全')
                    expected = np.zeros((10, 2)); current = state
                    for k in range(10):
                        error = float(np.arctan2(np.sin(goal[2]-current.yaw), np.cos(goal[2]-current.yaw)))
                        omega = 0. if abs(error) < 1e-12 else float(np.clip(error/.1, -vehicle.max_omega, vehicle.max_omega))
                        expected[k, 1] = omega
                        predicted = model.step(current, ControlCmd(0., omega), .1)
                        if not checker.swept_segment_free(current.to_array()[:3], predicted.to_array()[:3]):
                            raise ValueError('终端预测时域接触')
                        current = predicted
                    if not np.array_equal(sequence, expected):
                        raise ValueError('独立终端序列与归档不一致')
                    if (certificate['signed_rotation'] != delta or certificate['position_error'] != pos
                            or certificate['full_rotation_contact_safe'] is not True):
                        raise ValueError('终端证书内容不一致')
                state = model.step(state, ControlCmd(*map(float, cmd)), .1)
                if not np.array_equal(state.to_array(), after) or not checker.swept_segment_free(before[:3], after[:3]):
                    raise ValueError('真实转移或物理接触未独立复现')
                safe_margin = margin.swept_segment_free(before[:3], after[:3])
                if safe_margin != frame['planning_margin_safe']:
                    raise ValueError('余量声明不符')
                violations += not safe_margin
            if violations != record['summary']['planning_margin_violations']:
                raise ValueError('回合余量计数不符')
        summaries.append({'dataset_index': index, 'actual_steps': len(record['frames']),
                          'terminal_steps': terminal_steps, 'planning_margin_violations': violations})
    payload = {'status': 'completed', 'source_report_sha256': _sha256(source/'report.json'),
               'code': execution_identity(FILES), 'indices': config['indices'], 'episodes': summaries,
               'all_actual_states_and_sweeps_exact': True, 'all_terminal_sequences_and_certificates_exact': True,
               'elapsed_seconds': time.perf_counter()-started, 'production_qualified': False,
               'limits': '全部短检实际动作与终端模式独立重建；非终端CEM及LT22选择没有再次重算，不代表同300或BEV接入。'}
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output/'report.json', payload)
    return payload
