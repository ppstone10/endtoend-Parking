"""从落盘轨迹独立重建动作、参考和选中控制时域；不运行求解器。"""
import numpy as np
from interfaces import ControlCmd, VehicleState
from planner import RectangleFootprintCollisionChecker
from sim import DifferentialDriveModel
from experiments import validation_matrix as matrix


def audit_tape(path, vehicle, env, budget):
    """从落盘数组重建实际推进、完整H10和全部参考；不重算CEM种群排名。"""
    checkers = {name: RectangleFootprintCollisionChecker(env, vehicle_length=vehicle.length,
        vehicle_width=vehicle.width, collision_margin=margin, resolution=vehicle.collision_check_resolution)
        for name, margin in (('contact', 0.), ('planning', vehicle.collision_margin))}
    model = DifferentialDriveModel(**vehicle.vehicle_model_kwargs())
    contact, margin, horizons, references = 0, 0, 0, 0
    with np.load(path, allow_pickle=False) as tape:
        previous = tape['initial_state']
        count = len(tape['commands'])
        for before, after, command, sequence in zip(tape['states_before'], tape['states_after'],
                tape['commands'], tape['sequences'], strict=True):
            budget.check()
            state = VehicleState.from_array(before)
            actual = model.step(state, ControlCmd(*map(float, command)), .1)
            if not np.array_equal(previous, before) or not np.array_equal(actual.to_array(), after):
                raise ValueError('实际状态/动作/连续性不精确')
            contact += not checkers['contact'].swept_segment_free(before[:3], after[:3])
            margin += not checkers['planning'].swept_segment_free(before[:3], after[:3])
            safe = True
            if sequence.shape != (10, 2) or not np.isfinite(sequence).all():
                raise ValueError('选中时域结构无效')
            if not np.array_equal(command, sequence[0]):
                raise ValueError('首动作未执行选中序列')
            for v, omega in sequence:
                if abs(v) > vehicle.max_v or abs(omega) > vehicle.max_omega:
                    raise ValueError('选中序列超出车辆上限')
                after_h = model.step(state, ControlCmd(float(v), float(omega)), .1)
                safe &= checkers['planning'].swept_segment_free(state.to_array()[:3], after_h.to_array()[:3])
                state = after_h
            horizons += not safe
            previous = after
        queries = int(tape['references'])
        for ordinal in range(queries):
            budget.check()
            trajectory = matrix.Trajectory(tape[f'reference_{ordinal:04d}'], dt=float(tape[f'reference_dt_{ordinal:04d}']))
            state = VehicleState.from_array(tape[f'query_state_{ordinal:04d}'])
            references += not checkers['planning'].check_trajectory(state, trajectory)[0]
        terminal_steps = tape['terminal_steps']
        if len(terminal_steps) != len(set(terminal_steps.tolist())):
            raise ValueError('终端证书重复')
        for index in terminal_steps:
            if not 0 <= index < count:
                raise ValueError('终端证书步数越界')
            state = VehicleState.from_array(tape['states_before'][index])
            goal = tape['goal']
            position = np.linalg.norm(state.to_array()[:2]-goal[:2])
            angle = float(np.arctan2(np.sin(goal[2]-state.yaw), np.cos(goal[2]-state.yaw)))
            if not (position < float(tape['tol_pos']) and abs(angle) >= float(tape['tol_yaw'])):
                raise ValueError('终端接管不符合原车位阈值')
            if not checkers['planning'].swept_segment_free(state.to_array()[:3], np.array([state.x, state.y, state.yaw+angle])):
                raise ValueError('完整转角缺少0.2m余量')
            expected = np.zeros((10, 2))
            current = state
            for step in range(10):
                delta = float(np.arctan2(np.sin(goal[2]-current.yaw), np.cos(goal[2]-current.yaw)))
                omega = float(np.clip(delta/.1, -vehicle.max_omega, vehicle.max_omega)) if abs(delta) >= 1e-12 else 0.
                expected[step, 1] = omega
                current = model.step(current, ControlCmd(0., omega), .1)
            if not np.array_equal(expected, tape['sequences'][index]):
                raise ValueError('终端零平移控制与证书不一致')
    return {'controls': count, 'references': queries, 'actual_contact_steps': int(contact),
            'planning_margin_violations': int(margin), 'unsafe_selected_horizons': int(horizons),
            'unsafe_emitted_references': int(references), 'terminal_certificates_verified': len(terminal_steps),
            'transitions_exact': True}

