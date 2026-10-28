"""保存值不足以保证栅格重放；还原真实车辆状态的标量类型。"""
from __future__ import annotations

import numpy as np
from interfaces import ControlCmd, VehicleState


def scalar_type(value):
    if type(value) is float:
        return 'float'
    if isinstance(value, np.floating):
        return np.dtype(type(value)).name
    raise ValueError(f'不支持的采集标量类型：{type(value).__name__}')


def restore_pose(values, types):
    if len(types) != 3 or len(values) < 3:
        raise ValueError('采集位姿类型必须有三项')
    restored = []
    for value, name in zip(values[:3], types, strict=True):
        if name == 'float':
            restored.append(float(value))
        elif name in ('float32', 'float64'):
            restored.append(np.dtype(name).type(value))
        else:
            raise ValueError(f'未知标量类型：{name}')
    return restored


def reconstruct_capture_types(initial, states_before, states_after, commands, issue_steps, vehicle_model, dt):
    """只消费归档命令，不运行MPC；状态逐位不符就停止，绝不猜测类型。"""
    if not (len(states_before) == len(states_after) == len(commands)):
        raise ValueError('真实转移和命令分母不一致')
    wanted = set(int(s) for s in issue_steps)
    if len(wanted) != len(issue_steps):
        raise ValueError('预测时刻重复')
    state = VehicleState.from_array(np.asarray(initial))
    types = {}
    for step, (before, after, command) in enumerate(zip(states_before, states_after, commands, strict=True)):
        if not np.array_equal(state.to_array(), before):
            raise ValueError(f'#{step}前态不精确，不能重建采集类型')
        if step in wanted:
            types[step] = [scalar_type(x) for x in (state.x, state.y, state.yaw)]
        state = vehicle_model.step(state, ControlCmd(*map(float, command)), dt)
        if not np.array_equal(state.to_array(), after):
            raise ValueError(f'#{step}推进不精确，不能重建采集类型')
    if set(types) != wanted:
        raise ValueError('存在未重建的预测时刻')
    return types
