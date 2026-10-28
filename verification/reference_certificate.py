"""在私有控制上下文中证明原网络参考可在原预算内严格到达。"""
from copy import deepcopy
import numpy as np


def certify_reference(controller, reference, state, contact_checker, margin_checker, max_steps, budget_check=None):
    if max_steps <= 0 or controller.terminal.reached(state, controller.goal):
        return None
    if not contact_checker.pose_free(state.x, state.y, state.yaw) or not margin_checker.pose_free(state.x, state.y, state.yaw):
        return None
    shadow, trajectory = deepcopy((controller, reference))
    # 录制器将compute观察闭包装到实例上；副本必须重新绑定类方法，避免写真实账本。
    shadow.__dict__.pop('compute', None)
    context = {'last_u': controller._last_u.tolist(), 'rng_state': deepcopy(controller._rng.bit_generator.state),
               'reference_is_progress': getattr(controller, '_progress_traj', None) is reference,
               'progress_idx': getattr(controller, '_progress_idx', None),
               'aligned_cache': getattr(controller, '_aligned_cache', None)} if hasattr(controller, '_last_u') else None
    if context is not None and context['aligned_cache'] is not None:
        context['aligned_cache'] = context['aligned_cache'].tolist()
    current = deepcopy(state)
    before, after, controls = [], [], []
    for _ in range(int(max_steps)):
        if budget_check is not None:
            budget_check()
        cmd = shadow.compute(trajectory, current)
        proposed = shadow.vehicle_model.step(current, cmd, shadow.dt)
        if (not contact_checker.swept_segment_free(current.to_array()[:3], proposed.to_array()[:3])
                or not margin_checker.swept_segment_free(current.to_array()[:3], proposed.to_array()[:3])):
            return None
        before.append(current.to_array().tolist()); after.append(proposed.to_array().tolist())
        controls.append([cmd.v, cmd.omega]); current = proposed
        if shadow.terminal.reached(current, shadow.goal):
            return {'states_before': before, 'states_after': after, 'controls': controls,
                    'goal': [shadow.goal.x, shadow.goal.y, shadow.goal.yaw],
                    'tol_pos': shadow.terminal.tol_pos, 'tol_yaw': shadow.terminal.tol_yaw,
                    'reference': np.asarray(reference.points).tolist(), 'reference_dt': reference.dt,
                    'reference_dtype': np.asarray(reference.points).dtype.str,
                    'context': context, 'control_dt': shadow.dt, 'contact_safe': True, 'planning_margin_safe': True}
    return None
