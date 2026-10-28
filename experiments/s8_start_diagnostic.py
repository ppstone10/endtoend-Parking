"""S8起步诊断的预测与原采样审计；风险不进入MPC代价或动作选择。"""
import numpy as np


def predict_sequences(mpc, state, sequences):
    sequences = np.asarray(sequences, dtype=np.float64)
    if (sequences.ndim != 2 or sequences.shape[1] != 2 * mpc.horizon
            or not np.isfinite(sequences).all()):
        raise ValueError('非法原控制序列')
    count = len(sequences)
    x = np.full(count, state.x, dtype=np.float64)
    y = np.full(count, state.y, dtype=np.float64)
    yaw = np.full(count, state.yaw, dtype=np.float64)
    points = np.empty((count, mpc.horizon, 3))
    for step in range(mpc.horizon):
        v = np.clip(sequences[:, 2 * step], -mpc.max_v, mpc.max_v)
        omega = np.clip(sequences[:, 2 * step + 1], -mpc.max_omega, mpc.max_omega)
        x = x + v * np.cos(yaw) * mpc.dt
        y = y + v * np.sin(yaw) * mpc.dt
        yaw = yaw + omega * mpc.dt
        points[:, step] = np.column_stack((x, y, yaw))
    return points


def swept_flags(start, sequences, checker, check):
    points = np.asarray(sequences)
    if points.ndim != 3 or points.shape[2] != 3 or not np.isfinite(points).all():
        raise ValueError('非法预测位姿')
    flags = np.empty(points.shape[:2], dtype=bool)
    for index, sequence in enumerate(points):
        previous = np.asarray(start)[:3]
        for step, point in enumerate(sequence):
            check()
            flags[index, step] = not checker.swept_segment_free(previous, point)
            previous = point
    return flags


def summarize_iteration(samples, costs, flags, elite_frac):
    samples, costs, flags = np.asarray(samples), np.asarray(costs), np.asarray(flags)
    if (samples.ndim != 2 or costs.shape != (len(samples),) or flags.ndim != 2
            or flags.shape[0] != len(samples) or flags.shape[1] < 1
            or not np.isfinite(samples).all() or not np.isfinite(costs).all()):
        raise ValueError('非法CEM原采样/成本/风险')
    elite = np.argsort(costs)[:max(2, int(round(len(samples) * elite_frac)))]
    first_safe = ~flags[:, 0]
    full_safe = ~flags.any(axis=1)
    def best(mask):
        indices = np.flatnonzero(mask)
        return int(indices[np.argmin(costs[indices])]) if len(indices) else None
    minimum = int(np.argmin(costs))
    first_best, full_best = best(first_safe), best(full_safe)
    return dict(sample_count=len(samples), first_safe_count=int(first_safe.sum()),
        horizon_safe_count=int(full_safe.sum()), elite_count=len(elite),
        elite_first_safe_count=int(first_safe[elite].sum()), elite_horizon_safe_count=int(full_safe[elite].sum()),
        minimum_cost=float(costs[minimum]), minimum_cost_first_safe=bool(first_safe[minimum]),
        minimum_cost_horizon_safe=bool(full_safe[minimum]),
        best_first_safe_index=first_best, best_horizon_safe_index=full_best,
        best_first_safe_cost=None if first_best is None else float(costs[first_best]),
        best_horizon_safe_cost=None if full_best is None else float(costs[full_best]))
