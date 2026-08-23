"""闭环回合记录，供指标计算与可视化回放。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from interfaces import ControlCmd, Trajectory, VehicleState


@dataclass
class EpisodeRecord:
    """逐步记录状态、指令与当前参考轨迹（快照化，便于回放）。"""

    states: list[VehicleState] = field(default_factory=list)
    cmds: list[ControlCmd] = field(default_factory=list)
    refs: list[np.ndarray] = field(default_factory=list)  # 当前参考轨迹点快照 (N,3)
    plans: list[np.ndarray] = field(default_factory=list)  # 轨迹源输出快照 (M,3)
    collisions: list[bool] = field(default_factory=list)

    def log(
        self,
        state: VehicleState,
        cmd: ControlCmd,
        ref: Trajectory,
        plan: Trajectory,
        collision: bool,
    ) -> None:
        self.states.append(state)
        self.cmds.append(cmd)
        self.refs.append(np.array(ref.points, copy=True))
        self.plans.append(np.array(plan.points, copy=True))
        self.collisions.append(collision)

    @property
    def n_steps(self) -> int:
        return len(self.states)

    def path_length(self) -> float:
        if len(self.states) < 2:
            return 0.0
        xs = np.array([s.x for s in self.states])
        ys = np.array([s.y for s in self.states])
        return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))

    def tracking_rms(self) -> float:
        """每步车辆位置到当前参考轨迹最近点的横向偏差 RMS。"""
        if not self.states:
            return 0.0
        errs = []
        for state, ref in zip(self.states, self.refs):
            if ref.shape[0] == 0:
                continue
            d = np.hypot(ref[:, 0] - state.x, ref[:, 1] - state.y)
            errs.append(float(d.min()))
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0
