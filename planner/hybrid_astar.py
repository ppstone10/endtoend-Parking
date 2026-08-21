"""Hybrid A* 等专家轨迹生成器。

阶段三实现：从起点到目标泊车位姿生成专家轨迹，作为训练数据标签。
本阶段只保留接口契约。
"""

from __future__ import annotations

from interfaces import GoalPose, Trajectory, VehicleState


class HybridAStarPlanner:
    """Hybrid A* 专家轨迹生成器骨架。"""

    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt

    def plan(self, start: VehicleState, goal: GoalPose) -> Trajectory:
        """从起始状态规划到目标位姿的轨迹（骨架阶段未实现）。"""
        raise NotImplementedError("HybridAStarPlanner.plan 将在阶段三实现")