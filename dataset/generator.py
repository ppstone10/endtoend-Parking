"""训练数据集生成与加载。

阶段三/四实现：批量生成样本（BEV + 目标位姿 + 状态 → 专家轨迹）。
本阶段只保留接口契约。
"""

from __future__ import annotations

from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState
from dataclasses import dataclass


@dataclass
class TrainingSample:
    """单条训练样本。"""

    bev: BEVTensor
    goal: GoalPose
    state: VehicleState
    expert_trajectory: Trajectory


class DatasetGenerator:
    """训练样本生成器骨架。"""

    def generate(self, count: int) -> list[TrainingSample]:
        raise NotImplementedError("DatasetGenerator.generate 将在阶段三实现")