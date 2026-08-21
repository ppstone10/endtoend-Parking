"""MineParkingNet 端到端轨迹生成网络。

阶段四实现：输入 BEV + 目标泊车位姿 + 车辆运动状态，输出未来 N 个局部轨迹点。
本阶段只保留接口契约，不实现网络结构。
"""

from __future__ import annotations

from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState


class MineParkingNet:
    """端到端轨迹生成网络骨架。

    输入：BEVTensor、GoalPose、VehicleState；输出：Trajectory（未来 N 个局部轨迹点）。
    """

    def __init__(self, horizon: int = 20, dt: float = 0.1) -> None:
        self.horizon = horizon
        self.dt = dt

    def predict(
        self, bev: BEVTensor, goal: GoalPose, state: VehicleState
    ) -> Trajectory:
        """预测未来轨迹（骨架阶段返回占位轨迹）。

        正式实现将在阶段四使用训练好的权重替换。
        """
        raise NotImplementedError("MineParkingNet.predict 将在阶段四实现")