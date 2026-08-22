"""训练数据集生成与加载。

批量生成样本（BEV + 目标位姿 + 状态 → 专家轨迹）。样本坐标约定：
- BEV 为车辆中心局部系；
- goal 与 expert_trajectory 为全局坐标（世界系），网络侧按需转换。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState


@dataclass
class TrainingSample:
    """单条训练样本。"""

    bev: BEVTensor
    goal: GoalPose
    state: VehicleState
    expert_trajectory: Trajectory


class DatasetGenerator:
    """训练样本生成器。

    在给定环境中随机采样可行起始/目标位姿，使用 Hybrid A* 生成专家轨迹，
    并通过传感器 → Sensor2BEV → 融合链路生成 BEV。
    """

    def __init__(
        self,
        env,
        planner,
        sensor_pipeline,
        seed: int = 0,
        min_distance: float = 3.0,
        max_distance: float = 12.0,
    ) -> None:
        self.env = env
        self.planner = planner
        # sensor_pipeline: 提供 capture_bev(x, y, yaw) -> BEVTensor 的适配器。
        self.sensor_pipeline = sensor_pipeline
        self.rng = np.random.default_rng(seed)
        # 泊车场景轨迹通常较短；限制起终点距离保证规划快速收敛。
        self.min_distance = min_distance
        self.max_distance = max_distance

    def generate(self, count: int) -> list[TrainingSample]:
        """生成 count 条样本；跳过规划失败的样本直到凑够数量。"""
        samples: list[TrainingSample] = []
        attempts = 0
        max_attempts = count * 20 + 100
        while len(samples) < count and attempts < max_attempts:
            attempts += 1
            start, goal = self._random_pose_pair()
            if start is None:
                continue
            try:
                trajectory = self.planner.plan(start, goal)
            except (RuntimeError, ValueError):
                continue
            bev = self.sensor_pipeline.capture_bev(start.x, start.y, start.yaw)
            samples.append(
                TrainingSample(bev=bev, goal=goal, state=start, expert_trajectory=trajectory)
            )
        if len(samples) < count:
            raise RuntimeError(
                f"仅生成 {len(samples)}/{count} 条样本，尝试 {attempts} 次后仍未凑齐"
            )
        return samples

    def _random_pose_pair(self) -> tuple[VehicleState | None, GoalPose | None]:
        """随机采样无碰撞的起始/目标位姿，要求两点间距足够。"""
        half = self.env.world_size / 2.0 - 1.0
        for _ in range(50):
            sx = self.rng.uniform(-half, half)
            sy = self.rng.uniform(-half, half)
            syaw = self.rng.uniform(-np.pi, np.pi)
            gx = self.rng.uniform(-half, half)
            gy = self.rng.uniform(-half, half)
            gyaw = self.rng.uniform(-np.pi, np.pi)
            start = VehicleState(float(sx), float(sy), float(syaw))
            goal = GoalPose(float(gx), float(gy), float(gyaw))
            if not (self._pose_free(sx, sy, syaw) and self._pose_free(gx, gy, gyaw)):
                continue
            dist = np.hypot(gx - sx, gy - sy)
            if dist < self.min_distance or dist > self.max_distance:
                continue
            return start, goal
        return None, None

    def _pose_free(self, x: float, y: float, yaw: float) -> bool:
        """判断车辆矩形四角是否全部位于自由空间（与规划器一致）。"""
        half_l = self.planner.vehicle_length / 2.0
        half_w = self.planner.vehicle_width / 2.0
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        corners = [
            (x + half_l * cos_yaw - half_w * sin_yaw, y + half_l * sin_yaw + half_w * cos_yaw),
            (x + half_l * cos_yaw + half_w * sin_yaw, y + half_l * sin_yaw - half_w * cos_yaw),
            (x - half_l * cos_yaw - half_w * sin_yaw, y - half_l * sin_yaw + half_w * cos_yaw),
            (x - half_l * cos_yaw + half_w * sin_yaw, y - half_l * sin_yaw - half_w * cos_yaw),
        ]
        return all(self.env.is_free(cx, cy) for cx, cy in corners)