"""GT BEV 管道：由场景几何真值直接生成与生产同构的五通道 BEV。

用途：把"感知质量"从闭环验证中隔离出来。生产链路是
``LiDAR 点云 → LiDAR2BEV → BEVFusion ← Camera2BEV ← 相机图像``；
本模块用场景几何真值替换两个传感器源：

- ``occupancy``/``height``/``density``：场景中 ``emits_points`` 障碍的几何栅格
  化（与 ``metrics.bev_fidelity.rasterize_ground_truth_occupancy`` 同一真值口径，
  只计阻挡射线的障碍：悬崖禁入但不产生点云、地面标线可通行）；
- ``target``：目标车位矩形真值栅格化（与
  ``metrics.bev_fidelity.rasterize_ground_truth_target`` 同口径）；
- ``vehicle``：自车外廓栅格化（与 ``BEVFusion._vehicle_outline`` 同语义，但按
  栅格中心判定，不依赖栅格对齐取整）。

输出通道顺序、形状、分辨率与生产 BEV 完全一致，因此可直接替换
``SensorBEVPipeline`` 供 ``NetworkSource`` 使用，无需改动网络或闭环引擎。

注意：GT BEV 的 occupancy 是稠密几何值（车辆足迹范围内全 1），与生产稀疏
点云投影（实际占据约 1/5 栅格）分布不同。E2b/E3b 用它度量"感知完美时网络
能走多好"，其绝对数字隐含该分布差异，解读时必须与传感器 BEV 口径并列报告。
"""

from __future__ import annotations

import numpy as np

from interfaces import BEVConfig, BEVTensor, GoalPose
from metrics.bev_fidelity import (
    rasterize_ground_truth_occupancy,
    rasterize_ground_truth_target,
)

__all__ = [
    "GroundTruthBEVPipeline",
    "ground_truth_vehicle_outline",
]

# 几何真值没有逐障碍高度场：模拟 LiDAR 的安装高度为 1.0m，故占用栅格取该值，
# 与生产 height 通道（点云 z 最大值）在障碍高度量级上保持一致。
GT_OCCUPANCY_HEIGHT_M = 1.0
GT_DENSITY_VALUE = 1.0


def ground_truth_vehicle_outline(
    bev_config: BEVConfig, vehicle_length: float, vehicle_width: float
) -> np.ndarray:
    """按栅格中心生成自车外廓通道 (1,H,W)。"""
    h, w = bev_config.shape
    front, back, left, right = bev_config.extent
    resolution = bev_config.resolution
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    local_x = front - (rows + 0.5) * resolution
    local_y = -right + (cols + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    inside = (np.abs(grid_x) <= vehicle_length / 2.0) & (
        np.abs(grid_y) <= vehicle_width / 2.0
    )
    return inside.astype(np.float32)[None, :, :]


class GroundTruthBEVPipeline:
    """由场景几何真值生成五通道 BEV 的管道。

    env 为 ``sim.ParkingEnvironment``；bev_config 为统一 BEV 配置；
    vehicle_length/vehicle_width 为自车外廓尺寸。
    ``set_target_goals`` 与 ``SensorBEVPipeline`` 同名同义，供 ``NetworkSource``
    在回合开始时注入当前目标。
    """

    def __init__(
        self,
        env,
        bev_config: BEVConfig,
        *,
        vehicle_length: float,
        vehicle_width: float,
    ) -> None:
        if vehicle_length <= 0.0 or vehicle_width <= 0.0:
            raise ValueError("车辆外廓尺寸必须为正")
        self.env = env
        self.bev_config = bev_config
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.channels = ["occupancy", "height", "density", "target", "vehicle"]
        self._goals: list[GoalPose] = []
        self._vehicle_channel = ground_truth_vehicle_outline(
            bev_config, self.vehicle_length, self.vehicle_width
        )

    def set_target_goals(self, goals: list[GoalPose]) -> None:
        """设置当前目标（渲染到 target 通道）。"""
        self._goals = list(goals)

    def capture_bev(self, x: float, y: float, yaw: float) -> BEVTensor:
        """在指定位姿生成一帧 GT BEV。"""
        config = self.bev_config
        occupancy = rasterize_ground_truth_occupancy(self.env, x, y, yaw, config)
        height = np.where(occupancy > 0.0, GT_OCCUPANCY_HEIGHT_M, 0.0).astype(np.float32)
        density = np.where(occupancy > 0.0, GT_DENSITY_VALUE, 0.0).astype(np.float32)
        target = np.zeros_like(occupancy)
        for goal in self._goals:
            rasterized = rasterize_ground_truth_target(
                goal,
                self.vehicle_length,
                self.vehicle_width,
                x,
                y,
                yaw,
                config,
            )
            target = np.maximum(target, rasterized)
        data = np.stack(
            [
                occupancy.astype(np.float32),
                height,
                density,
                target.astype(np.float32),
                self._vehicle_channel[0].astype(np.float32),
            ],
            axis=0,
        )
        return BEVTensor(
            data=data,
            resolution=config.resolution,
            extent=config.extent,
            channels=self.channels,
        )
