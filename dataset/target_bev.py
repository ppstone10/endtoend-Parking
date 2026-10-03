"""目标车位几何栅格化的共享实现。

target 通道只需要表达"目标车位矩形在自车局部 BEV 中的位置"，因此可以直接由
目标位姿栅格化，而不必经过"相机渲染 → 单应反投影 → 逐格采样"这条会漏采的链路。
本模块把该栅格化集中一处，供 `SensorBEVPipeline` 的 target 替换与
`GroundTruthBEVPipeline` 共用，避免两套口径。
"""

from __future__ import annotations

import numpy as np

from interfaces import BEVConfig, GoalPose

__all__ = ["rasterize_goal_target"]


def rasterize_goal_target(
    goal: GoalPose,
    length: float,
    width: float,
    x: float,
    y: float,
    yaw: float,
    bev_config: BEVConfig,
) -> np.ndarray:
    """把目标车位矩形栅格化为自车中心局部 BEV 的 target 通道。

    车位为目标位姿处 length×width 的定向矩形（与 `SimulatedCamera` 的
    ``parking_area`` 语义一致）；栅格按"格心是否落在矩形内"判定，
    0.25m 分辨率下 6×3m 车位约占 24×12 格。
    """
    config = BEVConfig(resolution=bev_config.resolution, extent=bev_config.extent)
    h, w = config.shape
    front, back, left, right = config.extent
    res = config.resolution
    truth = np.zeros((h, w), dtype=np.float32)

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    local_x = front - (rows + 0.5) * res
    local_y = -right + (cols + 0.5) * res
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    global_x = x + cos_yaw * grid_x - sin_yaw * grid_y
    global_y = y + sin_yaw * grid_x + cos_yaw * grid_y

    cg, sg = np.cos(goal.yaw), np.sin(goal.yaw)
    dx = global_x - goal.x
    dy = global_y - goal.y
    goal_x = cg * dx + sg * dy
    goal_y = -sg * dx + cg * dy
    inside = (np.abs(goal_x) <= length / 2.0) & (np.abs(goal_y) <= width / 2.0)
    truth[inside] = 1.0
    return truth
