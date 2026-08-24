"""泊车位抽象：位姿 + 尺寸 + 容差 + 占用状态。

ParkingSpot 是任务层与场景库的核心单元：容差（tol_pos/tol_yaw）即闭环
到达判定阈值（REQUIREMENTS §3 各场景精度要求），占用状态由场景构造时
决定（相邻停放车辆为 kind=vehicle 障碍）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from interfaces import GoalPose


@dataclass
class ParkingSpot:
    """单个泊车位。

    pose 为车位中心目标位姿（车停稳后的位姿）；size 为车位框尺寸
    (length, width)（米）；tol_pos/tol_yaw 为到达判定容差；kind 标识
    车位类型（spot/berm_bay/crusher_slot/loading_point/fuel_bay/weigh_pad）；
    occupied 表示该位是否被占用（占用位不参与候选，仅作为障碍渲染）。
    """

    id: str
    pose: GoalPose
    size: tuple[float, float] = (7.0, 3.5)
    tol_pos: float = 0.3
    tol_yaw: float = float(np.deg2rad(10.0))
    kind: str = "spot"
    occupied: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def goal(self) -> GoalPose:
        """车位目标位姿（未占用时）。"""
        return self.pose

    def footprint_corners(self) -> np.ndarray:
        """车位框四角世界坐标 (4,2)，用于渲染与重叠检查。"""
        l, w = self.size
        cos_yaw, sin_yaw = np.cos(self.pose.yaw), np.sin(self.pose.yaw)
        local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        return local @ rot.T + np.array([self.pose.x, self.pose.y])

    def occupant_obstacle(self, vehicle_length: float, vehicle_width: float):
        """占用车辆对应的矩形障碍（车身尺寸，非车位框尺寸）。"""
        from sim.obstacles import RectangleObstacle

        cos_yaw, sin_yaw = np.cos(self.pose.yaw), np.sin(self.pose.yaw)
        cx = self.pose.x
        cy = self.pose.y
        ex = vehicle_length / 2.0 * cos_yaw
        ey = vehicle_length / 2.0 * sin_yaw
        px = vehicle_width / 2.0 * sin_yaw
        py = vehicle_width / 2.0 * cos_yaw
        return RectangleObstacle(
            x_min=cx - ex - px, x_max=cx + ex + px,
            y_min=cy - ey - py, y_max=cy + ey - py,
            kind="vehicle",
        )


def make_spot_row(
    base_x: float,
    base_y: float,
    yaw: float,
    count: int,
    pitch: float,
    spot_id_prefix: str,
    size: tuple[float, float] = (7.0, 3.5),
    tol_pos: float = 0.3,
    tol_yaw: float = float(np.deg2rad(10.0)),
    kind: str = "spot",
    occupied: list[bool] | None = None,
) -> list[ParkingSpot]:
    """沿 pitch 间隔生成一排同类车位。

    yaw 为车位朝向（车停稳后车头方向）；base 为第一个车位中心；
    occupied 未给定时全部空闲。
    """
    spots: list[ParkingSpot] = []
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    # 排方向：垂直于车位朝向。
    dx = -sin_yaw * pitch
    dy = cos_yaw * pitch
    for i in range(count):
        occ = bool(occupied[i]) if occupied is not None and i < len(occupied) else False
        spots.append(
            ParkingSpot(
                id=f"{spot_id_prefix}-{i}",
                pose=GoalPose(base_x + i * dx, base_y + i * dy, yaw),
                size=size,
                tol_pos=tol_pos,
                tol_yaw=tol_yaw,
                kind=kind,
                occupied=occ,
            )
        )
    return spots
