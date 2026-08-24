"""障碍物体系：形状 + 语义。

形状：轴对齐矩形（RectangleObstacle）、简单多边形（PolygonObstacle）、
圆（CircleObstacle）。语义由三个正交属性表达：

- kind：功能语义标签（wall/berm/cliff/rock/vehicle/equipment/line），
  供场景构造、统计与渲染样式使用；
- emits_points：是否阻挡 LiDAR 射线（产生点云）。悬崖（cliff）为 False
  ——射线越过崖边不返回点，实现"看得见的挡墙 + 看不见的崖"；
- forbidden：车辆进入是否判为碰撞。地面标线（line）为 False——可通行。

碰撞检测（is_free）只检查 forbidden 障碍；raycast 只与 emits_points
障碍求交；两者相互独立。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 语义 kind 常量
KIND_WALL = "wall"
KIND_BERM = "berm"        # 卸载挡墙：矮墙，挡射线
KIND_CLIFF = "cliff"      # 悬崖禁区：不挡射线、禁止进入
KIND_ROCK = "rock"        # 散布岩石
KIND_VEHICLE = "vehicle"  # 相邻停放车辆
KIND_EQUIPMENT = "equipment"  # 电铲等设备
KIND_LINE = "line"        # 地面标线：不挡射线、可通行

_EPS = 1e-9


class Obstacle(ABC):
    """障碍物抽象基类。

    子类为 frozen dataclass；kind/emits_points/forbidden 为带默认值的
    实例字段，见各子类定义。
    """

    @abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """点是否在障碍物内（边界含入）。"""

    @abstractmethod
    def ray_entry_distance(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """单位方向射线 (dx,dy) 从 (ox,oy) 出发首次进入障碍物的距离。

        无交或障碍在射线后方返回 None；起点在障碍物内部返回 0.0。
        """

    @property
    @abstractmethod
    def bbox(self) -> tuple[float, float, float, float]:
        """包围盒 (x_min, x_max, y_min, y_max)。"""


@dataclass(frozen=True)
class RectangleObstacle(Obstacle):
    """轴对齐矩形障碍物，坐标为全局世界坐标（米）。

    保留原四字段签名（kind 等语义字段带默认值），既有构造代码零改动。
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    kind: str = KIND_WALL
    emits_points: bool = True
    forbidden: bool = True

    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def ray_entry_distance(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """slab 法射线-AABB 求交。"""
        t_min = -math.inf
        t_max = math.inf
        for o, d, lo, hi in ((ox, dx, self.x_min, self.x_max), (oy, dy, self.y_min, self.y_max)):
            if abs(d) < _EPS:
                if o < lo or o > hi:
                    return None
                continue
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        if t_min > t_max or t_max < 0.0:
            return None  # 无交或交区间整体在射线反方向
        return max(t_min, 0.0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)


@dataclass(frozen=True)
class PolygonObstacle(Obstacle):
    """简单多边形障碍物（凸或凹，顶点按顺序给出）。

    vertices 为 (N,2) 顶点序列（世界坐标）；内部归一化为 tuple 存储，
    保证 frozen dataclass 的相等与哈希语义可用。
    """

    vertices: tuple[tuple[float, float], ...]
    kind: str = KIND_WALL
    emits_points: bool = True
    forbidden: bool = True

    def __post_init__(self) -> None:
        verts = tuple((float(p[0]), float(p[1])) for p in self.vertices)
        if len(verts) < 3:
            raise ValueError("多边形至少需要 3 个顶点")
        object.__setattr__(self, "vertices", verts)

    def contains_point(self, x: float, y: float) -> bool:
        """交点数奇偶判定（边界含入）。"""
        n = len(self.vertices)
        inside = False
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            if self._point_on_segment(x, y, x1, y1, x2, y2):
                return True
            if (y1 > y) != (y2 > y):
                # 求该边与水平线 y 的交点 x，判断交点是否在点左侧。
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x_cross > x:
                    inside = not inside
        return inside

    def ray_entry_distance(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """射线与各边线段求交，取最小非负参数。"""
        best: float | None = None
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            t = self._ray_segment_param(ox, oy, dx, dy, x1, y1, x2, y2)
            if t is not None and (best is None or t < best):
                best = t
        return best

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        return (min(xs), max(xs), min(ys), max(ys))

    @staticmethod
    def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
        """点是否在线段上（含端点，容差内）。"""
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) > 1e-9:
            return False
        return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9

    @staticmethod
    def _ray_segment_param(
        ox: float, oy: float, dx: float, dy: float, x1: float, y1: float, x2: float, y2: float
    ) -> float | None:
        """射线 p+t·r 与线段 q+u·s 的交点参数 t（u∈[0,1]）；无交返回 None。"""
        sx, sy = x2 - x1, y2 - y1
        denom = dx * sy - dy * sx
        if abs(denom) < _EPS:
            return None  # 平行（共线重叠为测度零情形，忽略）
        qx, qy = x1 - ox, y1 - oy
        t = (qx * sy - qy * sx) / denom
        u = (qx * dy - qy * dx) / denom
        if t < -_EPS or u < -_EPS or u > 1.0 + _EPS:
            return None
        return max(t, 0.0)


@dataclass(frozen=True)
class CircleObstacle(Obstacle):
    """圆形障碍物（散布岩石等）。"""

    x: float
    y: float
    radius: float
    kind: str = KIND_ROCK
    emits_points: bool = True
    forbidden: bool = True

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("圆半径必须为正")

    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius**2

    def ray_entry_distance(self, ox: float, oy: float, dx: float, dy: float) -> float | None:
        """射线-圆求交（单位方向）。"""
        fx, fy = ox - self.x, oy - self.y
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - self.radius**2
        if c <= 0.0:
            return 0.0  # 起点在圆内
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        t = (-b - math.sqrt(disc)) / 2.0
        if t < 0.0:
            t = (-b + math.sqrt(disc)) / 2.0
            if t < 0.0:
                return None
        return t

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.radius,
            self.x + self.radius,
            self.y - self.radius,
            self.y + self.radius,
        )
