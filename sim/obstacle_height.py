"""障碍物语义高度：`height` 通道的取值来源。

模拟 LiDAR 是固定安装高度（默认 1.0m）的二维扫描，命中点 z 恒为安装高度，因此
"height 通道"原本与 occupancy 完全重合、不携带任何信息（实测唯一值 {0,1}）。
本模块给每类障碍定义**语义高度**（代表该障碍在现实中需要多高才能挡住光束），
使 height 通道真正携带"障碍是高墙还是矮挡墙/车辆/岩石"的信息：

- 通道取值 = 该栅格内最高障碍的语义高度 / ``MAX_SEMANTIC_HEIGHT_M``，落在 (0, 1]；
- 同一 ``kind`` 的高度处处一致，因此这是**确定性的语义标签**，不是伪造的测距值；
- 未命中（未占用的栅格）保持 0。

这样既不动 ``ParkingEnvironment.raycast`` 的二维平面语义（不改碰撞与轨迹），
又让网络获得"障碍类型"这一真实有用特征。
"""

from __future__ import annotations

from .obstacles import (
    KIND_BERM,
    KIND_CLIFF,
    KIND_EQUIPMENT,
    KIND_LINE,
    KIND_ROCK,
    KIND_VEHICLE,
    KIND_WALL,
)

__all__ = [
    "DEFAULT_KIND_HEIGHT_M",
    "KIND_HEIGHT_M",
    "MAX_SEMANTIC_HEIGHT_M",
    "normalized_height",
    "semantic_height",
]

#: 各障碍类别的语义高度（米）：代表"现实中该障碍的高度量级"。
KIND_HEIGHT_M: dict[str, float] = {
    KIND_WALL: 3.0,
    KIND_VEHICLE: 2.5,
    KIND_EQUIPMENT: 4.0,
    KIND_BERM: 1.2,
    KIND_ROCK: 0.8,
    KIND_CLIFF: 0.0,   # 悬崖不挡射线、不产生点云
    KIND_LINE: 0.0,    # 地面标线可通行、不产生点云
}

DEFAULT_KIND_HEIGHT_M = 2.0

#: 归一化分母：通道取值 = 语义高度 / 该值。
MAX_SEMANTIC_HEIGHT_M = 4.0


def semantic_height(kind: str | None) -> float:
    """按障碍类别返回语义高度（米）；未知类别取默认值。"""
    if kind is None:
        return DEFAULT_KIND_HEIGHT_M
    return KIND_HEIGHT_M.get(str(kind), DEFAULT_KIND_HEIGHT_M)


def normalized_height(kind: str | None) -> float:
    """按障碍类别返回归一化高度，落在 (0, 1]（用于 height 通道）。"""
    value = semantic_height(kind) / MAX_SEMANTIC_HEIGHT_M
    return float(min(max(value, 0.0), 1.0))
