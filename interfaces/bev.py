"""BEV 环境表示定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BEVTensor:
    """以车辆为中心的栅格化 BEV 表示。

    data 形状为 (C, H, W)。通道语义由 channels 列表说明，约定顺序：
    障碍物占据、高度、点云密度、目标区域、车辆轮廓。
    resolution 为每栅格的米数，extent 为前后左右覆盖范围 (front, back, left, right) 米。
    """

    data: np.ndarray
    resolution: float
    extent: tuple[float, float, float, float]
    channels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 3:
            raise ValueError(f"BEV 数据形状必须为 (C, H, W)，实际 {self.data.ndim} 维")
        c, h, w = self.data.shape
        if len(self.channels) != c:
            raise ValueError(f"通道列表长度 {len(self.channels)} 与张量通道数 {c} 不一致")
        front, back, left, right = self.extent
        expected_h = round((front + back) / self.resolution)
        expected_w = round((left + right) / self.resolution)
        if h != expected_h or w != expected_w:
            raise ValueError(
                f"BEV 尺寸 ({h}, {w}) 与 extent/resolution 推导尺寸 "
                f"({expected_h}, {expected_w}) 不一致"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]