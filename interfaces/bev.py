"""BEV 环境表示定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BEVConfig:
    """跨场景与 Sensor2BEV 共享的 BEV 空间配置。

    extent 顺序为 ``(front, back, left, right)``，单位米；默认配置生成
    160×160 的 40×40m 栅格。覆盖范围必须能被 resolution 整除，避免
    转换器之间依赖 ``round`` 产生不一致尺寸。
    """

    resolution: float = 0.25
    extent: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 20.0)

    def __post_init__(self) -> None:
        resolution = float(self.resolution)
        try:
            extent = tuple(float(value) for value in self.extent)
        except TypeError as exc:
            raise ValueError("BEV extent 必须包含 front/back/left/right 四个数值") from exc
        if len(extent) != 4:
            raise ValueError("BEV extent 必须包含 front/back/left/right 四个数值")
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("BEV resolution 必须为有限正数")
        if any(not np.isfinite(value) or value <= 0.0 for value in extent):
            raise ValueError("BEV extent 的四个方向必须为有限正数")

        height = (extent[0] + extent[1]) / resolution
        width = (extent[2] + extent[3]) / resolution
        if not (
            np.isclose(height, round(height), rtol=0.0, atol=1e-9)
            and np.isclose(width, round(width), rtol=0.0, atol=1e-9)
        ):
            raise ValueError("BEV extent 总范围必须能被 resolution 整除")

        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "extent", extent)

    @property
    def shape(self) -> tuple[int, int]:
        front, back, left, right = self.extent
        return (
            round((front + back) / self.resolution),
            round((left + right) / self.resolution),
        )


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
        config = BEVConfig(resolution=self.resolution, extent=self.extent)
        self.resolution = config.resolution
        self.extent = config.extent
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 3:
            raise ValueError(f"BEV 数据形状必须为 (C, H, W)，实际 {self.data.ndim} 维")
        c, h, w = self.data.shape
        if len(self.channels) != c:
            raise ValueError(f"通道列表长度 {len(self.channels)} 与张量通道数 {c} 不一致")
        expected_h, expected_w = config.shape
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

    def to_metadata(self) -> dict[str, float | list[float] | list[str] | list[int]]:
        """返回可直接 JSON 序列化的稳定空间与通道元数据。"""
        return {
            "resolution": self.resolution,
            "extent": list(self.extent),
            "channels": list(self.channels),
            "shape": list(self.shape),
        }
