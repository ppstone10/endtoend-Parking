"""BEV 环境表示定义。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CAMERA_PROJECTION_LEGACY = "legacy_overlay_v1"
CAMERA_PROJECTION_CURRENT = "per_view_v2"
CAMERA_PROJECTION_GEOMETRY = "geometry_v1"


def assert_camera_projection_matches(actual: str | None, expected: str, *, target_channel="image") -> None:
    """显式GT源可用于诊断；image输入必须匹配训练投影版本。"""
    if expected not in (CAMERA_PROJECTION_LEGACY, CAMERA_PROJECTION_CURRENT):
        raise ValueError(f"未知模型camera_projection：{expected}")
    if actual not in (None, CAMERA_PROJECTION_LEGACY, CAMERA_PROJECTION_CURRENT, CAMERA_PROJECTION_GEOMETRY):
        raise ValueError(f"未知输入camera_projection：{actual}")
    if target_channel == "geometry":
        if actual == CAMERA_PROJECTION_CURRENT:
            raise ValueError("target_channel不匹配：输入为修复后的image，模型声明geometry")
        return
    if actual == CAMERA_PROJECTION_GEOMETRY:
        return
    actual = actual or CAMERA_PROJECTION_LEGACY
    if actual != expected:
        raise ValueError(
            f"camera_projection不匹配：输入{actual}，模型{expected}；"
            "请使用对应版本数据/权重，修复后的输入需要重新训练，不能只改checkpoint声明"
        )


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
    camera_projection记录target生成版本；None保留历史元数据，不能冒充per_view_v2。
    """

    data: np.ndarray
    resolution: float
    extent: tuple[float, float, float, float]
    channels: list[str] = field(default_factory=list)
    camera_projection: str | None = None

    def __post_init__(self) -> None:
        config = BEVConfig(resolution=self.resolution, extent=self.extent)
        self.resolution = config.resolution
        self.extent = config.extent
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.camera_projection not in (None, CAMERA_PROJECTION_LEGACY,
                                          CAMERA_PROJECTION_CURRENT, CAMERA_PROJECTION_GEOMETRY):
            raise ValueError(f"未知camera_projection：{self.camera_projection}")
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

    def to_metadata(self) -> dict:
        """返回可直接 JSON 序列化的稳定空间与通道元数据。"""
        metadata = {
            "resolution": self.resolution,
            "extent": list(self.extent),
            "channels": list(self.channels),
            "shape": list(self.shape),
        }
        if self.camera_projection is not None:
            metadata["camera_projection"] = self.camera_projection
        return metadata
