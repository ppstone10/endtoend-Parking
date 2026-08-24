"""可复现的传感器噪声配置。

内置 clean/low/high 三档用于实验分层；数值是可控仿真强度，不代表具体
硬件标定结果。调用方可传入自定义 NoiseProfile 扩展鲁棒性实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


def _validate_std(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} 不能为负")


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]")


class NoiseLevel(str, Enum):
    CLEAN = "clean"
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class LiDARNoiseConfig:
    """LiDAR 噪声参数，距离单位均为米。"""

    range_std: float = 0.0
    dropout_rate: float = 0.0
    range_jitter_std: float = 0.0

    def __post_init__(self) -> None:
        _validate_std("range_std", self.range_std)
        _validate_probability("dropout_rate", self.dropout_rate)
        _validate_std("range_jitter_std", self.range_jitter_std)

    @property
    def is_clean(self) -> bool:
        return self.range_std == self.dropout_rate == self.range_jitter_std == 0.0

    def to_metadata(self) -> dict[str, float]:
        return {
            "range_std_m": float(self.range_std),
            "dropout_rate": float(self.dropout_rate),
            "range_jitter_std_m": float(self.range_jitter_std),
        }


@dataclass(frozen=True)
class CameraNoiseConfig:
    """灰度相机像素与目标检测噪声参数。"""

    pixel_std: float = 0.0
    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0

    def __post_init__(self) -> None:
        _validate_std("pixel_std", self.pixel_std)
        _validate_probability("false_negative_rate", self.false_negative_rate)
        _validate_probability("false_positive_rate", self.false_positive_rate)

    @property
    def is_clean(self) -> bool:
        return (
            self.pixel_std == self.false_negative_rate == self.false_positive_rate == 0.0
        )

    def to_metadata(self) -> dict[str, float]:
        return {
            "pixel_std": float(self.pixel_std),
            "false_negative_rate": float(self.false_negative_rate),
            "false_positive_rate": float(self.false_positive_rate),
        }


@dataclass(frozen=True)
class NoiseProfile:
    """一组可应用于 LiDAR 与 Camera 的噪声配置。"""

    level: NoiseLevel
    lidar: LiDARNoiseConfig
    camera: CameraNoiseConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", NoiseLevel(self.level))

    @property
    def is_clean(self) -> bool:
        return self.lidar.is_clean and self.camera.is_clean

    def to_metadata(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "lidar": self.lidar.to_metadata(),
            "camera": self.camera.to_metadata(),
        }


LIDAR_NOISE_CLEAN = LiDARNoiseConfig()
CAMERA_NOISE_CLEAN = CameraNoiseConfig()

NOISE_PROFILES: dict[NoiseLevel, NoiseProfile] = {
    NoiseLevel.CLEAN: NoiseProfile(
        NoiseLevel.CLEAN,
        lidar=LIDAR_NOISE_CLEAN,
        camera=CAMERA_NOISE_CLEAN,
    ),
    NoiseLevel.LOW: NoiseProfile(
        NoiseLevel.LOW,
        lidar=LiDARNoiseConfig(
            range_std=0.02,
            dropout_rate=0.02,
            range_jitter_std=0.05,
        ),
        camera=CameraNoiseConfig(
            pixel_std=2.0,
            false_negative_rate=0.02,
            false_positive_rate=0.01,
        ),
    ),
    NoiseLevel.HIGH: NoiseProfile(
        NoiseLevel.HIGH,
        lidar=LiDARNoiseConfig(
            range_std=0.10,
            dropout_rate=0.15,
            range_jitter_std=0.30,
        ),
        camera=CameraNoiseConfig(
            pixel_std=10.0,
            false_negative_rate=0.15,
            false_positive_rate=0.10,
        ),
    ),
}


def get_noise_profile(value: NoiseLevel | str | NoiseProfile) -> NoiseProfile:
    """解析内置等级或直接返回自定义 profile。"""
    if isinstance(value, NoiseProfile):
        return value
    return NOISE_PROFILES[NoiseLevel(value)]


__all__ = [
    "CAMERA_NOISE_CLEAN",
    "LIDAR_NOISE_CLEAN",
    "NOISE_PROFILES",
    "CameraNoiseConfig",
    "LiDARNoiseConfig",
    "NoiseLevel",
    "NoiseProfile",
    "get_noise_profile",
]
