"""MineParkingNet 模型注册表。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch.nn as nn

from .network import MineParkingNet
from .variants import MineParkingNetV1, MineParkingNetV2


ModelFactory = Callable[..., nn.Module]
_MODEL_REGISTRY: dict[str, ModelFactory] = {
    "net-v0": MineParkingNet,
    "net-v1": MineParkingNetV1,
    "net-v2": MineParkingNetV2,
}


def available_models() -> tuple[str, ...]:
    """返回稳定排序的可构造模型名称。"""
    return tuple(sorted(_MODEL_REGISTRY))


def build_model(name: str, cfg: Mapping[str, Any] | None = None) -> nn.Module:
    """按名称和可序列化配置构造模型。"""
    try:
        factory = _MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"未知模型 {name}，可选：{available_models()}") from exc
    config = {} if cfg is None else dict(cfg)
    try:
        return factory(**config)
    except TypeError as exc:
        raise ValueError(f"模型 {name} 配置无效：{exc}") from exc
