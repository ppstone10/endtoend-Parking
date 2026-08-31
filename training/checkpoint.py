"""Trainer checkpoint 的只读模型恢复。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from model import build_model


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    model_name: str
    model_config: dict[str, Any]
    checkpoint: Path
    epoch: int


def load_model_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LoadedModel:
    """从 Trainer schema v1 checkpoint 恢复注册表模型。"""
    checkpoint = Path(path).resolve()
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"checkpoint 无法读取：{checkpoint}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("仅支持 Trainer schema v1 checkpoint")
    model_name = payload.get("model_name")
    model_config = payload.get("model_config")
    if not isinstance(model_name, str) or not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少模型名称或模型配置")
    model = build_model(model_name, model_config)
    try:
        model.load_state_dict(payload["model_state"])
    except (KeyError, RuntimeError) as exc:
        raise ValueError("checkpoint 模型权重与配置不兼容") from exc
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        model_name=model_name,
        model_config=dict(model_config),
        checkpoint=checkpoint,
        epoch=int(payload.get("epoch", -1)),
    )


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    model_name: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """只加载兼容模型权重，不继承优化器或训练进度。"""
    loaded = load_model_checkpoint(path)
    if loaded.model_name != model_name or loaded.model_config != model_config:
        raise ValueError("初始化 checkpoint 的模型名称或配置不一致")
    model.load_state_dict(loaded.model.state_dict())
    digest = hashlib.sha256()
    with loaded.checkpoint.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "policy": "model_weights_only",
        "checkpoint": str(loaded.checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "source_epoch": loaded.epoch,
    }


__all__ = [
    "LoadedModel",
    "initialize_model_from_checkpoint",
    "load_model_checkpoint",
]
