"""使用独立验证集校准变长模型的停止阈值。"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def calibrate_stop_threshold(
    stop_logits: np.ndarray,
    masks: np.ndarray,
    *,
    thresholds: Iterable[float] | None = None,
) -> dict:
    """选择预测长度 MAE 最小的阈值，并返回完整扫描证据。"""
    logits = np.asarray(stop_logits, dtype=np.float64)
    active = np.asarray(masks)
    if logits.ndim != 2 or active.shape != logits.shape:
        raise ValueError("stop_logits 与 masks 必须是相同的二维 shape")
    if not np.isfinite(logits).all() or not np.isfinite(active).all():
        raise ValueError("stop_logits 与 masks 必须为有限值")
    if not np.logical_or(active == 0, active == 1).all():
        raise ValueError("masks 只能包含 0/1")
    if np.any(active[:, 1:] > active[:, :-1]):
        raise ValueError("masks 的有效点必须是连续前缀")
    lengths = active.sum(axis=1).astype(np.int64)
    if len(lengths) == 0 or np.any(lengths <= 0):
        raise ValueError("校准至少需要一条非空轨迹")
    values = (
        np.arange(0.05, 0.951, 0.01, dtype=np.float64)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=np.float64)
    )
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(values >= 1.0)
    ):
        raise ValueError("threshold 网格必须是 (0,1) 内有限非空序列")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    candidates: list[dict[str, float]] = []
    for value in values.tolist():
        reached = probabilities >= value
        found = reached.any(axis=1)
        predicted = np.where(found, reached.argmax(axis=1) + 1, logits.shape[1])
        errors = predicted - lengths
        candidates.append(
            {
                "threshold": float(value),
                "length_mae_points": float(np.abs(errors).mean()),
                "length_bias_points": float(errors.mean()),
                "stop_found_rate": float(found.mean()),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["length_mae_points"],
            abs(item["length_bias_points"]),
            -item["stop_found_rate"],
            abs(item["threshold"] - 0.5),
        ),
    )
    return {
        "objective": "min_length_mae_on_validation",
        "selected_threshold": selected["threshold"],
        "selected": selected,
        "candidates": candidates,
    }


def write_deployment_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    threshold: float,
    calibration: dict,
) -> Path:
    """从 best 复制只用于推理的阈值校准 checkpoint。"""
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold 必须位于 (0,1)")
    source_path = Path(source)
    target_path = Path(destination)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("仅支持 Trainer schema v1 checkpoint")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 缺少 model_config")
    deployed = copy.deepcopy(payload)
    deployed["model_config"]["stop_threshold"] = float(threshold)
    deployed["resumable"] = False
    deployed["stop_calibration"] = copy.deepcopy(calibration)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_name(f"{target_path.name}.tmp")
    torch.save(deployed, temporary)
    temporary.replace(target_path)
    return target_path
