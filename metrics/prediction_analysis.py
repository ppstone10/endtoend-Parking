"""逐样本预测误差、终止长度与任务元数据分组统计。"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np
import torch

from training.trainer import Batch


@dataclass(frozen=True)
class PredictionBatchResult:
    """按数据集原顺序收集的自由滚动预测。"""

    points: np.ndarray
    stop_logits: np.ndarray | None
    targets: np.ndarray
    masks: np.ndarray
    inference_ms_per_sample: float


def collect_open_loop_predictions(
    model: torch.nn.Module,
    batches: Iterable[Batch],
    *,
    device: str | torch.device = "cpu",
) -> PredictionBatchResult:
    """执行无 teacher forcing 推理并保留逐样本输出。"""
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    points: list[torch.Tensor] = []
    stops: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    elapsed = 0.0
    sample_count = 0
    variable_output: bool | None = None
    with torch.no_grad():
        for batch in batches:
            bev, goal, state, target, mask = (
                tensor.to(target_device) for tensor in batch[:5]
            )
            started = time.perf_counter()
            forward_with_stop = getattr(model, "forward_with_stop", None)
            if callable(forward_with_stop):
                output = forward_with_stop(bev, goal, state)
                batch_points = output.points
                batch_stops = output.stop_logits
                current_variable = True
            else:
                batch_points = model(bev, goal, state)
                batch_stops = None
                current_variable = False
            elapsed += time.perf_counter() - started
            if variable_output is not None and variable_output != current_variable:
                raise ValueError("模型输出类型在 batch 间不一致")
            variable_output = current_variable
            if batch_points.ndim != 3 or batch_points.shape[2] != 3:
                raise ValueError("模型轨迹输出必须是 (B,N,3)")
            if batch_stops is not None and batch_stops.shape != batch_points.shape[:2]:
                raise ValueError("停止 logits 必须与轨迹输出前两维一致")
            sample_count += int(bev.shape[0])
            points.append(batch_points.detach().cpu())
            if batch_stops is not None:
                stops.append(batch_stops.detach().cpu())
            targets.append(target.detach().cpu())
            masks.append(mask.detach().cpu())
    if sample_count == 0:
        raise ValueError("评估 batches 不能为空")
    return PredictionBatchResult(
        points=torch.cat(points).numpy(),
        stop_logits=(torch.cat(stops).numpy() if stops else None),
        targets=torch.cat(targets).numpy(),
        masks=torch.cat(masks).numpy(),
        inference_ms_per_sample=elapsed * 1000.0 / sample_count,
    )


def analyze_prediction_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    stop_logits: np.ndarray | None = None,
    stop_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """返回总体/分组统计和带内部误差数组的逐样本记录。"""
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    active = np.asarray(masks)
    if predicted.ndim != 3 or predicted.shape[2] != 3:
        raise ValueError("predictions 必须是 (B,N,3)")
    if expected.shape != predicted.shape or active.shape != predicted.shape[:2]:
        raise ValueError("predictions、targets 与 masks 形状必须一致")
    if not (
        np.isfinite(predicted).all()
        and np.isfinite(expected).all()
        and np.isfinite(active).all()
    ):
        raise ValueError("predictions、targets 与 masks 必须为有限值")
    sample_count = predicted.shape[0]
    if len(metadata) != sample_count or any(not isinstance(item, dict) for item in metadata):
        raise ValueError("task_meta 必须与预测逐项对齐")
    if not np.logical_or(active == 0, active == 1).all():
        raise ValueError("masks 只能包含 0/1")
    if np.any(active[:, 1:] > active[:, :-1]):
        raise ValueError("masks 的有效点必须是连续前缀")
    lengths = active.sum(axis=1).astype(int)
    if np.any(lengths <= 0):
        raise ValueError("每条样本至少需要一个有效点")
    stops = None if stop_logits is None else np.asarray(stop_logits, dtype=np.float64)
    if stops is not None and stops.shape != predicted.shape[:2]:
        raise ValueError("stop_logits 必须与预测轨迹前两维一致")
    if stops is not None and not np.isfinite(stops).all():
        raise ValueError("stop_logits 必须为有限值")
    if not 0.0 < stop_threshold < 1.0:
        raise ValueError("stop_threshold 必须位于 (0,1)")

    rows: list[dict[str, Any]] = []
    for index, length in enumerate(lengths.tolist()):
        delta_xy = predicted[index, :length, :2] - expected[index, :length, :2]
        position_errors = np.linalg.norm(delta_xy, axis=1)
        yaw_delta = predicted[index, :length, 2] - expected[index, :length, 2]
        yaw_errors = np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
        predicted_length: int | None = None
        stop_found: bool | None = None
        if stops is not None:
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(stops[index], -80.0, 80.0)))
            candidates = np.flatnonzero(probabilities >= stop_threshold)
            stop_found = bool(len(candidates))
            predicted_length = int(candidates[0]) + 1 if stop_found else predicted.shape[1]
        item = metadata[index]
        difficulty = item.get("difficulty", {})
        if not isinstance(difficulty, dict):
            difficulty = {}
        rows.append(
            {
                "index": index,
                "task_id": str(item.get("task_id", f"sample-{index}")),
                "scene": str(item.get("scene_name", "unknown")),
                "task_type": str(item.get("task_type", "unknown")),
                "maneuver": str(difficulty.get("maneuver", "unknown")),
                "noise_level": str(difficulty.get("noise_level", "unknown")),
                "adjacent_occupancy": str(difficulty.get("adjacent_occupancy", "unknown")),
                "target_length": length,
                "predicted_length": predicted_length,
                "stop_found": stop_found,
                "ade_m": float(position_errors.mean()),
                "fde_m": float(position_errors[-1]),
                "yaw_mae_rad": float(yaw_errors.mean()),
                "_position_errors": position_errors,
                "_yaw_errors": yaw_errors,
            }
        )

    dimensions = ("scene", "task_type", "maneuver", "noise_level", "adjacent_occupancy")
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in dimensions:
        values = sorted({str(row[dimension]) for row in rows})
        groups[dimension] = {
            value: _aggregate([row for row in rows if row[dimension] == value])
            for value in values
        }
    return {"overall": _aggregate(rows), "groups": groups}, rows


def public_sample_metric(row: dict[str, Any]) -> dict[str, Any]:
    """移除仅供聚合使用的数组，得到可 JSON 序列化记录。"""
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("不能聚合空样本组")
    positions = np.concatenate([row["_position_errors"] for row in rows])
    yaws = np.concatenate([row["_yaw_errors"] for row in rows])
    fde = np.asarray([row["fde_m"] for row in rows], dtype=np.float64)
    result: dict[str, Any] = {
        "samples": len(rows),
        "valid_points": int(len(positions)),
        "ade_m": float(positions.mean()),
        "fde_m": float(fde.mean()),
        "fde_p90_m": float(np.percentile(fde, 90.0)),
        "yaw_mae_rad": float(yaws.mean()),
    }
    length_rows = [row for row in rows if row["predicted_length"] is not None]
    if length_rows:
        errors = np.asarray(
            [row["predicted_length"] - row["target_length"] for row in length_rows],
            dtype=np.float64,
        )
        result.update(
            {
                "predicted_length_mae_points": float(np.abs(errors).mean()),
                "predicted_length_bias_points": float(errors.mean()),
                "stop_found_rate": float(
                    np.mean([bool(row["stop_found"]) for row in length_rows])
                ),
            }
        )
    return result
