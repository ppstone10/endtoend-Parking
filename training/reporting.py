"""训练历史、完成报告与曲线产物。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from viz.style import COLORS, save_fig, setup_style

from .trainer import TrainingHistory


@dataclass(frozen=True)
class TrainingArtifacts:
    history_json: str
    report_json: str
    curve_png: str
    curve_pdf: str


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """同目录临时文件替换，避免中断留下半份 JSON。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def save_training_artifacts(
    history: TrainingHistory,
    report: dict[str, Any],
    output_dir: str | Path,
) -> TrainingArtifacts:
    """保存可机读历史/报告和论文可用的双格式训练诊断图。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    loss_axis, rollout_axis, schedule_axis, stop_axis = axes.flatten()
    epochs = list(range(1, len(history.train_loss) + 1))
    loss_axis.plot(epochs, history.train_loss, color=COLORS["actual"], label="Train loss")
    loss_axis.plot(epochs, history.val_loss, color=COLORS["plan"], label="Validation loss")
    if history.val_collision_loss and any(value > 0.0 for value in history.val_collision_loss):
        loss_axis.plot(
            epochs,
            history.val_collision_loss,
            color=COLORS["spot"],
            linestyle=":",
            label="Validation collision loss (raw)",
        )
    if history.best_epoch >= 0:
        loss_axis.axvline(history.best_epoch + 1, color=COLORS["expert"], linestyle="--", label="Best epoch")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Optimization and validation loss")
    loss_axis.legend()

    if history.train_rollout_ade_m:
        rollout_axis.plot(epochs, history.train_rollout_ade_m, color=COLORS["actual"], label="Train ADE")
        rollout_axis.plot(epochs, history.train_rollout_fde_m, color=COLORS["actual"], linestyle="--", label="Train FDE")
        rollout_axis.plot(epochs, history.val_rollout_ade_m, color=COLORS["plan"], label="Validation ADE")
        rollout_axis.plot(epochs, history.val_rollout_fde_m, color=COLORS["plan"], linestyle="--", label="Validation FDE")
    rollout_axis.set_xlabel("Epoch")
    rollout_axis.set_ylabel("Distance error (m)")
    rollout_axis.set_title("Free-roll trajectory error")
    rollout_axis.legend()

    if history.teacher_forcing_ratio:
        schedule_axis.plot(epochs, history.teacher_forcing_ratio, color=COLORS["expert"])
    if history.early_stopping_active and any(history.early_stopping_active):
        first_active = history.early_stopping_active.index(True) + 1
        schedule_axis.axvline(
            first_active,
            color=COLORS["spot"],
            linestyle="--",
            label="Early stopping active",
        )
        schedule_axis.legend()
    schedule_axis.set_ylim(-0.02, 1.02)
    schedule_axis.set_xlabel("Epoch")
    schedule_axis.set_ylabel("Ratio")
    schedule_axis.set_title("Teacher-forcing schedule")

    stop_values = history.val_stop_found_rate
    if stop_values and any(value is not None for value in stop_values):
        stop_axis.plot(epochs, stop_values, color=COLORS["expert"], label="Stop found rate")
        stop_axis.set_ylim(-0.02, 1.02)
        length_axis = stop_axis.twinx()
        length_axis.plot(
            epochs,
            history.val_predicted_length_mae_points,
            color=COLORS["spot"],
            linestyle="--",
            label="Length MAE",
        )
        length_axis.set_ylabel("Length MAE (points)")
        handles, labels = stop_axis.get_legend_handles_labels()
        second_handles, second_labels = length_axis.get_legend_handles_labels()
        stop_axis.legend(handles + second_handles, labels + second_labels)
    else:
        stop_axis.text(0.5, 0.5, "Fixed-horizon model", ha="center", va="center", transform=stop_axis.transAxes)
    stop_axis.set_xlabel("Epoch")
    stop_axis.set_ylabel("Rate")
    stop_axis.set_title("Variable-length stopping")
    curve_paths = save_fig(figure, str(destination / "training_curve"))
    plt.close(figure)
    history_path = atomic_write_json(destination / "history.json", history.to_dict())
    report_path = atomic_write_json(destination / "report.json", report)
    return TrainingArtifacts(
        history_json=str(history_path),
        report_json=str(report_path),
        curve_png=curve_paths[0],
        curve_pdf=curve_paths[1],
    )
