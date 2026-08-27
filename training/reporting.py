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
    """保存可机读历史/报告和论文可用的双格式 loss 曲线。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    setup_style()
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    epochs = range(1, len(history.train_loss) + 1)
    axis.plot(epochs, history.train_loss, color=COLORS["actual"], label="Train loss")
    axis.plot(epochs, history.val_loss, color=COLORS["plan"], label="Validation loss")
    if history.best_epoch >= 0:
        axis.axvline(history.best_epoch + 1, color=COLORS["expert"], linestyle="--", label="Best epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training history")
    axis.legend()
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
