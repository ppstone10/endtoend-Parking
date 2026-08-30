"""单 checkpoint 的任务分组误差与预测—专家叠加诊断。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DatasetGenerator
from metrics import (
    analyze_prediction_errors,
    collect_open_loop_predictions,
    public_sample_metric,
)
from training.checkpoint import load_model_checkpoint
from training.data import model_horizon, prepare_batches, validate_model_dataset
from training.reporting import atomic_write_json
from viz.prediction_analysis import save_grouped_metrics, save_prediction_overlays


def run_prediction_analysis(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    batch_size: int = 8,
    device: str = "cpu",
    overlay_count: int = 6,
) -> dict[str, Any]:
    """生成分组报告和最差样本叠加图。"""
    if overlay_count <= 0:
        raise ValueError("overlay_count 必须为正")
    data_source = Path(data_path).resolve()
    checkpoint_source = Path(checkpoint_path).resolve()
    destination = Path(output_dir).resolve()
    data = DatasetGenerator.load(data_source)
    metadata = data.get("task_meta")
    if not isinstance(data.get("bev_meta"), dict) or not isinstance(metadata, list):
        raise ValueError("预测诊断要求 schema v2 的 bev_meta/task_meta")
    loaded = load_model_checkpoint(checkpoint_source, device=device)
    validate_model_dataset(loaded.model, data)
    batches = prepare_batches(
        data,
        horizon=model_horizon(loaded.model),
        batch_size=batch_size,
    )
    predictions = collect_open_loop_predictions(loaded.model, batches, device=device)
    analysis, sample_metrics = analyze_prediction_errors(
        predictions.points,
        predictions.targets,
        predictions.masks,
        metadata,
        stop_logits=predictions.stop_logits,
        stop_threshold=float(getattr(loaded.model, "stop_threshold", 0.5)),
    )
    analysis["overall"]["inference_ms_per_sample"] = predictions.inference_ms_per_sample
    worst = sorted(sample_metrics, key=lambda item: item["fde_m"], reverse=True)
    worst_overall = [int(item["index"]) for item in worst[:overlay_count]]
    worst_by_task = []
    for task_type in sorted({str(item["task_type"]) for item in sample_metrics}):
        candidate = max(
            (item for item in sample_metrics if item["task_type"] == task_type),
            key=lambda item: item["fde_m"],
        )
        worst_by_task.append(int(candidate["index"]))

    destination.mkdir(parents=True, exist_ok=True)
    grouped_paths = save_grouped_metrics(
        analysis["groups"], destination / "grouped_metrics"
    )
    overall_paths = save_prediction_overlays(
        data,
        predictions.points,
        sample_metrics,
        worst_overall,
        destination / "worst_overall",
        title="Worst open-loop predictions by FDE",
    )
    task_paths = save_prediction_overlays(
        data,
        predictions.points,
        sample_metrics,
        worst_by_task,
        destination / "worst_by_task",
        title="Worst open-loop prediction in each task type",
    )
    report = {
        "schema_version": 1,
        "dataset": str(data_source),
        "checkpoint": str(loaded.checkpoint),
        "model_name": loaded.model_name,
        "model_config": loaded.model_config,
        "epoch": loaded.epoch,
        **analysis,
        "worst_samples": [public_sample_metric(item) for item in worst[:20]],
        "artifacts": {
            "grouped_metrics_png": grouped_paths[0],
            "grouped_metrics_pdf": grouped_paths[1],
            "worst_overall_png": overall_paths[0],
            "worst_overall_pdf": overall_paths[1],
            "worst_by_task_png": task_paths[0],
            "worst_by_task_pdf": task_paths[1],
        },
    }
    atomic_write_json(destination / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成单 checkpoint 的分组误差与预测—专家叠加图"
    )
    parser.add_argument("--data", required=True, help="schema v2 数据集 NPZ")
    parser.add_argument("--checkpoint", required=True, help="Trainer schema v1 checkpoint")
    parser.add_argument("--output", required=True, help="报告和图片输出目录")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overlays", type=int, default=6)
    args = parser.parse_args()
    report = run_prediction_analysis(
        args.data,
        args.checkpoint,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
        overlay_count=args.overlays,
    )
    print(f"完成：{Path(args.output).resolve() / 'report.json'}")
    print(
        f"ADE={report['overall']['ade_m']:.4f}m "
        f"FDE={report['overall']['fde_m']:.4f}m"
    )


if __name__ == "__main__":
    main()
