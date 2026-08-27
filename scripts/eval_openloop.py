"""一个或多个 Trainer checkpoint 的统一验证集开环评估。"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DatasetGenerator
from metrics import evaluate_open_loop
from training.checkpoint import load_model_checkpoint
from training.data import model_horizon, prepare_batches, validate_model_dataset
from training.reporting import atomic_write_json
from viz.open_loop import save_open_loop_comparison


def run_evaluation(
    data_path: str | Path,
    checkpoint_references: Iterable[str],
    *,
    output_dir: str | Path,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict:
    """在同一数据集上比较 checkpoint 并写入 JSON/PNG/PDF。"""
    data_source = Path(data_path).resolve()
    data = DatasetGenerator.load(data_source)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict] = {}
    for reference in checkpoint_references:
        label, checkpoint = _parse_reference(reference)
        if label in models:
            raise ValueError(f"checkpoint 标签重复：{label}")
        loaded = load_model_checkpoint(checkpoint, device=device)
        validate_model_dataset(loaded.model, data)
        batches = prepare_batches(
            data,
            horizon=model_horizon(loaded.model),
            batch_size=batch_size,
        )
        metrics = evaluate_open_loop(loaded.model, batches, device=device).to_dict()
        models[label] = {
            **metrics,
            "model_name": loaded.model_name,
            "model_config": loaded.model_config,
            "checkpoint": str(loaded.checkpoint),
            "epoch": loaded.epoch,
        }
        print(
            f"{label}: ADE={metrics['ade_m']:.4f}m "
            f"FDE={metrics['fde_m']:.4f}m",
            flush=True,
        )
    if not models:
        raise ValueError("至少需要一个 --checkpoint")
    plot_paths = save_open_loop_comparison(
        models, destination / "openloop_comparison"
    )
    report = {
        "schema_version": 1,
        "dataset": str(data_source),
        "models": models,
        "artifacts": {"plot_png": plot_paths[0], "plot_pdf": plot_paths[1]},
    }
    atomic_write_json(destination / "report.json", report)
    return report


def _parse_reference(reference: str) -> tuple[str, Path]:
    if "=" in reference:
        label, raw_path = reference.split("=", 1)
        if not label.strip() or not raw_path.strip():
            raise ValueError("checkpoint 必须为 LABEL=PATH 或 PATH")
        return label.strip(), Path(raw_path).resolve()
    path = Path(reference).resolve()
    return path.parent.name or path.stem, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在同一验证集上比较一个或多个 Trainer checkpoint"
    )
    parser.add_argument("--data", required=True, help="验证集 NPZ")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="可重复；格式 LABEL=PATH 或 PATH",
    )
    parser.add_argument("--output", default="runs/openloop-eval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_evaluation(
        args.data,
        args.checkpoint,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"报告：{Path(args.output).resolve() / 'report.json'}")


if __name__ == "__main__":
    main()
