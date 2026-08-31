"""在同一验证集和净空定义下比较多个轨迹模型 checkpoint。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dataset import DatasetGenerator
from metrics.open_loop import evaluate_open_loop
from training.checkpoint import load_model_checkpoint
from training.data import model_horizon, prepare_batches
from training.reporting import atomic_write_json
from training.safety import (
    SweptFootprintLoss,
    build_clearance_fields,
    safety_geometry_from_dataset,
)


def _checkpoint_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("checkpoint 必须写成 LABEL=PATH")
    return label.strip(), Path(path).resolve()


def evaluate_checkpoint(
    label: str,
    checkpoint: Path,
    data: dict,
    clearance_fields: torch.Tensor,
    safety_loss: SweptFootprintLoss,
    *,
    batch_size: int,
) -> dict:
    loaded = load_model_checkpoint(checkpoint)
    batches = prepare_batches(
        data,
        horizon=model_horizon(loaded.model),
        batch_size=batch_size,
        clearance_fields=clearance_fields,
    )
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in batches:
            bev, goal, state, _target, mask, clearance = batch
            forward_with_stop = getattr(loaded.model, "forward_with_stop", None)
            if callable(forward_with_stop):
                points = forward_with_stop(bev, goal, state).points
            else:
                points = loaded.model(bev, goal, state)
            count = int(bev.shape[0])
            weighted_loss += float(
                safety_loss(bev, points, mask, clearance, goal).cpu()
            ) * count
            sample_count += count
    metrics = evaluate_open_loop(loaded.model, batches).to_dict()
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": loaded.epoch,
        "samples": sample_count,
        "clearance_loss": weighted_loss / sample_count,
        "open_loop": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="以统一连续净空损失和开环指标复算安全训练消融"
    )
    parser.add_argument("--data", required=True, help="schema v2 验证集 NPZ")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=_checkpoint_argument,
        metavar="LABEL=PATH",
        help="待评估 checkpoint，可重复传入",
    )
    parser.add_argument("--output", required=True, help="消融 JSON 输出路径")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extra-margin-m", type=float, default=0.1)
    parser.add_argument("--goal-exempt-radius-m", type=float, default=0.0)
    parser.add_argument("--goal-exempt-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须为正")
    if args.goal_exempt_radius_m < 0.0 or not 0.0 <= args.goal_exempt_weight <= 1.0:
        parser.error("goal-exempt-radius-m 必须非负且 goal-exempt-weight 位于 [0,1]")

    data_path = Path(args.data).resolve()
    data = DatasetGenerator.load(data_path)
    geometry = safety_geometry_from_dataset(data)
    clearance_fields = build_clearance_fields(
        data["bevs"], geometry, extra_margin_m=args.extra_margin_m
    )
    safety_loss = SweptFootprintLoss(
        geometry,
        extra_margin_m=args.extra_margin_m,
        mode="clearance_field",
        goal_exempt_radius_m=args.goal_exempt_radius_m,
        goal_exempt_weight=args.goal_exempt_weight,
    )
    results = [
        evaluate_checkpoint(
            label,
            checkpoint,
            data,
            clearance_fields,
            safety_loss,
            batch_size=args.batch_size,
        )
        for label, checkpoint in args.checkpoint
    ]
    baseline = results[0]["clearance_loss"]
    for result in results:
        result["clearance_loss_reduction_vs_first"] = (
            (baseline - result["clearance_loss"]) / baseline
        )
    report = {
        "schema_version": 1,
        "data": str(data_path),
        "metric": {
            "name": "continuous_swept_footprint_clearance_loss",
            "extra_margin_m": args.extra_margin_m,
            "geometry": geometry.to_dict(),
            "goal_exempt_radius_m": args.goal_exempt_radius_m,
            "goal_exempt_weight": args.goal_exempt_weight,
            "baseline_label": results[0]["label"],
        },
        "results": results,
    }
    output = atomic_write_json(Path(args.output).resolve(), report)
    for result in results:
        metrics = result["open_loop"]
        print(
            f"{result['label']}: clearance={result['clearance_loss']:.6f}, "
            f"reduction={result['clearance_loss_reduction_vs_first']:.1%}, "
            f"ADE={metrics['ade_m']:.4f}m, FDE={metrics['fde_m']:.4f}m"
        )
    print(f"报告：{output}")


if __name__ == "__main__":
    main()
