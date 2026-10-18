"""将自由反馈近端运动学指标与实际闭环结果按 dataset_index 对齐。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import DatasetGenerator
from metrics import collect_open_loop_predictions
from training.checkpoint import load_model_checkpoint
from training.data import prepare_batches, model_horizon, validate_model_dataset
from training.reporting import atomic_write_json


def prefix_metrics(points: np.ndarray, *, dt: float) -> dict:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(points).all():
        raise ValueError("近端轨迹必须至少两个有限位姿，dt 必须为正")
    xy = np.diff(points[:, :2], axis=0)
    yaw_delta = np.arctan2(np.sin(np.diff(points[:, 2])), np.cos(np.diff(points[:, 2])))
    midpoint = points[:-1, 2] + yaw_delta / 2
    lateral = np.abs(-np.sin(midpoint) * xy[:, 0] + np.cos(midpoint) * xy[:, 1])
    return {
        "start_position_error_m": float(np.linalg.norm(points[0, :2])),
        "start_yaw_error_rad": float(abs(np.arctan2(np.sin(points[0, 2]), np.cos(points[0, 2])))),
        "max_lateral_residual_m": float(lateral.max()),
        "max_speed_mps": float(np.linalg.norm(xy, axis=1).max() / dt),
        "max_yaw_rate_radps": float(np.abs(yaw_delta).max() / dt),
    }


def align_outcomes(report: dict, samples: int, task_ids: list[str] | None = None) -> dict[int, dict]:
    rows = report["episodes"]
    mapping = {int(row["meta"]["dataset_index"]): row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(range(samples)):
        raise ValueError("闭环报告必须覆盖数据集全部索引且无重复")
    if task_ids is not None and any(mapping[index]["meta"].get("task_id") != identity for index, identity in enumerate(task_ids)):
        raise ValueError("闭环任务身份与当前数据顺序不一致")
    return mapping


def summarize_prefix_rows(rows: list[dict]) -> dict:
    summary = {"samples": len(rows)}
    if not rows:
        return summary
    for source in ("network", "expert"):
        summary[source] = {}
        for key in rows[0][source]:
            values = np.asarray([row[source][key] for row in rows])
            summary[source][key] = {"mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}
        summary[source]["lateral_over_002_rate"] = float(np.mean([row[source]["max_lateral_residual_m"] > 0.02 for row in rows]))
    return summary


def run_analysis(baseline_path: Path, output: Path, *, prefix_points=6):
    if prefix_points < 2:
        raise ValueError("prefix_points 至少为 2")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "completed" or baseline["experiment"]["name"] != "E5" or baseline["experiment"]["safety_mode"] != "none":
        raise ValueError("配对分析要求已完成的无干预 E5 报告")
    protocol = baseline["protocol"]
    data = DatasetGenerator.load(Path(protocol["data"]))
    outcomes = align_outcomes(baseline, len(data["task_meta"]), [meta["task_id"] for meta in data["task_meta"]])
    loaded = load_model_checkpoint(protocol["checkpoint"])
    validate_model_dataset(loaded.model, data)
    batches = prepare_batches(data, horizon=model_horizon(loaded.model), batch_size=8)
    predictions = collect_open_loop_predictions(loaded.model, batches)
    dt = float(np.asarray(data["dt"]).reshape(-1)[0])
    rows = []
    for index, meta in enumerate(data["task_meta"]):
        count = min(prefix_points, int(predictions.masks[index].sum()))
        outcome = outcomes[index]
        rows.append({
            "dataset_index": index, "task_id": meta["task_id"], "scene_name": meta["scene_name"],
            "maneuver": meta["difficulty"]["maneuver"], "outcome": "success" if outcome["success"] else outcome["failure"],
            "network": prefix_metrics(predictions.points[index, :count], dt=dt),
            "expert": prefix_metrics(predictions.targets[index, :count], dt=dt),
        })
    grouped = {}
    for scene in sorted({row["scene_name"] for row in rows}):
        grouped[scene] = {outcome: summarize_prefix_rows([row for row in rows if row["scene_name"] == scene and row["outcome"] == outcome])
                          for outcome in sorted({row["outcome"] for row in rows if row["scene_name"] == scene})}
    result = {
        "baseline": str(baseline_path.resolve()), "checkpoint": protocol["checkpoint"],
        "prefix_points": prefix_points, "dt": dt, "samples": len(rows),
        "by_outcome": {outcome: summarize_prefix_rows([row for row in rows if row["outcome"] == outcome]) for outcome in sorted({row["outcome"] for row in rows})},
        "by_scene_outcome": grouped, "rows": rows,
        "limits": "初始存档输入上的统计关联；不证明闭环访问状态的因果；时间前缀不等同MPC实际窗口",
    }
    atomic_write_json(output, result)
    print(json.dumps(result["by_outcome"], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix-points", default=6, type=int)
    args = parser.parse_args()
    run_analysis(args.baseline_report, args.output, prefix_points=args.prefix_points)


if __name__ == "__main__":
    main()
