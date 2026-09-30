"""闭环五组对照实验入口（E1–E5）。

用法示例：

    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/run_validation_experiment.py \
        --experiment E1 --samples 34 --indices-file runs/validation/indices-34.json \
        --output runs/validation/E1-gt-expert-mpc

    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/run_validation_experiment.py \
        --experiment E2 --data data/task_dataset/tracked_pivot_v7_3000/val.npz \
        --model runs/training/v7-flow-v3/net-v1/deployment.pt \
        --samples 34 --output runs/validation/E2-gt-nn-ideal

``--list`` 打印可用实验组合。``--indices-file`` 用于跨实验锁定同一批索引。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.validation_matrix import (
    EXPERIMENT_SPECS,
    ExperimentSpec,
    run_validation_experiment,
)


def _print_episode(ordinal: int, total: int, index: int, bundle) -> None:
    result = bundle.result
    status = "成功" if result.success else f"失败({result.failure})"
    cycles = bundle.cycles
    print(
        f"[{ordinal}/{total}] #{index} {result.meta.get('task_id', '')} {status} "
        f"位置 {result.final_pos_err:.2f}m 航向 {np.degrees(result.final_yaw_err):.1f}° "
        f"步数 {result.steps} 周期 {cycles.get('cycles', 0)} "
        f"发散 {cycles.get('divergence_events', 0)} "
        f"drift {cycles.get('drift_slope')} "
        f"推理 {result.inference_ms:.1f}ms",
        flush=True,
    )


def _load_indices(path: str | None) -> list[int] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("indices", "selected_indices", "evaluated_indices"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"索引文件 {path} 缺少 indices/selected_indices 字段")
    if not isinstance(payload, list) or not all(isinstance(v, int) for v in payload):
        raise ValueError(f"索引文件 {path} 必须给出整数索引列表")
    return payload


def _summary_line(report: dict) -> str:
    overall = report["overall"]
    parts = [
        f"实验 {report['experiment']['name']}（{report['experiment']['description']}）",
        f"样本 {overall['evaluated_samples']}/{overall['requested_indices']}"
        f"（复原失败 {overall['reconstruct_failures']}）",
        f"成功率 {overall['success_rate']:.1%}",
        f"碰撞率 {overall['collision_rate']:.1%}",
        f"终点位置 {overall['final_pos_err_mean']:.2f}m",
        f"终点航向 {np.degrees(overall['final_yaw_err_mean']):.1f}°",
        f"跟踪RMS {overall['tracking_rms_mean']:.3f}m",
    ]
    if "cycles_drift_slope_mean" in overall:
        parts.append(f"drift{overall['cycles_drift_slope_mean']:+.4f}")
    if "cycles_divergence_events_mean" in overall:
        parts.append(f"发散{overall['cycles_divergence_events_mean']:.2f}/样本")
    if overall.get("failures"):
        parts.append(f"失败分类 {overall['failures']}")
    return "；".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="E1",
        help=f"实验组合：{sorted(EXPERIMENT_SPECS)}；或 'custom' 配合 --bev-source/--trajectory-source/--executor",
    )
    parser.add_argument("--bev-source", choices=["sensor", "gt"], default="sensor")
    parser.add_argument(
        "--trajectory-source",
        choices=["expert", "expert_replan", "network"],
        default="expert",
    )
    parser.add_argument("--executor", choices=["mpc_vehicle", "ideal_path"], default="mpc_vehicle")
    parser.add_argument(
        "--safety-mode",
        choices=["none", "expert_fallback", "hierarchical"],
        default="none",
    )
    parser.add_argument("--data", default="data/task_dataset/tracked_pivot_v7_3000/val.npz")
    parser.add_argument("--model", default="runs/training/v7-flow-v3/net-v1/deployment.pt")
    parser.add_argument("--samples", type=int, default=34, help="<=0 表示全部")
    parser.add_argument("--selection", choices=["stratified", "head"], default="stratified")
    parser.add_argument("--indices-file", default="")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ideal-steps-per-point",
        type=float,
        default=None,
        help="理想执行每前进一个轨迹点所用的控制周期数；默认按网络 dt/MPC dt 对齐",
    )
    parser.add_argument("--hierarchical-lookahead", type=float, default=3.0)
    parser.add_argument("--output", default="runs/validation/report.json")
    parser.add_argument("--list", action="store_true", help="列出可用实验组合")
    args = parser.parse_args()

    if args.list:
        for name, spec in sorted(EXPERIMENT_SPECS.items()):
            print(f"{name:5s} {spec.to_metadata()}")
        return

    if args.experiment == "custom":
        spec = ExperimentSpec(
            name="custom",
            bev_source=args.bev_source,
            trajectory_source=args.trajectory_source,
            executor=args.executor,
            safety_mode=args.safety_mode,
            description="自定义组合",
        )
    else:
        if args.experiment not in EXPERIMENT_SPECS:
            parser.error(f"未知实验组合：{args.experiment}")
        spec = EXPERIMENT_SPECS[args.experiment]
        if (
            args.bev_source != "sensor"
            or args.trajectory_source != "expert"
            or args.executor != "mpc_vehicle"
        ):
            spec = ExperimentSpec(
                name=spec.name,
                bev_source=args.bev_source,
                trajectory_source=args.trajectory_source,
                executor=args.executor,
                safety_mode=args.safety_mode,
                description=spec.description,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json" if output.suffix != ".json" else output

    def progress(ordinal: int, total: int, bundle) -> None:
        _print_episode(ordinal, total, bundle.result.meta["dataset_index"], bundle)

    report = run_validation_experiment(
        spec,
        data_path=args.data,
        checkpoint_path=args.model if spec.trajectory_source == "network" else None,
        output_path=report_path,
        samples=args.samples,
        selection=args.selection,
        max_steps=args.max_steps,
        replan_every=args.replan_every,
        control_seed=args.seed,
        indices=_load_indices(args.indices_file),
        ideal_steps_per_point=args.ideal_steps_per_point,
        hierarchical_lookahead=args.hierarchical_lookahead,
        progress=progress,
    )
    print(_summary_line(report))
    print(f"报告：{report_path.resolve()}")


if __name__ == "__main__":
    main()
