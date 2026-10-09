"""对一次训练产出的候选 checkpoint 做离线闭环排序，测量该配置的闭环上界。

动机：训练内选型每 2 轮只看一个快照，且闭环指标逐轮在 0–25% 波动。要判断"某个训练侧改动
是否真的抬高了上界"，必须对**同一训练产出的多个 checkpoint**做同一批样本的闭环评估，
取最高分作为该配置的上界，再与基线配置的上界比较。

与训练内选型使用同一个 `ClosedLoopSelector`（同一分层样本集、同一评估口径），
因此结果与训练日志可直接对照。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/rank_checkpoints.py \
        --run runs/training/v16-closed-loop-selected/net-v1 \
        --data data/task_dataset/tracked_pivot_v8_3000/val.npz \
        --samples 60 --output runs/validation/rank-v16.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.checkpoint import load_model_checkpoint
from training.closed_loop_selection import ClosedLoopSelectionConfig, ClosedLoopSelector
from training.reporting import atomic_write_json


def _checkpoints(run_dir: Path) -> list[Path]:
    """收集可评估的 checkpoint：周期快照 + best + best_closed_loop + last。"""
    names = sorted(path.name for path in run_dir.glob("epoch*.pt"))
    for extra in ("best.pt", "best_closed_loop.pt", "last.pt", "deployment.pt"):
        if (run_dir / extra).is_file():
            names.append(extra)
    return [run_dir / name for name in dict.fromkeys(names)]


def rank(
    run_dir: Path,
    data_path: Path,
    *,
    samples: int,
    max_steps: int,
    replan_every: int,
    device: str,
) -> dict:
    config = ClosedLoopSelectionConfig(
        enabled=True,
        samples=samples,
        max_steps=max_steps,
        replan_every=replan_every,
    )
    selector = ClosedLoopSelector(config, data_path=data_path, device=device)
    # 预先固定任务集合，保证每个 checkpoint 评的是同一批样本。
    selector._ensure_episodes()

    rows: list[dict] = []
    started = time.perf_counter()
    for path in _checkpoints(run_dir):
        loaded = load_model_checkpoint(path, device=device)
        # 让选择器按该 checkpoint 的输入契约构造感知链路。
        setattr(loaded.model, "model_config", loaded.model_config)
        metrics = selector.evaluate(loaded.model)
        row = {
            "checkpoint": path.name,
            "epoch": loaded.epoch,
            "stop_threshold": float(getattr(loaded.model, "stop_threshold", 0.0)),
            **metrics,
        }
        rows.append(row)
        print(
            f"  {path.name:24s} epoch {loaded.epoch:>3} 成功 {metrics['success_rate']:6.1%} "
            f"碰撞 {metrics['collision_rate']:6.1%} 位置 {metrics['final_pos_err_mean']:.2f}m "
            f"失败 {metrics['failure_counts']}",
            flush=True,
        )
    rows.sort(key=lambda item: (-item["success_rate"], item["collision_rate"]))
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "data": str(data_path),
        "samples": selector.config.samples,
        "selected_indices": selector.selected_indices,
        "elapsed_sec": time.perf_counter() - started,
        "ranking": rows,
        "best": rows[0] if rows else None,
        "best_success_rate": rows[0]["success_rate"] if rows else None,
        "median_success_rate": (
            sorted(row["success_rate"] for row in rows)[len(rows) // 2] if rows else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="训练输出目录（含 epoch*.pt / best.pt）")
    parser.add_argument(
        "--data", default="data/task_dataset/tracked_pivot_v8_3000/val.npz"
    )
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"训练目录不存在：{run_dir}")
    report = rank(
        run_dir,
        Path(args.data).resolve(),
        samples=args.samples,
        max_steps=args.max_steps,
        replan_every=args.replan_every,
        device=args.device,
    )
    destination = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "checkpoint_ranking.json"
    )
    atomic_write_json(destination, report)
    print()
    print(
        f"共评估 {len(report['ranking'])} 个 checkpoint（{report['samples']} 样本，"
        f"{report['elapsed_sec']:.0f}s）"
    )
    print(
        f"  上界 {report['best_success_rate']:.1%}（{report['best']['checkpoint']}，"
        f"epoch {report['best']['epoch']}）"
    )
    print(f"  中位 {report['median_success_rate']:.1%}")
    print(f"报告：{destination}")


if __name__ == "__main__":
    main()
