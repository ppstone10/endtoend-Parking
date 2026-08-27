"""运行或恢复全专家能力单元校准。"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import CalibrationSettings, run_calibration
from sim import MINING_DRILL_RIG, load_vehicle_config


def main() -> int:
    parser = argparse.ArgumentParser(description="校准全部专家可生成任务单元")
    parser.add_argument("--samples-per-cell", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--task-budget-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vehicle-config", type=Path, default=None)
    args = parser.parse_args()
    vehicle_config = (
        MINING_DRILL_RIG
        if args.vehicle_config is None
        else load_vehicle_config(args.vehicle_config)
    )
    try:
        report = run_calibration(
            args.output,
            CalibrationSettings(
                samples_per_cell=args.samples_per_cell,
                seed=args.seed,
                max_retries=args.max_retries,
                task_budget_s=args.task_budget_s,
            ),
            vehicle_config,
        )
    except KeyboardInterrupt:
        print(f"校准已中断；使用相同命令可从 {args.output} 恢复", flush=True)
        return 130
    print(
        f"校准完成：{report['completed_cases']}/{report['planned_cases']}，"
        f"成功 {report['status_counts'].get('success', 0)}，"
        f"失败 {report['status_counts'].get('failed', 0)}，"
        f"预算超时 {report['status_counts'].get('task_budget_exceeded', 0)}",
        flush=True,
    )
    print(f"报告：{args.output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
