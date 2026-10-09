"""比较多个训练配方的 checkpoint 闭环排序，回答"哪个改动真的抬高了上界"。

输入是 `scripts/rank_checkpoints.py` 产出的排名 JSON。输出：
- 每个配置的**上界**（最优 checkpoint 的闭环成功率）与**中位**（代表平均 epoch 质量）；
- 各配置内相邻评估轮之间的区间类型（warmup/rising/plateau/falling），
  用来说明"上界提升"是来自更多时间落在高分区，还是只是换了个幸运 epoch；
- 与基线配置的上界差。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/compare_recipes.py \
        --label v16=baseline=runs/validation/rank-v16.json \
        --label v17=endpoint=runs/validation/rank-v17.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _classify(previous: float | None, current: float) -> str:
    """按闭环成功率的变化幅度给相邻评估轮打区间标签。"""
    if previous is None:
        return "warmup"
    delta = current - previous
    if delta >= 0.05:
        return "rising"
    if delta <= -0.05:
        return "falling"
    return "plateau"


def summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    # ranking 已按成功率降序，按 epoch 重新排序以刻画时间趋势。
    by_epoch = sorted(report["ranking"], key=lambda row: row["epoch"])
    segments = []
    previous: float | None = None
    for row in by_epoch:
        segments.append(
            {
                "epoch": row["epoch"],
                "success_rate": row["success_rate"],
                "segment": _classify(previous, row["success_rate"]),
            }
        )
        previous = row["success_rate"]
    best = report["best"]
    return {
        "path": str(path),
        "candidates": len(report["ranking"]),
        "samples": report["samples"],
        "best_success_rate": report["best_success_rate"],
        "best_checkpoint": best["checkpoint"],
        "best_epoch": best["epoch"],
        "median_success_rate": report["median_success_rate"],
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        metavar="NAME=DESC=PATH",
        help="可重复；NAME 用于对齐，DESC 是配方说明，PATH 是 rank_checkpoints 报告",
    )
    parser.add_argument("--baseline", default="", help="作为差值的基线 NAME")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    entries = []
    for raw in args.label:
        parts = raw.split("=", 2)
        if len(parts) != 3:
            raise SystemExit(f"--label 需要 NAME=DESC=PATH 形式，收到 {raw!r}")
        name, description, path = parts
        entries.append((name, description, summarize(Path(path).resolve())))

    baseline = args.baseline or entries[0][0]
    reference = next((item[2] for item in entries if item[0] == baseline), None)

    print(f"{'配方':22s} {'候选':>4s} {'上界':>7s} {'中位':>7s} {'最优 epoch':>10s} {'相对基线':>9s}")
    for name, description, summary in entries:
        delta = ""
        if reference is not None and summary is not reference:
            delta = f"{summary['best_success_rate'] - reference['best_success_rate']:+.1%}"
        print(
            f"{name + ' ' + description:22s} {summary['candidates']:>4d} "
            f"{summary['best_success_rate']:>6.1%} {summary['median_success_rate']:>6.1%} "
            f"{summary['best_epoch']:>10d} {delta:>9s}"
        )

    print()
    print("逐轮区间（按 epoch 升序；rising/plateau/falling 由 ±5pt 阈值判定）:")
    for name, description, summary in entries:
        segments = summary["segments"]
        text = " ".join(f"{seg['epoch']}:{seg['success_rate']:.2f}{seg['segment'][0]}" for seg in segments)
        counts = {
            kind: sum(1 for seg in segments if seg["segment"] == kind)
            for kind in ("rising", "plateau", "falling")
        }
        print(f"  {name}（{description}）{text}")
        print(
            f"     区间计数 {counts}；高分区(>=20%)轮数 "
            f"{sum(1 for seg in segments if seg['success_rate'] >= 0.20)}"
        )

    if args.output:
        payload = {
            "baseline": baseline,
            "recipes": [
                {"name": name, "description": description, **summary}
                for name, description, summary in entries
            ],
        }
        destination = Path(args.output).resolve()
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告：{destination}")


if __name__ == "__main__":
    main()
