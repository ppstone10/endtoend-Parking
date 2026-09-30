"""闭环五组对照实验记分卡入口：读报告 → 对照表 + 判据核对 + 决策树归因。

用法：

    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/summarize_validation.py \
        --input runs/validation/main --output runs/validation

输入目录下每个 ``<实验>/report.json``（或 ``<实验>@<权重>/report.json``）都被纳入，
输出 ``summary.md`` / ``summary.json``。本脚本不跑实验，只做汇总与判定。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.comparison import build_scorecard, format_markdown, load_reports
from training.reporting import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/validation/main")
    parser.add_argument("--output", default="runs/validation")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="把 Markdown 记分卡同时打印到标准输出（不轮询、只打印一次）",
    )
    args = parser.parse_args()

    reports = load_reports(args.input)
    if not reports:
        raise SystemExit(f"{Path(args.input).resolve()} 下没有找到任何 */report.json")
    scorecard = build_scorecard(reports)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    markdown = format_markdown(scorecard)
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    atomic_write_json(output / "summary.json", scorecard)
    decision = scorecard["decision"]
    print(f"记分卡：{output / 'summary.md'}（{len(scorecard['rows'])} 组）")
    print(f"决策落点：{decision['branch']} —— {decision['reason']}")
    if args.stdout:
        print()
        print(markdown)


if __name__ == "__main__":
    main()
