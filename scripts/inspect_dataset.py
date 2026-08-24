"""输出数据集统计，并可保存 BEV+专家轨迹抽检图。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator, render_sample_overlay, summarize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 schema v1/v2 NPZ 数据集")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples 不能为负")

    data = DatasetGenerator.load(args.path)
    summary = summarize_dataset(data)
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output is None:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(encoded, encoding="utf-8")
    count = int(summary["sample_count"])
    if args.samples and count:
        indices = np.linspace(
            0, count - 1, min(args.samples, count), dtype=int
        ).tolist()
        for index in dict.fromkeys(indices):
            render_sample_overlay(data, index, args.output / f"sample_{index:05d}.png")


if __name__ == "__main__":
    main()
