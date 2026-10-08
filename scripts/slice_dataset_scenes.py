"""按场景切分数据集分片，用于"数据组成"对照实验。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/slice_dataset_scenes.py \
        --data data/task_dataset/tracked_pivot_v8_3000/train.npz \
        --scenes S1_parking_lot,S2_diagonal_lot,S4_dump_area,S6_loading_face \
        --output runs/validation/slice-4scenes/train-4scenes.npz

保留 schema v2 的全部字段，只按 `task_meta.scene_name` 过滤样本；
`bev_meta` 与 `dt` 原样沿用，因此切片与源数据同构。
"""

from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator


def slice_scenes(data_path: Path, output_path: Path, scenes: set[str]) -> dict:
    data = DatasetGenerator.load(data_path)
    metadata = list(data["task_meta"])
    keep = [
        index
        for index, item in enumerate(metadata)
        if str(item.get("scene_name")) in scenes
    ]
    if not keep:
        raise ValueError(f"没有任何样本属于场景 {sorted(scenes)}")

    np.savez_compressed(
        output_path,
        schema_version=np.asarray(int(data["schema_version"]), dtype=np.uint16),
        bev_meta=np.asarray(
            DatasetGenerator._encode_metadata(data["bev_meta"], "bev_meta"),
            dtype=np.str_,
        ),
        task_meta=np.asarray(
            [
                DatasetGenerator._encode_metadata(metadata[index], "task_meta")
                for index in keep
            ],
            dtype=np.str_,
        ),
        bevs=np.asarray(data["bevs"])[keep],
        goals=np.asarray(data["goals"])[keep],
        states=np.asarray(data["states"])[keep],
        trajs=np.asarray(data["trajs"])[keep],
        masks=np.asarray(data["masks"])[keep],
        dt=np.asarray(data["dt"]),
    )
    counts = collections.Counter(str(metadata[i]["scene_name"]) for i in keep)
    return {
        "source": str(data_path),
        "output": str(output_path),
        "kept": len(keep),
        "total": len(metadata),
        "scene_counts": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scenes", required=True, help="逗号分隔的场景名白名单"
    )
    args = parser.parse_args()
    scenes = {value.strip() for value in args.scenes.split(",") if value.strip()}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = slice_scenes(Path(args.data).resolve(), output, scenes)
    print(
        f"切片完成：保留 {report['kept']}/{report['total']} 条 → {report['output']}\n"
        f"  场景分布 {report['scene_counts']}"
    )


if __name__ == "__main__":
    main()
