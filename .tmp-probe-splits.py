"""临时探针：核查 test 划分与 train 划分在当前代码几何下的可复原率。"""

from __future__ import annotations

import sys
from pathlib import Path

from dataset import DatasetGenerator
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
)
from sim import VehicleConfig


def probe(name: str) -> None:
    data_path = Path(f"data/task_dataset/tracked_pivot_v7_3000/{name}.npz")
    data = DatasetGenerator.load(data_path)
    metadata = data["task_meta"]
    manifest = load_dataset_manifest(data_path)
    vehicle = VehicleConfig(**manifest["vehicle_model"])
    ok = 0
    failed_scenes: dict[str, int] = {}
    for item in metadata:
        try:
            reconstruct_dataset_task(item, root_seed=int(manifest["seed"]), vehicle=vehicle)
            ok += 1
        except ValueError:
            scene = str(item.get("scene_name"))
            failed_scenes[scene] = failed_scenes.get(scene, 0) + 1
    print(f"{name}: 可复原 {ok}/{len(metadata)}；不可复原场景分布 {failed_scenes}")


for split in ("test", "train"):
    probe(split)
