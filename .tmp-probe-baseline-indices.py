"""临时探针：提取既有 34 条基线索引中在当前代码下可复原的子集。"""

from __future__ import annotations

import json
from pathlib import Path

from dataset import DatasetGenerator
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
)
from sim import VehicleConfig

data_path = Path("data/task_dataset/tracked_pivot_v7_3000/val.npz")
baseline = json.loads(
    Path("runs/closed-loop/v7-flow-v3/net-v1/val-stratified-34-k10/report.json").read_text(
        encoding="utf-8"
    )
)
original = baseline["evaluation"]["selected_indices"]
data = DatasetGenerator.load(data_path)
manifest = load_dataset_manifest(data_path)
vehicle = VehicleConfig(**manifest["vehicle_model"])
recoverable = []
lost = []
for index in original:
    try:
        reconstruct_dataset_task(
            data["task_meta"][index], root_seed=int(manifest["seed"]), vehicle=vehicle
        )
        recoverable.append(int(index))
    except ValueError:
        lost.append(int(index))
print("原始 34 条中的可复原子集：", recoverable)
print("丢失索引：", lost)
Path("runs/validation/baseline-22-indices.json").write_text(
    json.dumps({"indices": recoverable, "lost": lost}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
