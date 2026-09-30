"""BEV 重建一致性核验：确认重建只改了 BEV，监督标签与任务身份逐项不变。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/verify_bev_rebuild.py \
        --old data/task_dataset/tracked_pivot_v7_3000/val.npz \
        --new runs/validation/rebuild-val/val-360.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator


def _align(old: dict, new: dict) -> tuple[list[int], list[int]]:
    """按 task_id 对齐两个数据集，返回 (old_idx, new_idx) 对。"""
    old_ids = [str(meta.get("task_id")) for meta in old["task_meta"]]
    new_ids = [str(meta.get("task_id")) for meta in new["task_meta"]]
    new_lookup = {task_id: index for index, task_id in enumerate(new_ids)}
    pairs = [
        (index, new_lookup[task_id])
        for index, task_id in enumerate(old_ids)
        if task_id in new_lookup
    ]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def verify(old_path: Path, new_path: Path) -> dict:
    old = DatasetGenerator.load(old_path)
    new = DatasetGenerator.load(new_path)
    old_idx, new_idx = _align(old, new)
    if not old_idx:
        raise ValueError("两个数据集没有共同的 task_id")

    old_bevs = np.asarray(old["bevs"])
    new_bevs = np.asarray(new["bevs"])
    channels = [str(c) for c in np.asarray(old["bev_meta"]["channels"])]
    if channels != [str(c) for c in np.asarray(new["bev_meta"]["channels"])]:
        raise ValueError("BEV 通道定义发生变化，无法逐通道对比")
    if list(old_bevs.shape[1:]) != list(new_bevs.shape[1:]):
        raise ValueError("BEV 形状发生变化")

    goals_equal = bool(
        np.allclose(
            np.asarray(old["goals"])[old_idx], np.asarray(new["goals"])[new_idx]
        )
    )
    states_equal = bool(
        np.allclose(
            np.asarray(old["states"])[old_idx], np.asarray(new["states"])[new_idx]
        )
    )
    masks_equal = bool(
        np.array_equal(
            np.asarray(old["masks"])[old_idx], np.asarray(new["masks"])[new_idx]
        )
    )
    # 轨迹按 mask 有效前缀比较（新数据 horizon 可能不同，补零区不参与）。
    traj_equal = True
    old_trajs = np.asarray(old["trajs"])
    new_trajs = np.asarray(new["trajs"])
    old_masks = np.asarray(old["masks"])
    new_masks = np.asarray(new["masks"])
    for old_i, new_i in zip(old_idx, new_idx):
        old_len = int(np.sum(old_masks[old_i] > 0.5))
        new_len = int(np.sum(new_masks[new_i] > 0.5))
        if old_len != new_len or not np.allclose(
            old_trajs[old_i, :old_len], new_trajs[new_i, :new_len], atol=1e-6
        ):
            traj_equal = False
            break

    per_channel_changed: dict[str, int] = {}
    per_channel_diff_cells: dict[str, float] = {}
    target_index = channels.index("target") if "target" in channels else None
    target_empty_old = 0
    target_empty_new = 0
    remaining_empty_ids: list[str] = []
    for old_i, new_i in zip(old_idx, new_idx):
        for channel_index, name in enumerate(channels):
            changed = not np.array_equal(
                old_bevs[old_i, channel_index], new_bevs[new_i, channel_index]
            )
            per_channel_changed[name] = per_channel_changed.get(name, 0) + int(changed)
            if changed:
                diff = np.abs(
                    old_bevs[old_i, channel_index].astype(np.float64)
                    - new_bevs[new_i, channel_index].astype(np.float64)
                )
                per_channel_diff_cells[name] = (
                    per_channel_diff_cells.get(name, 0.0) + float(np.count_nonzero(diff))
                )
        if target_index is not None:
            if not np.any(old_bevs[old_i, target_index]):
                target_empty_old += 1
            if not np.any(new_bevs[new_i, target_index]):
                target_empty_new += 1
                remaining_empty_ids.append(str(old["task_meta"][old_i].get("task_id")))

    return {
        "schema_version": 1,
        "old": str(old_path),
        "new": str(new_path),
        "aligned_samples": len(old_idx),
        "old_only": len(old["task_meta"]) - len(old_idx),
        "new_only": len(new["task_meta"]) - len(new_idx),
        "labels_unchanged": {
            "goals_equal": goals_equal,
            "states_equal": states_equal,
            "masks_equal": masks_equal,
            "trajectories_equal": traj_equal,
        },
        "bev_changed_samples_per_channel": per_channel_changed,
        "bev_changed_cells_per_channel": {
            key: int(value) for key, value in per_channel_diff_cells.items()
        },
        "target_empty": {
            "old": target_empty_old,
            "new": target_empty_new,
            "old_rate": target_empty_old / len(old_idx),
            "new_rate": target_empty_new / len(old_idx),
        },
        "remaining_empty_task_ids": remaining_empty_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", default="data/task_dataset/tracked_pivot_v7_3000/val.npz")
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = verify(Path(args.old).resolve(), Path(args.new).resolve())
    labels = report["labels_unchanged"]
    target = report["target_empty"]
    print(f"对齐样本 {report['aligned_samples']} 条（旧独有 {report['old_only']}，新独有 {report['new_only']}）")
    print(
        "监督标签不变："
        f"goal={labels['goals_equal']} state={labels['states_equal']} "
        f"mask={labels['masks_equal']} traj={labels['trajectories_equal']}"
    )
    print(
        f"target 空帧：旧 {target['old']}（{target['old_rate']:.1%}）"
        f" → 新 {target['new']}（{target['new_rate']:.1%}）"
    )
    changed = report["bev_changed_samples_per_channel"]
    for name, count in sorted(changed.items()):
        print(f"  通道 {name}: {count} 条样本发生变化")
    if args.output:
        import json

        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
