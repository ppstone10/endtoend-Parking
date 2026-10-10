"""把恢复归档按机动配比裁剪后再与原训练集合并，避免恢复集带偏机动分布。

动机（实测机制）：`build_recovery_dataset.py` 的候选按回合步长从早期开始取，
而泊车回合早期以**前进**为主，因此默认采集出的恢复集严重偏向前进。
实测 v19 原始 4 场景训练集是前进/倒车 820/810（50.3% 前进），
第一轮无配额恢复集是 198/20（**90.8% 前进**），合并后整体前进占比升到 55.1%，
闭环碰撞率随之从 15.4% 升到 43.4%。
采集端的 `--max-recoveries-per-maneuver` 只能约束全局配比，
无法保证**目标场景子集**的配比（实测 4 场景子集仍为 211/27）。

本脚本在合并端工作：对恢复样本按机动分类，把每一类裁剪到各类数量的最小值，
再把裁剪后的恢复集与原训练集合并，使最终训练集的机动比例不被恢复集带偏。
裁剪是确定性的（按 source_dataset_index + rollout_step 排序取前 N），以便复现。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/balance_recovery_dataset.py \
        --archive data/task_dataset/v19_dagger_balanced/train_with_recovery.npz \
        --output runs/validation/v24-balance/train-balanced.npz \
        --report runs/validation/v24-balance/balance-report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator

#: 参与配比平衡的机动类型；其余（若有）一律保留，避免静默丢弃未知类别。
BALANCED_MANEUVERS = ("forward", "reverse")


def _maneuver(item: dict[str, Any]) -> str:
    return str((item.get("difficulty") or {}).get("maneuver", "unknown"))


def _sort_key(item: dict[str, Any]) -> tuple[int, int]:
    recovery = item.get("recovery") or {}
    return (
        int(recovery.get("source_dataset_index", -1)),
        int(recovery.get("rollout_step", -1)),
    )


def balance(
    archive_path: Path,
    output_path: Path,
    report_path: Path | None,
    repeat_scenes: dict[str, int] | None = None,
) -> dict[str, Any]:
    data = DatasetGenerator.load(archive_path)
    metadata = list(data["task_meta"])
    if int(data.get("schema_version", -1)) != 2:
        raise ValueError("配比平衡要求 schema v2 数据集")

    recovery_indices = [i for i, item in enumerate(metadata) if item.get("recovery")]
    original_indices = [i for i, item in enumerate(metadata) if not item.get("recovery")]
    if not recovery_indices:
        raise ValueError("归档中没有恢复样本，无需平衡")

    by_maneuver: dict[str, list[int]] = collections.defaultdict(list)
    for index in recovery_indices:
        by_maneuver[_maneuver(metadata[index])].append(index)
    for indices in by_maneuver.values():
        indices.sort(key=lambda i: _sort_key(metadata[i]))

    balanced_classes = [k for k in by_maneuver if k in BALANCED_MANEUVERS]
    if len(balanced_classes) < 2:
        raise ValueError(
            f"恢复样本缺少可平衡的机动类别：{sorted(by_maneuver)}"
        )
    target = min(len(by_maneuver[k]) for k in balanced_classes)

    kept_recovery: list[int] = []
    dropped: dict[str, int] = {}
    for maneuver, indices in by_maneuver.items():
        if maneuver in BALANCED_MANEUVERS:
            kept_recovery.extend(indices[:target])
            dropped[maneuver] = max(0, len(indices) - target)
        else:
            kept_recovery.extend(indices)

    kept = sorted(original_indices + kept_recovery)
    # 场景上采样：只重复**原始**样本，避免把恢复集一起放大而破坏其机动配比。
    repeats = repeat_scenes or {}
    scene_added: dict[str, int] = {}
    if repeats:
        extra: list[int] = []
        for index in original_indices:
            scene = str(metadata[index].get("scene_name"))
            factor = repeats.get(scene, 1)
            if factor > 1:
                extra.extend([index] * (factor - 1))
                scene_added[scene] = scene_added.get(scene, 0) + (factor - 1)
        kept = sorted(kept + extra)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(2, dtype=np.uint16),
        bev_meta=np.asarray(
            DatasetGenerator._encode_metadata(data["bev_meta"], "bev_meta"),
            dtype=np.str_,
        ),
        task_meta=np.asarray(
            [
                DatasetGenerator._encode_metadata(metadata[i], "task_meta")
                for i in kept
            ],
            dtype=np.str_,
        ),
        bevs=np.asarray(data["bevs"])[kept],
        goals=np.asarray(data["goals"])[kept],
        states=np.asarray(data["states"])[kept],
        trajs=np.asarray(data["trajs"])[kept],
        masks=np.asarray(data["masks"])[kept],
        dt=np.asarray(data["dt"]),
    )

    def maneuver_counts(indices) -> dict[str, int]:
        return dict(
            sorted(collections.Counter(_maneuver(metadata[i]) for i in indices).items())
        )

    before = maneuver_counts(range(len(metadata)))
    after = maneuver_counts(kept)
    report = {
        "schema_version": 1,
        "archive": str(archive_path),
        "output": str(output_path),
        "samples_before": len(metadata),
        "samples_after": len(kept),
        "recovery_before": len(recovery_indices),
        "recovery_after": len(kept_recovery),
        "per_maneuver_target": target,
        "dropped_recovery": dropped,
        "maneuvers_before": before,
        "maneuvers_after": after,
        "recovery_maneuvers_before": maneuver_counts(recovery_indices),
        "recovery_maneuvers_after": maneuver_counts(kept_recovery),
        "forward_share_before": (
            before.get("forward", 0) / len(metadata) if metadata else 0.0
        ),
        "forward_share_after": (
            after.get("forward", 0) / len(kept) if kept else 0.0
        ),
        "scene_repeat": dict(sorted(repeats.items())),
        "scene_repeat_added": dict(sorted(scene_added.items())),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="含恢复样本的合并归档")
    parser.add_argument("--output", required=True, help="平衡后的训练归档")
    parser.add_argument("--report", default="")
    parser.add_argument("--repeat-scene", action="append", default=[], metavar="SCENE=N",
                        help="把某场景的**原始**样本重复 N 次（可重复）；用于少数场景重加权")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    repeat_scenes: dict[str, int] = {}
    for raw in args.repeat_scene:
        name, _, value = raw.partition("=")
        if not name or not value.isdigit() or int(value) < 1:
            raise SystemExit(f"--repeat-scene 需要 SCENE=N（N≥1），收到 {raw!r}")
        repeat_scenes[name] = int(value)
    report = balance(
        Path(args.archive).resolve(),
        output,
        Path(args.report).resolve() if args.report else None,
        repeat_scenes=repeat_scenes,
    )
    print(
        f"样本 {report['samples_before']} → {report['samples_after']}；"
        f"恢复 {report['recovery_before']} → {report['recovery_after']}"
        f"（每类上限 {report['per_maneuver_target']}）"
    )
    print(f"  机动（整体）{report['maneuvers_before']} → {report['maneuvers_after']}")
    print(
        f"  机动（恢复集）{report['recovery_maneuvers_before']} → "
        f"{report['recovery_maneuvers_after']}"
    )
    print(
        f"  前进占比 {report['forward_share_before']:.1%} → "
        f"{report['forward_share_after']:.1%}"
    )
    print(f"输出：{output}")


if __name__ == "__main__":
    main()
