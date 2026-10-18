"""沿用专家构建门禁，生成与已知训练身份不重叠的闭环选型集。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import DatasetGenerator, build_task_components
from experiments.closed_loop_evaluation import load_dataset_manifest
from scripts.build_dataset import _build_split_in_batches, _prepare_checkpoints, plan_fingerprint
from sim import TaskSampler, VehicleConfig
from training.reporting import atomic_write_json


def collect_metadata_ids(metadata) -> set[str]:
    identities = set()
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key in {"task_id", "source_task_id"} and isinstance(value, str) and value:
                identities.add(value)
            else:
                identities.update(collect_metadata_ids(value))
    elif isinstance(metadata, list):
        for item in metadata:
            identities.update(collect_metadata_ids(item))
    return identities


def scan_training_identities(roots: list[Path]) -> tuple[set[str], list[dict]]:
    paths = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("*.npz"):
            if ".checkpoints" not in path.parts and any(word in path.name.lower() for word in ("train", "recovery")):
                paths.add(path.resolve())
    identities, sources = set(), []
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as archive:
            if "task_meta" not in archive.files:
                sources.append({"path": str(path), "status": "no_task_metadata"})
                continue
            encoded = np.asarray(archive["task_meta"])
        metadata = [json.loads(str(value)) for value in encoded]
        found = collect_metadata_ids(metadata)
        identities.update(found)
        sources.append({
            "path": str(path), "status": "scanned", "samples": len(metadata),
            "identities": len(found),
            "metadata_sha256": hashlib.sha256("\n".join(str(value) for value in encoded).encode()).hexdigest(),
        })
    if not identities:
        raise ValueError("没有可审计的训练任务身份，不能宣称独立选型")
    return identities, sources


def sample_selection_tasks(metadata, sampler, excluded: set[str], *, max_attempts=100):
    tasks, used, next_indices = [], set(excluded), {}
    for item in metadata:
        scene = item["scene_name"]
        if scene.startswith("S9"):
            raise ValueError("选型模板不得包含最终 S9 测试场景")
        kind, difficulty = item["task_type"], item["difficulty"]
        key = (scene, kind)
        for _ in range(max_attempts):
            index = next_indices.get(key, 0)
            next_indices[key] = index + 1
            try:
                task = sampler.sample(
                    scene, kind, sample_index=index, maneuver=difficulty["maneuver"],
                    adjacent_occupancy=int(difficulty["adjacent_occupancy"]),
                    noise_level=difficulty["noise_level"],
                )
            except ValueError:
                continue
            if task.task_id not in used:
                used.add(task.task_id)
                tasks.append(task)
                break
        else:
            raise RuntimeError(f"新任务采样耗尽：{scene}/{kind}")
    return tuple(tasks)


def _strata(metadata):
    return dict(sorted(Counter(
        f"{row['scene_name']}/{row['task_type']}/{row['difficulty']['maneuver']}/"
        f"{row['difficulty']['noise_level']}/{row['difficulty']['adjacent_occupancy']}"
        for row in metadata
    ).items()))


def build_selection_dataset(template: Path, roots: list[Path], output: Path, *, seed: int, batch_size=10):
    template, output = template.resolve(), output.resolve()
    original_manifest = load_dataset_manifest(template)
    if seed == original_manifest["seed"]:
        raise ValueError("选型集必须使用不同于模板的根 seed")
    vehicle = VehicleConfig(**original_manifest["vehicle_model"])
    with np.load(template, allow_pickle=False) as archive:
        metadata = [json.loads(str(value)) for value in archive["task_meta"]]
    excluded, sources = scan_training_identities(roots)
    # 同时排除原回归模板；不是从旧验证集中挑一个子集充当新选型集。
    excluded.update(collect_metadata_ids(metadata))
    sampler = TaskSampler(seed=seed, vehicle_length=vehicle.length, vehicle_width=vehicle.width, collision_margin=vehicle.collision_margin)
    tasks = sample_selection_tasks(metadata, sampler, excluded)
    identity = {
        "schema_version": 1, "purpose": "checkpoint_selection_only", "seed": seed,
        "template": str(template), "template_metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        "excluded_sources": sources, "excluded_identity_count": len(excluded),
        "plan_fingerprint": plan_fingerprint(SimpleNamespace(train=tasks, val=(), test=())),
        "vehicle_model": vehicle.to_metadata(), "batch_size": batch_size,
    }
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("purpose") != "checkpoint_selection_only":
            raise FileExistsError("输出目录不是本工具的选型集")
    checkpoints = _prepare_checkpoints(output, identity)
    atomic_write_json(output / "exclusion-audit.json", {"sources": sources, "identity_count": len(excluded)})
    generator = DatasetGenerator(component_factory=lambda task: build_task_components(task, vehicle))
    partial, generation = _build_split_in_batches(
        split_name="selection", tasks=tasks, generator=generator, vehicle_config=vehicle,
        checkpoint_root=checkpoints, output=output, seed=seed, max_retries=100,
        batch_size=batch_size, reserved_task_ids=excluded | {task.task_id for task in tasks}, task_sampler=sampler,
    )
    with np.load(partial, allow_pickle=False) as archive:
        produced = [json.loads(str(value)) for value in archive["task_meta"]]
        max_points = int(np.asarray(archive["masks"]).sum(axis=1).max())
    ids = [row["task_id"] for row in produced]
    if len(ids) != len(tasks) or len(set(ids)) != len(ids) or set(ids) & excluded:
        raise RuntimeError("最终选型任务有重复、身份重叠或数量不匹配")
    if _strata(produced) != _strata(metadata):
        raise RuntimeError("生成后分层配额与模板不一致，拒绝冻结")
    destination = output / "selection.npz"
    partial.replace(destination)
    manifest = {
        "schema_version": 1, "purpose": "checkpoint_selection_only", "seed": seed,
        "vehicle_model": vehicle.to_metadata(), "files": {"selection": destination.name},
        "count": len(ids), "max_expert_points": max_points, "strata": _strata(produced),
        "training_identity_overlap": 0, "unique_tasks": len(set(ids)),
        "excluded_identity_count": len(excluded), "excluded_sources": sources,
        "plan_fingerprint": identity["plan_fingerprint"], "generation": generation,
        "limits": "独立任务身份；相同场景大类；仅覆盖列出的已知归档来源；非最终测试集；T5静态评测",
    }
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("count", "training_identity_overlap", "unique_tasks", "max_expert_points")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--exclude-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026101801)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    build_selection_dataset(args.template, args.exclude_root, args.output, seed=args.seed, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
