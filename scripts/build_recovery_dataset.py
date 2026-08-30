"""可续建地采集学习器闭环偏离状态并生成专家恢复训练集。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller import MPCController
from dataset import (
    DatasetGenerator,
    build_recovery_sample,
    build_task_components,
    select_recovery_candidates,
)
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
    select_evaluation_indices,
)
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import ClosedLoopEngine, NetworkSource, TerminalChecker
from sim import DifferentialDriveModel, VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.data import validate_model_dataset
from training.reporting import atomic_write_json


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(args, data_path: Path, checkpoint_path: Path, manifest: dict) -> dict:
    stat = data_path.stat()
    return {
        "schema_version": 1,
        "data": str(data_path),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "plan_fingerprint": manifest.get("plan_fingerprint"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_digest(checkpoint_path),
        "samples": args.samples,
        "selection": args.selection,
        "rollout_steps": args.rollout_steps,
        "replan_every": args.replan_every,
        "state_stride": args.state_stride,
        "min_deviation": args.min_deviation,
        "min_yaw_deviation_deg": args.min_yaw_deviation_deg,
        "max_recoveries_per_task": args.max_recoveries_per_task,
        "seed": args.seed,
        "vehicle_model": manifest.get("vehicle_model"),
    }


def _prepare_checkpoint_dir(output: Path, identity: dict[str, Any]) -> Path:
    checkpoint_dir = output / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity_path = checkpoint_dir / "identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(
                "恢复数据检查点身份与本次参数不一致；请更换 --output，勿混用旧检查点"
            )
    else:
        atomic_write_json(identity_path, identity)
    return checkpoint_dir


def _save_part(samples, generator: DatasetGenerator, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    generator.save(samples, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="原始 train schema v2 NPZ")
    parser.add_argument("--model", required=True, help="当前 deployment checkpoint")
    parser.add_argument("--output", required=True, help="恢复数据输出目录")
    parser.add_argument("--samples", type=int, default=240, help="源任务数；<=0 为全部")
    parser.add_argument("--selection", choices=["stratified", "head"], default="stratified")
    parser.add_argument("--rollout-steps", type=int, default=120)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--state-stride", type=int, default=10)
    parser.add_argument("--min-deviation", type=float, default=0.25)
    parser.add_argument("--min-yaw-deviation-deg", type=float, default=5.0)
    parser.add_argument("--max-recoveries-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (
            args.rollout_steps,
            args.replan_every,
            args.state_stride,
            args.max_recoveries_per_task,
        )
    ):
        parser.error("rollout/replan/stride/max-recoveries 参数必须为正")
    if (
        not np.isfinite(args.min_deviation)
        or args.min_deviation < 0.0
        or not np.isfinite(args.min_yaw_deviation_deg)
        or args.min_yaw_deviation_deg < 0.0
    ):
        parser.error("位置/航向偏离阈值必须为有限非负数")

    data_path = Path(args.data).resolve()
    checkpoint_path = Path(args.model).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = DatasetGenerator.load(data_path)
    metadata = data.get("task_meta")
    if int(data.get("schema_version", -1)) != 2 or not isinstance(metadata, list):
        raise ValueError("恢复采集要求 schema v2 数据集与 task_meta")
    manifest = load_dataset_manifest(data_path)
    vehicle = VehicleConfig(**manifest["vehicle_model"])
    loaded = load_model_checkpoint(checkpoint_path)
    validate_model_dataset(loaded.model, data)
    indices = select_evaluation_indices(
        metadata, samples=args.samples, strategy=args.selection
    )
    identity = _identity(args, data_path, checkpoint_path, manifest)
    checkpoint_dir = _prepare_checkpoint_dir(output, identity)
    generator = DatasetGenerator()
    stats = {"source_completed": 0, "recovery_samples": 0, "planner_failures": 0}
    failure_reasons: Counter[str] = Counter()

    for ordinal, index in enumerate(indices, start=1):
        part_path = checkpoint_dir / f"part-{index:05d}.npz"
        done_path = checkpoint_dir / f"part-{index:05d}.done.json"
        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            stats["source_completed"] += 1
            stats["recovery_samples"] += int(done["recovery_samples"])
            stats["planner_failures"] += int(done["planner_failures"])
            failure_reasons.update(done.get("planner_failure_reasons", {}))
            print(f"[{ordinal}/{len(indices)}] index={index} 已完成，跳过", flush=True)
            continue

        restored = reconstruct_dataset_task(
            metadata[index], root_seed=int(manifest["seed"]), vehicle=vehicle
        )
        planner, pipeline = build_task_components(restored.task, vehicle)
        source = NetworkSource(pipeline, loaded.model)
        actual_collision_checker = RectangleFootprintCollisionChecker(
            restored.task.scene.env,
            vehicle_length=vehicle.length,
            vehicle_width=vehicle.width,
            collision_margin=0.0,
            resolution=vehicle.collision_check_resolution,
        )
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
            mpc=MPCController(
                dt=0.1,
                horizon=10,
                seed=args.seed + index,
                **vehicle.mpc_kwargs(),
            ),
            source=source,
            terminal=TerminalChecker(restored.tol_pos, restored.tol_yaw),
            env=restored.task.scene.env,
            collision_checker=actual_collision_checker,
            replan_every=args.replan_every,
            max_steps=args.rollout_steps,
            **vehicle.collision_kwargs(),
        )
        start = VehicleState.from_array(np.asarray(data["states"])[index])
        result = engine.run(start, restored.goal)
        valid_length = int(np.asarray(data["masks"])[index].sum())
        candidates = select_recovery_candidates(
            result.record.states,
            result.record.collisions,
            np.asarray(data["trajs"])[index, :valid_length],
            stride=args.state_stride,
            min_deviation_m=args.min_deviation,
            min_yaw_deviation_rad=np.deg2rad(args.min_yaw_deviation_deg),
            yaw_radius_m=float(
                np.hypot(
                    vehicle.length / 2.0 + vehicle.collision_margin,
                    vehicle.width / 2.0 + vehicle.collision_margin,
                )
            ),
        )
        recovered = []
        planner_failures = 0
        part_failure_reasons: Counter[str] = Counter()
        for candidate in candidates:
            if len(recovered) >= args.max_recoveries_per_task:
                break
            if not planner._collision_checker.pose_free(
                candidate.state.x, candidate.state.y, candidate.state.yaw
            ):
                continue
            try:
                recovered.append(
                    build_recovery_sample(
                        candidate,
                        source_index=index,
                        source_metadata=metadata[index],
                        goal=restored.goal,
                        planner=planner,
                        pipeline=pipeline,
                        checkpoint_identity=identity["checkpoint_sha256"],
                    )
                )
            except (RuntimeError, ValueError) as exc:
                planner_failures += 1
                part_failure_reasons[str(exc)] += 1
        if recovered:
            _save_part(recovered, generator, part_path)
        atomic_write_json(
            done_path,
            {
                "source_index": index,
                "task_id": restored.task.task_id,
                "candidate_count": len(candidates),
                "recovery_samples": len(recovered),
                "planner_failures": planner_failures,
                "planner_failure_reasons": dict(sorted(part_failure_reasons.items())),
                "rollout_failure": result.failure,
            },
        )
        stats["source_completed"] += 1
        stats["recovery_samples"] += len(recovered)
        stats["planner_failures"] += planner_failures
        failure_reasons.update(part_failure_reasons)
        print(
            f"[{ordinal}/{len(indices)}] {restored.task.task_id} "
            f"候选={len(candidates)} 恢复={len(recovered)} 规划失败={planner_failures}",
            flush=True,
        )

    parts = sorted(checkpoint_dir.glob("part-*.npz"))
    if not parts:
        raise RuntimeError("没有生成恢复样本；请检查偏离阈值或当前策略闭环记录")
    recovery_path = output / "recovery.npz"
    DatasetGenerator.merge_archives(parts, recovery_path)
    recovery_data = DatasetGenerator.load(recovery_path)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in recovery_data["task_meta"]:
        group_counts["scene_name"][str(item.get("scene_name", "unknown"))] += 1
        group_counts["task_type"][str(item.get("task_type", "unknown"))] += 1
        group_counts["maneuver"][str(item.get("difficulty", {}).get("maneuver", "unknown"))] += 1
        group_counts["trigger"][str(item.get("recovery", {}).get("trigger", "unknown"))] += 1
    combined_path = output / "train_with_recovery.npz"
    DatasetGenerator.merge_archives([data_path, recovery_path], combined_path)
    report = {
        "schema_version": 1,
        "status": "completed",
        "identity": identity,
        "selected_indices": indices,
        **stats,
        "planner_failure_reasons": dict(sorted(failure_reasons.items())),
        "recovery_groups": {
            dimension: dict(sorted(counts.items()))
            for dimension, counts in sorted(group_counts.items())
        },
        "recovery_archive": str(recovery_path),
        "combined_train_archive": str(combined_path),
    }
    atomic_write_json(output / "report.json", report)
    print(
        f"完成：{stats['recovery_samples']} 条恢复样本；合并训练集 {combined_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
