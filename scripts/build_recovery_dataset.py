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
    select_recovery_candidates_with_diagnostics,
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


def _directory_evidence_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.glob("part-*.done.json"))
    if not files:
        raise ValueError(f"优先任务目录没有完成检查点：{path}")
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _resolve_checkpoint_dir(path: Path) -> Path:
    candidate = path / ".checkpoints" if (path / ".checkpoints").is_dir() else path
    if not candidate.is_dir():
        raise ValueError(f"找不到恢复检查点目录：{path}")
    return candidate


def _load_priority_indices(path: Path) -> tuple[list[int], dict[str, Any]]:
    checkpoint_dir = _resolve_checkpoint_dir(path)
    identity_path = checkpoint_dir / "identity.json"
    if not identity_path.exists():
        raise ValueError(f"优先任务目录缺少身份文件：{identity_path}")
    source_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    selected: list[int] = []
    outcome_counts: Counter[str] = Counter()
    for done_path in sorted(checkpoint_dir.glob("part-*.done.json")):
        done = json.loads(done_path.read_text(encoding="utf-8"))
        failure = done.get("rollout_failure")
        if failure in {"collision", "timeout"} and int(done["recovery_samples"]) == 0:
            selected.append(int(done["source_index"]))
            outcome_counts[str(failure)] += 1
    if not selected:
        raise ValueError("上一轮检查点中没有碰撞/超时且零恢复的困难源任务")
    if len(selected) != len(set(selected)):
        raise ValueError("优先任务检查点包含重复 source_index")
    return sorted(selected), {
        "rule": "rollout_failure_in_collision_timeout_and_zero_recovery",
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "evidence_sha256": _directory_evidence_digest(checkpoint_dir),
        "source_plan_fingerprint": source_identity.get("plan_fingerprint"),
        "source_checkpoint_sha256": source_identity.get("checkpoint_sha256"),
        "outcome_counts": dict(sorted(outcome_counts.items())),
    }


def _identity(args, data_path: Path, checkpoint_path: Path, manifest: dict) -> dict:
    stat = data_path.stat()
    return {
        "schema_version": 2,
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
        "priority_evidence": args.priority_evidence,
        "base_recovery": args.base_recovery_identity,
    }


def _recovery_keys(metadata: list[dict[str, Any]]) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    for item in metadata:
        recovery = item.get("recovery")
        if not isinstance(recovery, dict):
            raise ValueError("恢复归档包含缺失 recovery 元数据的样本")
        keys.append(
            (
                int(recovery["source_dataset_index"]),
                int(recovery["rollout_step"]),
            )
        )
    return keys


def _assert_unique_recovery_samples(metadata: list[dict[str, Any]]) -> None:
    keys = _recovery_keys(metadata)
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        preview = ", ".join(f"{index}@{step}" for index, step in duplicates[:5])
        raise ValueError(f"恢复归档存在重复来源状态：{preview}")


def _validate_recovery_provenance(
    recovery_metadata: list[dict[str, Any]],
    source_metadata: list[dict[str, Any]],
) -> None:
    for item in recovery_metadata:
        recovery = item.get("recovery", {})
        source_index = int(recovery.get("source_dataset_index", -1))
        if source_index < 0 or source_index >= len(source_metadata):
            raise ValueError("恢复归档的来源索引超出当前原始训练集")
        expected_task_id = str(source_metadata[source_index].get("task_id", ""))
        if str(recovery.get("source_task_id", "")) != expected_task_id:
            raise ValueError("恢复归档的来源任务身份与当前原始训练集不一致")


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
    parser.add_argument(
        "--priority-from",
        help="从上一轮恢复输出/检查点自动补采碰撞或超时且零恢复的任务",
    )
    parser.add_argument(
        "--base-recovery",
        help="保留并合并的既有 recovery.npz；与新样本重复时拒绝输出",
    )
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
    args.priority_evidence = None
    if args.priority_from:
        indices, args.priority_evidence = _load_priority_indices(
            Path(args.priority_from).resolve()
        )
        if (
            args.priority_evidence["source_plan_fingerprint"]
            != manifest.get("plan_fingerprint")
            or args.priority_evidence["source_checkpoint_sha256"]
            != _checkpoint_digest(checkpoint_path)
        ):
            raise ValueError("优先任务证据与当前数据计划或策略 checkpoint 不一致")
    else:
        indices = select_evaluation_indices(
            metadata, samples=args.samples, strategy=args.selection
        )
    if any(index < 0 or index >= len(metadata) for index in indices):
        raise ValueError("优先任务检查点与当前训练集索引不兼容")
    args.base_recovery_identity = None
    base_recovery_path: Path | None = None
    if args.base_recovery:
        base_recovery_path = Path(args.base_recovery).resolve()
        base_data = DatasetGenerator.load(base_recovery_path)
        base_metadata = base_data.get("task_meta")
        if int(base_data.get("schema_version", -1)) != 2 or not isinstance(
            base_metadata, list
        ):
            raise ValueError("--base-recovery 必须是 schema v2 恢复归档")
        _assert_unique_recovery_samples(base_metadata)
        _validate_recovery_provenance(base_metadata, metadata)
        args.base_recovery_identity = {
            "path": str(base_recovery_path),
            "sha256": _checkpoint_digest(base_recovery_path),
            "samples": len(base_metadata),
        }
    identity = _identity(args, data_path, checkpoint_path, manifest)
    checkpoint_dir = _prepare_checkpoint_dir(output, identity)
    generator = DatasetGenerator()
    stats = {"source_completed": 0, "recovery_samples": 0, "planner_failures": 0}
    failure_reasons: Counter[str] = Counter()
    selection_totals: Counter[str] = Counter()
    recovered_trigger_totals: Counter[str] = Counter()

    for ordinal, index in enumerate(indices, start=1):
        part_path = checkpoint_dir / f"part-{index:05d}.npz"
        done_path = checkpoint_dir / f"part-{index:05d}.done.json"
        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            stats["source_completed"] += 1
            stats["recovery_samples"] += int(done["recovery_samples"])
            stats["planner_failures"] += int(done["planner_failures"])
            failure_reasons.update(done.get("planner_failure_reasons", {}))
            selection_totals.update(done.get("selection_diagnostics", {}))
            recovered_trigger_totals.update(done.get("recovered_triggers", {}))
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
        selection = select_recovery_candidates_with_diagnostics(
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
            pose_free=planner._collision_checker.pose_free,
            initial_state=start,
        )
        candidates = selection.candidates
        recovered = []
        planner_failures = 0
        part_failure_reasons: Counter[str] = Counter()
        for candidate in candidates:
            if len(recovered) >= args.max_recoveries_per_task:
                break
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
                "selection_diagnostics": selection.diagnostics,
                "recovered_triggers": dict(
                    sorted(Counter(item.task_meta["recovery"]["trigger"] for item in recovered).items())
                ),
            },
        )
        stats["source_completed"] += 1
        stats["recovery_samples"] += len(recovered)
        stats["planner_failures"] += planner_failures
        failure_reasons.update(part_failure_reasons)
        selection_totals.update(selection.diagnostics)
        recovered_trigger_totals.update(
            item.task_meta["recovery"]["trigger"] for item in recovered
        )
        print(
            f"[{ordinal}/{len(indices)}] {restored.task.task_id} "
            f"候选={len(candidates)} 恢复={len(recovered)} 规划失败={planner_failures}",
            flush=True,
        )

    parts = sorted(checkpoint_dir.glob("part-*.npz"))
    recovery_inputs = ([base_recovery_path] if base_recovery_path else []) + parts
    if not recovery_inputs:
        raise RuntimeError("没有生成恢复样本；请检查偏离阈值或当前策略闭环记录")
    recovery_path = output / "recovery.npz"
    DatasetGenerator.merge_archives(recovery_inputs, recovery_path)
    recovery_data = DatasetGenerator.load(recovery_path)
    _assert_unique_recovery_samples(recovery_data["task_meta"])
    _validate_recovery_provenance(recovery_data["task_meta"], metadata)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in recovery_data["task_meta"]:
        group_counts["scene_name"][str(item.get("scene_name", "unknown"))] += 1
        group_counts["task_type"][str(item.get("task_type", "unknown"))] += 1
        group_counts["maneuver"][str(item.get("difficulty", {}).get("maneuver", "unknown"))] += 1
        group_counts["trigger"][str(item.get("recovery", {}).get("trigger", "unknown"))] += 1
    combined_path = output / "train_with_recovery.npz"
    DatasetGenerator.merge_archives([data_path, recovery_path], combined_path)
    collision_events = int(selection_totals["collision_events"])
    def _coverage(key: str) -> dict[str, float | int]:
        covered = int(selection_totals[key])
        return {
            "covered_events": covered,
            "total_events": collision_events,
            "coverage_rate": covered / collision_events if collision_events else 0.0,
        }
    report = {
        "schema_version": 2,
        "status": "completed",
        "identity": identity,
        "selected_indices": indices,
        "selection_source": args.priority_evidence or {
            "rule": args.selection,
            "samples": args.samples,
        },
        **stats,
        "base_recovery_samples": int(
            args.base_recovery_identity["samples"] if args.base_recovery_identity else 0
        ),
        "total_recovery_samples": len(recovery_data["task_meta"]),
        "planner_failure_reasons": dict(sorted(failure_reasons.items())),
        "selection_diagnostics": dict(sorted(selection_totals.items())),
        "recovered_triggers": dict(sorted(recovered_trigger_totals.items())),
        "collision_selection_ablation": {
            "fixed_stride": _coverage("stride_collision_events_covered"),
            "immediate_pre_collision": _coverage(
                "immediate_pre_collision_margin_safe_events"
            ),
            "last_margin_safe_backtrack": _coverage(
                "last_margin_safe_backtrack_events"
            ),
        },
        "collision_backtrack_labeling": {
            "selected_candidates": int(
                selection_totals["last_margin_safe_backtrack_events"]
            ),
            "expert_labels_generated": int(
                recovered_trigger_totals["collision_backtrack"]
            ),
            "success_rate": (
                recovered_trigger_totals["collision_backtrack"]
                / selection_totals["last_margin_safe_backtrack_events"]
                if selection_totals["last_margin_safe_backtrack_events"]
                else 0.0
            ),
        },
        "recovery_groups": {
            dimension: dict(sorted(counts.items()))
            for dimension, counts in sorted(group_counts.items())
        },
        "recovery_archive": str(recovery_path),
        "combined_train_archive": str(combined_path),
    }
    atomic_write_json(output / "report.json", report)
    print(
        f"完成：新增 {stats['recovery_samples']} 条，恢复集共 "
        f"{len(recovery_data['task_meta'])} 条；合并训练集 {combined_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
