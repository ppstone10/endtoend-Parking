"""按任务矩阵构建 train/val/test schema v2 数据集。

默认只规划任务清单时可使用 ``--dry-run``；实际构建会把中间结果写为
固定小批次检查点；每个检查点通过双门禁后原子落盘，可用相同参数续建。
三个 split 全部成功后才合并为正式文件并写 manifest。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (
    DatasetGenerator,
    SensorBEVPipeline,
    build_task_components,
    build_task_plan,
    generate_with_retries,
    require_maneuver_consistency,
    require_trajectory_feasibility,
    summarize_dataset,
)
from sim import (
    MINING_DRILL_RIG,
    VehicleConfig,
    TaskSampler,
    load_vehicle_config,
)


def build_components(task, vehicle_config: VehicleConfig = MINING_DRILL_RIG):
    """按 Task 场景、噪声和 BEV 配置构造专家规划/传感器组件。"""
    return build_task_components(task, vehicle_config)


def plan_summary(plan) -> dict:
    """返回 dry-run 与 manifest 共用的任务计划统计。"""
    summary = {}
    for split_name in ("train", "val", "test"):
        tasks = getattr(plan, split_name)
        summary[split_name] = {
            "count": len(tasks),
            "scene_counts": dict(sorted(Counter(task.scene_name for task in tasks).items())),
            "task_type_counts": dict(
                sorted(Counter(task.task_type.value for task in tasks).items())
            ),
            "noise_level_counts": dict(
                sorted(Counter(task.difficulty.noise_level.value for task in tasks).items())
            ),
        }
    return summary


def plan_fingerprint(plan) -> str:
    """绑定任务起终位姿、难度与场景参数，防止采样逻辑变化后误续建。"""
    payload = [
        task.to_metadata()
        for split_name in ("train", "val", "test")
        for task in getattr(plan, split_name)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _strict_inspection(path: Path, vehicle_config: VehicleConfig) -> dict[str, Any]:
    summary = summarize_dataset(
        DatasetGenerator.load(path), vehicle_config=vehicle_config
    )
    require_maneuver_consistency(summary)
    require_trajectory_feasibility(summary)
    return summary


def _prepare_checkpoints(output: Path, identity: dict[str, Any]) -> Path:
    checkpoint_root = output / ".checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    identity_path = checkpoint_root / "identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(
                "现有检查点与本次 seed、任务计划、车辆模型或批大小不一致；"
                "请改用新的输出目录"
            )
    else:
        _write_json_atomic(identity_path, identity)
    return checkpoint_root


def _build_split_in_batches(
    *,
    split_name: str,
    tasks: tuple,
    generator: DatasetGenerator,
    vehicle_config: VehicleConfig,
    checkpoint_root: Path,
    output: Path,
    seed: int,
    max_retries: int,
    batch_size: int,
    reserved_task_ids: set[str],
    task_sampler: TaskSampler,
) -> tuple[Path, dict[str, Any]]:
    split_root = checkpoint_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    failure_reasons: Counter[str] = Counter()
    failure_count = 0
    replacement_count = 0
    generated_this_run = 0
    started = time.perf_counter()
    batch_count = math.ceil(len(tasks) / batch_size)

    for batch_index, offset in enumerate(range(0, len(tasks), batch_size)):
        batch = tasks[offset : offset + batch_size]
        part = split_root / f"part-{batch_index:05d}.npz"
        report_path = split_root / f"part-{batch_index:05d}.json"
        resumed = part.exists() and report_path.exists()
        if resumed:
            inspection = _strict_inspection(part, vehicle_config)
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            if int(report_data.get("count", -1)) != len(batch):
                raise RuntimeError(f"检查点样本数不匹配：{part}")
        else:
            print(
                f"{split_name} 批次 {batch_index + 1}/{batch_count} 开始："
                f"任务 {offset + 1}-{offset + len(batch)}",
                flush=True,
            )
            def report_retry(event: dict[str, Any]) -> None:
                retry = int(event["retry"])
                if retry == 1 or retry % 5 == 0:
                    print(
                        f"{split_name} 批次 {batch_index + 1}/{batch_count} "
                        f"原任务 {event['original_task_id']}："
                        f"第 {retry}/{event['max_attempts']} 次失败，"
                        f"类别 {event['failure_code']}，继续同分层重采",
                        flush=True,
                    )

            report = generate_with_retries(
                batch,
                generator=generator,
                seed=seed,
                max_retries=max_retries,
                reserved_task_ids=reserved_task_ids,
                task_sampler=task_sampler,
                progress_callback=report_retry,
            )
            reserved_task_ids.update(task.task_id for task in report.replacements)
            temporary = part.with_name(f"{part.stem}.tmp.npz")
            generator.save(list(report.samples), temporary)
            inspection = _strict_inspection(temporary, vehicle_config)
            temporary.replace(part)
            report_data = {
                "count": len(report.samples),
                "failure_count": report.failure_count,
                "failure_reasons": report.failure_reasons,
                "replacement_task_ids": [
                    task.task_id for task in report.replacements
                ],
            }
            _write_json_atomic(report_path, report_data)
            generated_this_run += len(batch)

        replacement_ids = report_data.get("replacement_task_ids", [])
        reserved_task_ids.update(str(task_id) for task_id in replacement_ids)
        failure_count += int(report_data.get("failure_count", 0))
        failure_reasons.update(report_data.get("failure_reasons", {}))
        replacement_count += len(replacement_ids)
        parts.append(part)
        elapsed = time.perf_counter() - started
        completed = offset + len(batch)
        rate = generated_this_run / elapsed if elapsed > 0.0 else 0.0
        remaining = (len(tasks) - completed) / rate if rate > 0.0 else 0.0
        action = "恢复" if resumed else "完成"
        estimate = (
            f"预计剩余 {remaining / 60.0:.1f} 分钟"
            if rate > 0.0
            else "等待新批次后估算剩余时间"
        )
        batch_failures = int(report_data.get("failure_count", 0))
        print(
            f"{split_name} 批次 {batch_index + 1}/{batch_count} {action}："
            f"{completed}/{len(tasks)}，本批失败重采 {batch_failures}，"
            f"累计 {failure_count}，{estimate}",
            flush=True,
        )

    partial = output / f"{split_name}.partial.npz"
    DatasetGenerator.merge_archives(parts, partial)
    inspection = _strict_inspection(partial, vehicle_config)
    report = {
        "count": len(tasks),
        "failure_count": failure_count,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "max_retries_per_task": max_retries,
        "replacement_count": replacement_count,
        "maneuver_consistency": inspection["maneuver_consistency"],
        "trajectory_feasibility": inspection["trajectory_feasibility"],
    }
    return partial, report


def main() -> None:
    parser = argparse.ArgumentParser(description="构建任务分层泊车数据集")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--test-scene", default="S9_mine_complex")
    parser.add_argument("--output", type=Path, default=Path("data/task_dataset"))
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="每个可恢复检查点包含的任务数（默认 10）",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--vehicle-config",
        type=Path,
        default=None,
        help="履带钻机 JSON 配置；默认 configs/vehicles/tracked_drill_rig.json",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正")
    vehicle_config = (
        MINING_DRILL_RIG
        if args.vehicle_config is None
        else load_vehicle_config(args.vehicle_config)
    )

    plan = build_task_plan(
        total_count=args.count,
        seed=args.seed,
        test_scene=args.test_scene,
        vehicle_length=vehicle_config.length,
        vehicle_width=vehicle_config.width,
        collision_margin=vehicle_config.collision_margin,
    )
    planned = plan_summary(plan)
    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    generator = DatasetGenerator(
        component_factory=lambda task: build_components(task, vehicle_config)
    )
    identity = {
        "schema_version": 1,
        "seed": args.seed,
        "test_scene": args.test_scene,
        "requested_count": args.count,
        "max_retries": args.max_retries,
        "batch_size": args.batch_size,
        "vehicle_model": vehicle_config.to_metadata(),
        "plan": planned,
        "plan_fingerprint": plan_fingerprint(plan),
    }
    checkpoint_root = _prepare_checkpoints(args.output, identity)
    reports = {}
    partial_paths: dict[str, Path] = {}
    reserved_task_ids = {
        task.task_id
        for split_name in ("train", "val", "test")
        for task in getattr(plan, split_name)
    }
    task_sampler = TaskSampler(
        seed=args.seed,
        vehicle_length=vehicle_config.length,
        vehicle_width=vehicle_config.width,
        collision_margin=vehicle_config.collision_margin,
    )
    for split_name in ("train", "val", "test"):
        partial, report = _build_split_in_batches(
            split_name=split_name,
            tasks=tuple(getattr(plan, split_name)),
            generator=generator,
            vehicle_config=vehicle_config,
            checkpoint_root=checkpoint_root,
            output=args.output,
            seed=args.seed,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
            reserved_task_ids=reserved_task_ids,
            task_sampler=task_sampler,
        )
        partial_paths[split_name] = partial
        reports[split_name] = report
        print(
            f"{split_name}: {report['count']} 条，"
            f"规划失败重采 {report['failure_count']} 次"
        )

    files = {}
    for split_name, partial in partial_paths.items():
        final_path = args.output / f"{split_name}.npz"
        partial.replace(final_path)
        files[split_name] = final_path.name
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "test_scene": args.test_scene,
        "requested_count": args.count,
        "max_retries": args.max_retries,
        "batch_size": args.batch_size,
        "vehicle_model": vehicle_config.to_metadata(),
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "files": files,
        "plan": planned,
        "plan_fingerprint": plan_fingerprint(plan),
        "generation": reports,
    }
    _write_json_atomic(args.output / "manifest.json", manifest)
    print(f"数据集构建完成：{args.output}")


if __name__ == "__main__":
    main()
