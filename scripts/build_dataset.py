"""按任务矩阵构建 train/val/test schema v2 数据集。

默认只规划任务清单时可使用 ``--dry-run``；实际构建会把中间结果写为
``*.partial.npz``，三个 split 全部成功后再改为正式文件名并写 manifest。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import (
    DatasetGenerator,
    SensorBEVPipeline,
    build_task_plan,
    generate_with_retries,
)
from interfaces import CameraIntrinsics
from planner import HybridAStarPlanner
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import MINING_TRUCK, SimulatedCamera, SimulatedLiDAR, get_noise_profile


def build_components(task):
    """按 Task 场景、噪声和 BEV 配置构造专家规划/传感器组件。"""
    profile = get_noise_profile(task.difficulty.noise_level)
    seed_sequence = np.random.SeedSequence([task.seed, 2, 8])
    lidar_seed, camera_seed = (
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(2)
    )
    intrinsics = CameraIntrinsics(
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=240.0,
        image_width=640,
        image_height=480,
    )
    lidar_range = math.hypot(
        max(task.scene.bev_config.extent[:2]),
        max(task.scene.bev_config.extent[2:]),
    )
    pipeline = SensorBEVPipeline(
        lidar_sensor=SimulatedLiDAR(
            task.scene.env,
            beams=360,
            max_range=lidar_range,
            noise=profile,
            seed=lidar_seed,
        ),
        camera_sensor=SimulatedCamera(
            task.scene.env,
            intrinsics,
            parking_area=(MINING_TRUCK.length, MINING_TRUCK.width),
            noise=profile,
            seed=camera_seed,
        ),
        lidar2bev=LiDAR2BEV(config=task.scene.bev_config),
        camera2bev=Camera2BEV(config=task.scene.bev_config),
        bev_fusion=BEVFusion(
            vehicle_length=MINING_TRUCK.length,
            vehicle_width=MINING_TRUCK.width,
        ),
    )
    planner = HybridAStarPlanner(task.scene.env, **MINING_TRUCK.planner_kwargs())
    return planner, pipeline


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


def main() -> None:
    parser = argparse.ArgumentParser(description="构建任务分层泊车数据集")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--test-scene", default="S9_mine_complex")
    parser.add_argument("--output", type=Path, default=Path("data/task_dataset"))
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = build_task_plan(
        total_count=args.count,
        seed=args.seed,
        test_scene=args.test_scene,
    )
    planned = plan_summary(plan)
    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    generator = DatasetGenerator(component_factory=build_components)
    reports = {}
    partial_paths: dict[str, Path] = {}
    reserved_task_ids = {
        task.task_id
        for split_name in ("train", "val", "test")
        for task in getattr(plan, split_name)
    }
    for split_name in ("train", "val", "test"):
        report = generate_with_retries(
            getattr(plan, split_name),
            generator=generator,
            seed=args.seed,
            max_retries=args.max_retries,
            reserved_task_ids=reserved_task_ids,
        )
        reserved_task_ids.update(task.task_id for task in report.replacements)
        partial = args.output / f"{split_name}.partial.npz"
        generator.save(list(report.samples), partial)
        partial_paths[split_name] = partial
        reports[split_name] = {
            "count": len(report.samples),
            "failure_count": report.failure_count,
            "failure_reasons": report.failure_reasons,
            "max_retries_per_task": args.max_retries,
            "replacement_count": len(report.replacements),
        }
        print(
            f"{split_name}: {len(report.samples)} 条，"
            f"规划失败重采 {report.failure_count} 次"
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
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "files": files,
        "plan": planned,
        "generation": reports,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"数据集构建完成：{args.output}")


if __name__ == "__main__":
    main()
