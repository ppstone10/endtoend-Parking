"""重建数据集某个 split 的 BEV（不改专家轨迹、不改任务身份）。

用途：传感器模型（相机视角、BEV 通道语义）变更后，用同一批任务身份重算 BEV，
得到可直接与旧数据逐格对比的"改造后"数据，用于先做小样本验证、再决定是否全量重训。

关键约束：
- **任务身份不变**：按 manifest 的 seed 与原 `task_meta` 复原同一个 Task（含
  占用、噪声档、场景几何），目标仍按候选顺序取第一个可规划者，与生成时一致；
- **专家轨迹不变**：轨迹直接沿用已存 `trajs`/`masks`，只重算 BEV，因此新旧数据
  的监督标签完全相同，任何指标差异都只能来自 BEV；
- **相机配置进入 task_meta**：把视角配置写进 `dataset.camera`，使新旧数据可区分。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/rebuild_bev_split.py \
        --data data/task_dataset/tracked_pivot_v7_3000/val.npz \
        --output runs/validation/rebuild-val/val-360.npz
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator, build_planner_and_pipeline
from dataset.generator import TrainingSample
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
)
from interfaces import GoalPose, Trajectory, VehicleState
from sim import VehicleConfig
from sim.sensor_camera import DEFAULT_VIEW_YAWS_DEG
from training.reporting import atomic_write_json


def _trajectory_from_arrays(trajs: np.ndarray, masks: np.ndarray, index: int, dt: float) -> Trajectory:
    valid = int(np.sum(masks[index] > 0.5))
    points = np.asarray(trajs[index, :valid], dtype=np.float64)
    return Trajectory(points=points, dt=dt)


def _plan_selected_goal(planner, task) -> GoalPose:
    """按候选顺序取第一个可规划目标（与生成时的目标策略一致）。"""
    candidates = (task.goal,) if task.goal is not None else task.candidate_goals
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            planner.plan(task.start, candidate.as_goal_pose())
            return candidate.as_goal_pose()
        except (RuntimeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"没有可规划目标：{last_error}")


def rebuild(
    data_path: Path,
    output_path: Path,
    *,
    target_channel: str,
    view_yaws_deg: tuple[float, ...],
    limit: int = 0,
) -> dict:
    data = DatasetGenerator.load(data_path)
    if int(data["schema_version"]) != 2:
        raise ValueError("仅支持 schema v2 数据集")
    manifest = load_dataset_manifest(data_path)
    vehicle = VehicleConfig(**manifest["vehicle_model"])
    metadata = data["task_meta"]
    states = np.asarray(data["states"])
    trajs = np.asarray(data["trajs"])
    masks = np.asarray(data["masks"])
    dt = float(np.asarray(data["dt"]).reshape(-1)[0])

    indices = list(range(len(metadata)))
    if limit > 0:
        indices = indices[:limit]

    samples: list[TrainingSample] = []
    skipped: list[dict] = []
    started = time.perf_counter()
    for ordinal, index in enumerate(indices, start=1):
        try:
            restored = reconstruct_dataset_task(
                metadata[index], root_seed=int(manifest["seed"]), vehicle=vehicle
            )
        except ValueError as exc:
            skipped.append({"index": index, "reason": f"任务无法复原：{exc}"})
            continue
        planner, pipeline = build_planner_and_pipeline(
            restored.task, vehicle, target_channel=target_channel
        )
        if hasattr(pipeline, "camera_sensor"):
            pipeline.camera_sensor.view_yaws_deg = view_yaws_deg
        try:
            goal = _plan_selected_goal(planner, restored.task)
        except RuntimeError as exc:
            skipped.append({"index": index, "reason": str(exc)})
            continue
        pipeline.set_target_goals([goal])
        state = VehicleState.from_array(states[index])
        bev = pipeline.capture_bev(state.x, state.y, state.yaw)
        task_meta = dict(metadata[index])
        dataset_meta = dict(task_meta.get("dataset") or {})
        camera_config: dict = {"target_channel": target_channel}
        if target_channel == "geometry":
            camera_config["target_source"] = "geometry_rasterization"
        else:
            camera_config.update(
                {
                    "view_yaws_deg": list(view_yaws_deg),
                    "num_views": len(view_yaws_deg),
                    "height_m": pipeline.camera_sensor.height,
                    "pitch_deg": float(np.degrees(pipeline.camera_sensor.pitch)),
                }
            )
        dataset_meta["camera"] = camera_config
        task_meta["dataset"] = dataset_meta
        samples.append(
            TrainingSample(
                bev=bev,
                goal=goal,
                state=state,
                expert_trajectory=_trajectory_from_arrays(trajs, masks, index, dt),
                task_meta=task_meta,
            )
        )
        if ordinal % 50 == 0:
            print(f"  已重建 {ordinal}/{len(indices)}", flush=True)

    if not samples:
        raise RuntimeError("没有任何样本重建成功")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DatasetGenerator(env=None).save(samples, output_path)
    report = {
        "schema_version": 1,
        "source": str(data_path),
        "output": str(output_path),
        "requested": len(indices),
        "rebuilt": len(samples),
        "skipped": skipped,
        "target_channel": target_channel,
        "view_yaws_deg": list(view_yaws_deg),
        "elapsed_sec": time.perf_counter() - started,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/task_dataset/tracked_pivot_v7_3000/val.npz"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-channel",
        choices=["geometry", "image"],
        default="geometry",
        help="target 通道来源：geometry=车位矩形几何真值（默认）；image=相机渲染+反投影（旧口径）",
    )
    parser.add_argument(
        "--views",
        default=",".join(str(value) for value in DEFAULT_VIEW_YAWS_DEG),
        help="环视视角（度，逗号分隔），仅 target-channel=image 时生效",
    )
    parser.add_argument("--limit", type=int, default=0, help=">0 时只重建前 N 条（小样本验证）")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    views = tuple(float(value) for value in args.views.split(",") if value.strip())
    report = rebuild(
        Path(args.data).resolve(),
        Path(args.output).resolve(),
        target_channel=args.target_channel,
        view_yaws_deg=views,
        limit=args.limit,
    )
    destination = (
        Path(args.report).resolve()
        if args.report
        else Path(args.output).resolve().with_suffix(".report.json")
    )
    atomic_write_json(destination, report)
    print(
        f"重建完成：请求 {report['requested']} 条，成功 {report['rebuilt']} 条，"
        f"跳过 {len(report['skipped'])} 条；耗时 {report['elapsed_sec']:.1f}s"
    )
    print(f"产物：{report['output']}；报告：{destination}")


if __name__ == "__main__":
    main()
