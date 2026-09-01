"""L1 感知→BEV 保真度评测：按场景×噪声档输出退化报告。

对每个场景在合法起点区内采样若干车辆位姿（朝向目标车位方向，模拟真实泊车
起点朝向），对每个位姿构建带目标车位的传感器管道并采集融合 BEV，与两类
真值比较：

- occupancy：高分辨率 LiDAR 采样真值（主口径）与场景几何真值（参考）；
- target：目标车位矩形真值栅格。

输出 IoU/Precision/Recall/命中率与退化曲线。

运行：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/evaluate_bev_fidelity.py \\
        --output runs/bev-fidelity/L1 \\
        --poses-per-scene 6
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset.components import build_task_components
from interfaces import GoalPose
from metrics.bev_fidelity import (
    aggregate_bev_fidelity,
    compute_bev_fidelity_metrics,
    rasterize_ground_truth_occupancy,
    rasterize_ground_truth_target,
    rasterize_lidar_truth_occupancy,
)
from sim import MINING_DRILL_RIG, NoiseLevel, VehicleConfig
from sim.scenes import SCENE_REGISTRY, build_scene
from training.reporting import atomic_write_json
from viz.bev_fidelity import save_bev_fidelity_degradation


def _sample_vehicle_pose(scene, vehicle: VehicleConfig, goal: GoalPose, rng: np.random.Generator):
    """在场景 spawn_zones 内采样一个朝向目标、四角无碰撞的车辆位姿。"""
    env = scene.env
    zones = scene.spawn_zones or [(-env.world_size / 2 + 2, env.world_size / 2 - 2,
                                    -env.world_size / 2 + 2, env.world_size / 2 - 2)]
    half_l = vehicle.length / 2.0
    half_w = vehicle.width / 2.0
    for _ in range(400):
        (x_min, x_max, y_min, y_max) = zones[int(rng.integers(0, len(zones)))]
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        dist = math.hypot(goal.x - x, goal.y - y)
        if dist < 3.0 or dist > 20.0:
            continue
        # 车头朝向目标（局部前向 +x 指向目标方向）。
        yaw = float(np.arctan2(goal.y - y, goal.x - x))
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        corners = [
            (x + half_l * cos_yaw - half_w * sin_yaw, y + half_l * sin_yaw + half_w * cos_yaw),
            (x + half_l * cos_yaw + half_w * sin_yaw, y + half_l * sin_yaw - half_w * cos_yaw),
            (x - half_l * cos_yaw - half_w * sin_yaw, y - half_l * sin_yaw + half_w * cos_yaw),
            (x - half_l * cos_yaw + half_w * sin_yaw, y - half_l * sin_yaw - half_w * cos_yaw),
        ]
        if all(env.is_free(cx, cy) for cx, cy in corners):
            return x, y, yaw
    raise RuntimeError(f"场景 {scene.name} 在 spawn_zones 内采样合法车辆位姿失败")


def _build_pipeline_for_noise(task, vehicle: VehicleConfig, noise: NoiseLevel):
    """复用任务组件工厂，但显式覆盖噪声档。"""
    original = task.difficulty.noise_level
    task.difficulty.noise_level = noise
    try:
        planner, pipeline = build_task_components(task, vehicle)
        return planner, pipeline
    finally:
        task.difficulty.noise_level = original


def evaluate_scene_noise(
    task,
    vehicle: VehicleConfig,
    noise: NoiseLevel,
    poses: Iterable[tuple[float, float, float]],
    goal: GoalPose,
) -> dict:
    """对单个场景×噪声档的多个位姿计算保真度并聚合。"""
    planner, pipeline = _build_pipeline_for_noise(task, vehicle, noise)
    bev_config = pipeline.bev_config
    metrics_list = []
    geometry_list = []
    for x, y, yaw in poses:
        pipeline.set_target_goals([goal])
        bev = pipeline.capture_bev(x, y, yaw)
        lidar_truth = rasterize_lidar_truth_occupancy(
            task.scene.env, x, y, yaw, bev_config
        )
        geometry_truth = rasterize_ground_truth_occupancy(
            task.scene.env, x, y, yaw, bev_config
        )
        truth_target = rasterize_ground_truth_target(
            goal, vehicle.length, vehicle.width, x, y, yaw, bev_config
        )
        metrics_list.append(
            compute_bev_fidelity_metrics(bev, lidar_truth, truth_target)
        )
        geometry_list.append(
            compute_bev_fidelity_metrics(bev, geometry_truth, truth_target)
        )
    return {
        "lidar_truth": aggregate_bev_fidelity(metrics_list).to_dict(),
        "geometry_truth": aggregate_bev_fidelity(geometry_list).to_dict(),
    }


def run_bev_fidelity(
    scenes: Iterable[str],
    *,
    output_dir: str | Path,
    poses_per_scene: int = 6,
    noise_levels: Iterable[NoiseLevel | str] = (NoiseLevel.CLEAN, NoiseLevel.LOW, NoiseLevel.HIGH),
    seed: int = 0,
    vehicle: VehicleConfig = MINING_DRILL_RIG,
) -> dict:
    """遍历场景×噪声档，输出整体/分组指标与退化曲线。"""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scene_names = list(scenes)
    if not scene_names:
        raise ValueError("至少需要一个场景")

    overall_buckets: dict[str, list[dict]] = {}
    per_scene_report: dict[str, dict[str, dict]] = {}
    per_scene: dict[str, dict[str, dict]] = {}
    for scene_name in scene_names:
        bundle = build_scene(scene_name, seed=seed)
        task = _mock_task(bundle)
        free_spots = [spot for spot in bundle.spots if not spot.occupied]
        if not free_spots:
            raise RuntimeError(f"场景 {scene_name} 没有空闲车位可作 target")
        goal = free_spots[0].pose
        poses = [
            _sample_vehicle_pose(bundle, vehicle, goal, rng) for _ in range(poses_per_scene)
        ]
        per_scene_report[scene_name] = {}
        per_scene[scene_name] = {}
        for noise in noise_levels:
            level = NoiseLevel(noise)
            result = evaluate_scene_noise(task, vehicle, level, poses, goal)
            lidar = result["lidar_truth"]
            per_scene_report[scene_name][level.value] = result
            per_scene[scene_name][level.value] = {
                "occupancy_iou": lidar["occupancy_iou"],
                "target_hit_rate": lidar["target_hit_rate"],
            }
            overall_buckets.setdefault(level.value, []).append(lidar)
            print(
                f"{scene_name} {level.value}: "
                f"occ IoU={lidar['occupancy_iou']:.3f} "
                f"P={lidar['occupancy_precision']:.3f} "
                f"R={lidar['occupancy_recall']:.3f} "
                f"target={lidar['target_hit_rate']:.3f}",
                flush=True,
            )

    overall = {level: _agg_from_dicts(bucket) for level, bucket in overall_buckets.items()}
    plot_paths = save_bev_fidelity_degradation(per_scene, destination / "bev_fidelity_degradation")
    report = {
        "schema_version": 1,
        "task": "L1_bev_fidelity",
        "seed": seed,
        "poses_per_scene": poses_per_scene,
        "vehicle_model": vehicle.to_metadata(),
        "overall": overall,
        "per_scene": per_scene_report,
        "artifacts": {"plot_png": plot_paths[0], "plot_pdf": plot_paths[1]},
    }
    atomic_write_json(destination / "report.json", report)
    return report


def _agg_from_dicts(items: list[dict]) -> dict:
    fields = ["occupancy_iou", "occupancy_precision", "occupancy_recall",
              "target_iou", "target_hit_rate", "height_mae", "density_mae"]
    aggregated = {"samples": sum(int(item["samples"]) for item in items)}
    for field in fields:
        aggregated[field] = float(np.mean([item[field] for item in items]))
    return aggregated


def _mock_task(bundle):
    """把场景包包装成可直接走 build_task_components 的最小 Task 代理。

    build_task_components 只使用 task.scene、task.difficulty.noise_level 与
    task.seed，用轻量代理避免完整构造 Task 采样。
    """
    from types import SimpleNamespace

    difficulty = SimpleNamespace(noise_level=NoiseLevel.CLEAN)
    return SimpleNamespace(
        scene=bundle,
        difficulty=difficulty,
        seed=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="要评测的场景名；默认全部 S1–S9",
    )
    parser.add_argument("--output", default="runs/bev-fidelity/L1")
    parser.add_argument("--poses-per-scene", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenes = args.scenes if args.scenes else sorted(SCENE_REGISTRY)
    report = run_bev_fidelity(
        scenes,
        output_dir=args.output,
        poses_per_scene=args.poses_per_scene,
        seed=args.seed,
    )
    print(f"报告：{Path(args.output).resolve() / 'report.json'}")


if __name__ == "__main__":
    main()