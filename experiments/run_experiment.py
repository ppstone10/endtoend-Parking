"""批量实验 runner（骨架版）。

用法：
    python experiments/run_experiment.py --config experiments/configs/ground_baseline.json

读取 JSON 配置 → 构建引擎 → 批量执行回合 → summarize → 结果落盘 JSON。
配置 schema 随 M2/M4 扩展（场景库、噪声、方法网格、断点续跑）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller import MPCController
from interfaces import GoalPose, VehicleState
from metrics import summarize
from runtime import ClosedLoopEngine, ExpertSource, TerminalChecker
from sim import DifferentialDriveModel, ParkingEnvironment, RectangleObstacle, get_vehicle
from planner import HybridAStarPlanner, RectangleFootprintCollisionChecker


def build_env(spec: dict) -> ParkingEnvironment:
    """按配置构建环境。当前支持 corridor 预设与空场景。"""
    kind = spec.get("kind", "corridor")
    if kind == "corridor":
        return ParkingEnvironment(
            world_size=40.0,
            obstacles=[
                RectangleObstacle(-15.0, 15.0, -6.0, -2.0),
                RectangleObstacle(-15.0, 15.0, 2.0, 6.0),
            ],
        )
    if kind == "empty":
        return ParkingEnvironment(world_size=spec.get("world_size", 40.0))
    raise ValueError(f"未知环境类型 {kind}（M2 场景库接入后扩展）")


def sample_task(env: ParkingEnvironment, planner: HybridAStarPlanner, rng: np.random.Generator, cfg: dict):
    """重试式采样无碰撞且距离合规的位姿对（M2 由 TaskSampler 替代）。"""
    dist_range = cfg.get("task", {}).get("distance_range", [3.0, 12.0])
    lo, hi = dist_range
    half = env.world_size / 2.0 - 1.0
    for _ in range(500):
        sx, sy, syaw = rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(-np.pi, np.pi)
        gx, gy, gyaw = rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(-np.pi, np.pi)
        if not (planner._pose_free(sx, sy, syaw) and planner._pose_free(gx, gy, gyaw)):
            continue
        if not lo <= np.hypot(gx - sx, gy - sy) <= hi:
            continue
        return VehicleState(float(sx), float(sy), float(syaw)), GoalPose(float(gx), float(gy), float(gyaw))
    raise RuntimeError("任务采样失败：500 次尝试未找到合规位姿对")


def run_experiment(config: dict) -> dict:
    """按配置执行批量实验并返回汇总结果。"""
    vehicle = get_vehicle(config.get("vehicle", "mining_truck"))
    env = build_env(config.get("env", {"kind": "corridor"}))
    planner_cfg = config.get("planner", {})
    planner_kwargs = vehicle.planner_kwargs()
    planner_kwargs["collision_margin"] = planner_cfg.get(
        "collision_margin", planner_kwargs["collision_margin"]
    )
    planner = HybridAStarPlanner(env=env, **planner_kwargs)
    source = ExpertSource(planner)
    actual_collision_checker = RectangleFootprintCollisionChecker(
        env,
        vehicle_length=vehicle.length,
        vehicle_width=vehicle.width,
        collision_margin=0.0,
        resolution=vehicle.collision_check_resolution,
    )

    term_cfg = config.get("terminal", {})
    engine = ClosedLoopEngine(
        vehicle_model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
        mpc=MPCController(dt=0.1, horizon=10, seed=0, **vehicle.mpc_kwargs()),
        source=source,
        terminal=TerminalChecker(
            tol_pos=term_cfg.get("tol_pos", 0.3),
            tol_yaw=np.deg2rad(term_cfg.get("tol_yaw_deg", 10.0)),
        ),
        env=env,
        replan_every=config.get("replan_every", 1),
        max_steps=config.get("max_steps", 600),
        collision_checker=actual_collision_checker,
        **vehicle.collision_kwargs(),
    )

    n = config.get("episodes", 50)
    rng = np.random.default_rng(config.get("seed", 0))
    results = []
    plan_failures = 0
    t0 = time.perf_counter()
    while len(results) < n:
        start, goal = sample_task(env, planner, rng, config)
        try:
            result = engine.run(start, goal)
        except (RuntimeError, ValueError):
            # 规划失败（无解/发散/超限）：换任务重采，单独计数不入回合。
            plan_failures += 1
            if plan_failures > 10 * n:
                raise RuntimeError(f"规划失败 {plan_failures} 次，采样区域可能不可行")
            continue
        result.meta = {"episode": len(results), "vehicle": vehicle.name, "env": config.get("env", {}).get("kind", "corridor")}
        results.append(result)
        if (len(results)) % 25 == 0:
            print(f"  进度 {len(results)}/{n}，当前成功率 {sum(r.success for r in results) / len(results):.0%}")
    elapsed = time.perf_counter() - t0

    summary = summarize(results)
    summary["config"] = config
    summary["elapsed_sec"] = elapsed
    summary["plan_failures"] = plan_failures
    summary["per_episode"] = [r.to_dict() for r in results]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    name = os.path.splitext(os.path.basename(args.config))[0]
    print(f"实验 {name}：{config.get('episodes', 50)} 回合，车辆 {config.get('vehicle', 'mining_truck')}")
    summary = run_experiment(config)

    out_path = args.out or os.path.join(os.path.dirname(args.config), "..", "results", f"{name}.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    print(
        f"结果：成功率 {summary['success_rate']:.1%}，碰撞率 {summary['collision_rate']:.1%}，"
        f"位置误差 {summary['final_pos_err_mean']:.2f}±{summary['final_pos_err_std']:.2f}m，"
        f"航向误差 {np.degrees(summary['final_yaw_err_mean']):.1f}°，"
        f"跟踪RMS {summary['tracking_rms_mean']:.3f}m，耗时 {summary['elapsed_sec']:.0f}s"
    )
    print(f"已保存到 {out_path}")


if __name__ == "__main__":
    main()
