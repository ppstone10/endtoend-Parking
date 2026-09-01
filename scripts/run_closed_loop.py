"""专家基线或数据集任务网络闭环评测入口。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller import MPCController
from experiments.closed_loop_evaluation import run_dataset_network_evaluation
from interfaces import GoalPose, VehicleState
from metrics import summarize
from planner import HybridAStarPlanner, RectangleFootprintCollisionChecker
from runtime import ClosedLoopEngine, ExpertSource, TerminalChecker
from sim import DifferentialDriveModel, MINING_DRILL_RIG
from scripts.train import build_env


def _print_episode(index: int, total: int, result) -> None:
    status = "成功" if result.success else f"失败({result.failure})"
    print(
        f"[{index}/{total}] {result.meta.get('task_id', '')} {status} "
        f"位置 {result.final_pos_err:.2f}m 航向 {np.degrees(result.final_yaw_err):.1f}° "
        f"步数 {result.steps} 推理 {result.inference_ms:.1f}ms",
        flush=True,
    )


def _run_expert(args: argparse.Namespace) -> dict:
    env = build_env()
    planner = HybridAStarPlanner(env=env, **MINING_DRILL_RIG.planner_kwargs())
    source = ExpertSource(planner)
    actual_collision_checker = RectangleFootprintCollisionChecker(
        env,
        vehicle_length=MINING_DRILL_RIG.length,
        vehicle_width=MINING_DRILL_RIG.width,
        collision_margin=0.0,
        resolution=MINING_DRILL_RIG.collision_check_resolution,
    )
    engine = ClosedLoopEngine(
        vehicle_model=DifferentialDriveModel(**MINING_DRILL_RIG.vehicle_model_kwargs()),
        mpc=MPCController(
            dt=0.1,
            horizon=10,
            seed=args.seed,
            **MINING_DRILL_RIG.mpc_kwargs(),
        ),
        source=source,
        terminal=TerminalChecker(tol_pos=0.3, tol_yaw=np.deg2rad(10.0)),
        env=env,
        replan_every=args.replan_every,
        max_steps=args.steps,
        collision_checker=actual_collision_checker,
        **MINING_DRILL_RIG.collision_kwargs(),
    )
    rng = np.random.default_rng(args.seed)
    tasks = []
    half = env.world_size / 2.0 - 1.0
    while len(tasks) < args.samples:
        sx, sy, syaw = (
            rng.uniform(-half, half),
            rng.uniform(-half, half),
            rng.uniform(-np.pi, np.pi),
        )
        gx, gy, gyaw = (
            rng.uniform(-half, half),
            rng.uniform(-half, half),
            rng.uniform(-np.pi, np.pi),
        )
        if not (
            planner._pose_free(sx, sy, syaw)
            and planner._pose_free(gx, gy, gyaw)
        ):
            continue
        if 3.0 <= np.hypot(gx - sx, gy - sy) <= 12.0:
            tasks.append(
                (
                    VehicleState(float(sx), float(sy), float(syaw)),
                    GoalPose(float(gx), float(gy), float(gyaw)),
                )
            )
    results = []
    for index, (start, goal) in enumerate(tasks):
        result = engine.run(start, goal)
        results.append(result)
        _print_episode(index + 1, len(tasks), result)
    return {"overall": summarize(results)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["expert", "network"], default="expert")
    parser.add_argument(
        "--samples", type=int, default=5, help="network 中 <=0 表示全部样本"
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--replan-every",
        type=int,
        default=10,
        help="网络整段轨迹的控制复用周期；当前 deployment 默认 10",
    )
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--selection", choices=["stratified", "head"], default="stratified"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--safety-mode",
        choices=["none", "expert_fallback", "hierarchical"],
        default="none",
        help="网络轨迹安全/分层模式；对照实验保留 none，安全防线使用 expert_fallback，"
             "分层短距局部规划使用 hierarchical",
    )
    args = parser.parse_args()

    if args.source == "expert":
        report = _run_expert(args)
    else:
        if not args.data or not args.model:
            parser.error("network 源要求同时提供 --data 与 --model")
        report = run_dataset_network_evaluation(
            args.data,
            args.model,
            output_path=(Path(args.output) if args.output else None),
            samples=args.samples,
            selection=args.selection,
            max_steps=args.steps,
            replan_every=args.replan_every,
            control_seed=args.seed,
            progress=_print_episode,
            safety_mode=args.safety_mode,
        )
    summary = report["overall"]
    print(
        f"汇总：成功率 {summary['success_rate']:.1%}，碰撞率 {summary['collision_rate']:.1%}，"
        f"位置误差 {summary['final_pos_err_mean']:.2f}±{summary['final_pos_err_std']:.2f}m，"
        f"航向误差 {np.degrees(summary['final_yaw_err_mean']):.1f}°，"
        f"跟踪 RMS {summary['tracking_rms_mean']:.3f}m"
    )
    if args.output and args.source == "network":
        print(f"报告：{Path(args.output).resolve()}")
    shield = report.get("safety_shield")
    if shield is not None:
        print(
            f"安全门禁：检查 {shield['checks']} 次，干预 {shield['interventions']} 次 "
            f"({shield['intervention_rate']:.1%})，回退失败 {shield['fallback_failures']} 次"
        )


if __name__ == "__main__":
    main()
