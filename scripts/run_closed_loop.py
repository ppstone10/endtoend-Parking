"""闭环泊车演示：ClosedLoopEngine 驱动（M1 地基）。

用法：
    python scripts/run_closed_loop.py [--source expert|network] [--samples N] \
        [--steps MAX] [--data DATA.npz] [--model M.pt] [--replan-every K]

expert：专家轨迹 + MPC（地基基线，验收成功率）；
network：感知 → BEV → MineParkingNet → MPC 滚动闭环（端到端主线）。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller import MPCController
from interfaces import GoalPose, VehicleState
from metrics import summarize
from runtime import ClosedLoopEngine, ExpertSource, NetworkSource, TerminalChecker
from sim import DifferentialDriveModel, MINING_DRILL_RIG
from scripts.train import build_env, build_pipeline


def load_model(path: str, bev_channels: int, horizon: int, dt: float):
    import torch

    from model import MineParkingNet

    model = MineParkingNet(bev_channels=bev_channels, horizon=horizon, dt=dt)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def make_source(kind: str, args, data: dict | None):
    env = build_env()
    if kind == "expert":
        from planner import HybridAStarPlanner
        return ExpertSource(
            HybridAStarPlanner(env=env, **MINING_DRILL_RIG.planner_kwargs())
        ), env
    model = load_model(
        args.model,
        bev_channels=data["bevs"].shape[1],
        horizon=min(20, data["trajs"].shape[1]),
        dt=float(data["dt"][0]),
    )
    return NetworkSource(build_pipeline(env), model), env


def sample_task(data: dict, index: int) -> tuple[VehicleState, GoalPose, int]:
    """从数据集取第 index 条样本的起始状态与目标（全局坐标）。"""
    goal = data["goals"][index]
    state_arr = data["states"][index]
    n_valid = int(data["masks"][index].sum())
    return (
        VehicleState(
            float(state_arr[0]), float(state_arr[1]), float(state_arr[2]),
            float(state_arr[3]), float(state_arr[4]),
        ),
        GoalPose(float(goal[0]), float(goal[1]), float(goal[2])),
        n_valid,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["expert", "network"], default="expert")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--model", type=str, default="mineparkingnet.pt")
    args = parser.parse_args()

    data = None
    if args.source == "network":
        if args.data:
            from dataset import DatasetGenerator

            data = DatasetGenerator.load(args.data)
        else:
            print("network 源需要 --data 指定数据集（起点/目标来自样本）")
            sys.exit(1)

    source, env = make_source(args.source, args, data)

    # 回合来源：network 用数据集样本；expert 随机采样无碰撞位姿对。
    tasks = []
    if data is not None:
        for i in range(min(args.samples, len(data["bevs"]))):
            start, goal, _ = sample_task(data, i)
            tasks.append((start, goal))
    else:
        from planner import HybridAStarPlanner
        planner = HybridAStarPlanner(env=env, **MINING_DRILL_RIG.planner_kwargs())
        rng = np.random.default_rng(0)
        half = env.world_size / 2.0 - 1.0
        while len(tasks) < args.samples:
            sx, sy, syaw = rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(-np.pi, np.pi)
            gx, gy, gyaw = rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(-np.pi, np.pi)
            if not (planner._pose_free(sx, sy, syaw) and planner._pose_free(gx, gy, gyaw)):
                continue
            if not 3.0 <= np.hypot(gx - sx, gy - sy) <= 12.0:
                continue
            tasks.append(
                (VehicleState(float(sx), float(sy), float(syaw)), GoalPose(float(gx), float(gy), float(gyaw)))
            )

    engine = ClosedLoopEngine(
        vehicle_model=DifferentialDriveModel(
            **MINING_DRILL_RIG.vehicle_model_kwargs()
        ),
        mpc=MPCController(
            dt=0.1,
            horizon=10,
            seed=0,
            **MINING_DRILL_RIG.mpc_kwargs(),
        ),
        source=source,
        terminal=TerminalChecker(tol_pos=0.3, tol_yaw=np.deg2rad(10.0)),
        env=env,
        replan_every=args.replan_every,
        max_steps=args.steps,
        **MINING_DRILL_RIG.collision_kwargs(),
    )

    print(f"轨迹源={args.source}，共 {len(tasks)} 个回合（到达阈值 0.3m / 10°）")
    results = []
    for i, (start, goal) in enumerate(tasks):
        result = engine.run(start, goal)
        results.append(result)
        print(
            f"  回合 {i}: {'成功' if result.success else f'失败({result.failure})'} "
            f"位置误差 {result.final_pos_err:.2f}m 航向误差 {np.degrees(result.final_yaw_err):.1f}° "
            f"步数 {result.steps} 路径 {result.path_length:.1f}m 时间 {result.parking_time:.1f}s"
        )

    summary = summarize(results)
    print(
        f"成功率 {summary['success_rate']:.0%}，碰撞率 {summary['collision_rate']:.0%}，"
        f"平均位置误差 {summary['final_pos_err_mean']:.2f}±{summary['final_pos_err_std']:.2f}m，"
        f"平均航向误差 {np.degrees(summary['final_yaw_err_mean']):.1f}°，"
        f"平均跟踪RMS {summary['tracking_rms_mean']:.2f}m，"
        f"平均推理 {summary['inference_ms_mean']:.1f}ms"
    )


if __name__ == "__main__":
    main()
