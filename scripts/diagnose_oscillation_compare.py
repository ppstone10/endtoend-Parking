"""方向 B 决定性对比：同一振荡样本，v7(成功) vs v9(振荡) 的近端终点预测质量。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller import MPCController
from dataset import DatasetGenerator, build_task_components
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
)
from interfaces import VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import ClosedLoopEngine, NetworkSource, TerminalChecker
from sim import DifferentialDriveModel, VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.data import validate_model_dataset
from pathlib import Path


def analyze(index, data, manifest, checkpoint, *, control_seed=0, replan_every=10, max_steps=600):
    metadata = data["task_meta"][index]
    vehicle = VehicleConfig(**manifest["vehicle_model"])
    restored = reconstruct_dataset_task(metadata, root_seed=int(manifest["seed"]), vehicle=vehicle)
    planner, pipeline = build_task_components(restored.task, vehicle)
    loaded = load_model_checkpoint(checkpoint)
    validate_model_dataset(loaded.model, data)
    goal = restored.goal
    records = []

    class HookedSource(NetworkSource):
        def next_trajectory(self, state):
            traj, ms = super().next_trajectory(state)
            records.append({"state": (float(state.x), float(state.y), float(state.yaw)), "traj": np.asarray(traj.points, copy=True)})
            return traj, ms

    source = HookedSource(pipeline, loaded.model)
    checker = RectangleFootprintCollisionChecker(
        restored.task.scene.env,
        vehicle_length=vehicle.length,
        vehicle_width=vehicle.width,
        collision_margin=0.0,
        resolution=vehicle.collision_check_resolution,
    )
    engine = ClosedLoopEngine(
        vehicle_model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
        mpc=MPCController(dt=0.1, horizon=10, seed=control_seed + index, **vehicle.mpc_kwargs()),
        source=source,
        terminal=TerminalChecker(restored.tol_pos, restored.tol_yaw),
        env=restored.task.scene.env,
        replan_every=replan_every,
        max_steps=max_steps,
        collision_checker=checker,
        **vehicle.collision_kwargs(),
    )
    start = VehicleState.from_array(np.asarray(data["states"])[index])
    result = engine.run(start, goal)

    near_rows = []  # only when veh->goal < 3m
    for rec in records:
        sx, sy, syaw = rec["state"]
        d_goal = float(np.hypot(goal.x - sx, goal.y - sy))
        if d_goal > 3.0:
            continue
        traj = rec["traj"]
        n = traj.shape[0]
        end = traj[-1]
        d_end_goal = float(np.hypot(goal.x - end[0], goal.y - end[1]))
        end_yaw_local = float(np.arctan2(np.sin(end[2] - syaw), np.cos(end[2] - syaw)))
        goal_yaw_local = float(np.arctan2(np.sin(goal.yaw - syaw), np.cos(goal.yaw - syaw)))
        yaw_err = float(np.arctan2(np.sin(end_yaw_local - goal_yaw_local), np.cos(end_yaw_local - goal_yaw_local)))
        seg = np.diff(traj[:, :2], axis=0)
        head = np.arctan2(seg[:, 1], seg[:, 0])
        flips = int(np.sum(np.abs(np.diff(head)) > np.pi / 2.0)) if head.shape[0] >= 2 else 0
        near_rows.append(
            (round(d_goal, 2), n, round(float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))), 2),
             round(d_end_goal, 2), round(yaw_err, 2), flips)
        )
    return {
        "result": (result.success, result.failure, result.steps,
                   round(result.final_pos_err, 2), round(np.degrees(result.final_yaw_err), 1)),
        "near_rows": near_rows,
    }


def main():
    data = DatasetGenerator.load(r"data\task_dataset\tracked_pivot_v7_3000\val.npz")
    manifest = load_dataset_manifest(Path(r"data\task_dataset\tracked_pivot_v7_3000\val.npz"))
    v7_ckpt = r"runs\training\v7-flow-v3\net-v1\deployment.pt"
    v9_ckpt = r"runs\training\v9-safety-v1\net-v1\deployment.pt"
    for idx in [10, 11, 21, 30, 48, 57, 66]:
        r7 = analyze(idx, data, manifest, v7_ckpt)
        r9 = analyze(idx, data, manifest, v9_ckpt)
        print("=" * 100)
        print(f"idx={idx}  v7: success={r7['result'][0]} failure={r7['result'][1]} pos={r7['result'][3]} yaw={r7['result'][4]}deg"
              f"   |   v9: success={r9['result'][0]} failure={r9['result'][1]} pos={r9['result'][3]} yaw={r9['result'][4]}deg")
        print(f"  v7 近端(<3m): veh->goal, n, len, end->goal, yaw_err, flips")
        for row in r7["near_rows"][:6]:
            print(f"    {row}")
        print(f"  v9 近端(<3m): veh->goal, n, len, end->goal, yaw_err, flips")
        for row in r9["near_rows"][:6]:
            print(f"    {row}")


if __name__ == "__main__":
    main()