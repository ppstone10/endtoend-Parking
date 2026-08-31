"""方向 B 聚焦：车辆距目标距离 vs 网络预测轨迹终点收敛性。"""

from __future__ import annotations

import json
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

    # sample replay rows (cap at 14)
    rows = []
    for rec in records[:: max(1, len(records) // 14)]:
        sx, sy, syaw = rec["state"]
        d_to_goal = float(np.hypot(goal.x - sx, goal.y - sy))
        traj = rec["traj"]
        n = traj.shape[0]
        # local end of predicted trajectory
        end = traj[-1]
        dx, dy = end[0] - sx, end[1] - sy
        c, s = np.cos(syaw), np.sin(syaw)
        local_end = np.array([c * dx + s * dy, -s * dx + c * dy])
        # distance from predicted end to goal
        d_end_goal = float(np.hypot(goal.x - end[0], goal.y - end[1]))
        end_yaw_local = float(np.arctan2(np.sin(end[2] - syaw), np.cos(end[2] - syaw)))
        goal_yaw_local = float(np.arctan2(np.sin(goal.yaw - syaw), np.cos(goal.yaw - syaw)))
        yaw_err = float(np.arctan2(np.sin(end_yaw_local - goal_yaw_local), np.cos(end_yaw_local - goal_yaw_local)))
        # direction flips along predicted trajectory
        pts = traj
        seg = np.diff(pts[:, :2], axis=0)
        head = np.arctan2(seg[:, 1], seg[:, 0])
        flips = int(np.sum(np.abs(np.diff(head)) > np.pi / 2.0)) if head.shape[0] >= 2 else 0
        rows.append(
            {
                "d_veh_goal": round(d_to_goal, 2),
                "traj_n": n,
                "traj_len_m": round(float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))), 2),
                "end_local": (round(float(local_end[0]), 2), round(float(local_end[1]), 2)),
                "d_end_goal": round(d_end_goal, 2),
                "yaw_err": round(yaw_err, 2),
                "flips": flips,
            }
        )
    return {
        "index": index,
        "task_id": restored.task.task_id,
        "maneuver": metadata["difficulty"]["maneuver"],
        "result": (result.success, result.failure, result.steps,
                   round(result.final_pos_err, 2), round(np.degrees(result.final_yaw_err), 1)),
        "tol_pos": restored.tol_pos,
        "rows": rows,
    }


def main():
    data = DatasetGenerator.load(r"data\task_dataset\tracked_pivot_v7_3000\val.npz")
    manifest = load_dataset_manifest(Path(r"data\task_dataset\tracked_pivot_v7_3000\val.npz"))
    checkpoint = r"runs\training\v9-safety-v1\net-v1\deployment.pt"
    for idx in [10, 20, 48, 66, 140]:
        rep = analyze(idx, data, manifest, checkpoint)
        print("=" * 100)
        print(f"idx={idx} {rep['task_id']} {rep['maneuver']} tol_pos={rep['tol_pos']}")
        print(f"  result: success={rep['result'][0]} failure={rep['result'][1]} steps={rep['result'][2]} "
              f"pos={rep['result'][3]}m yaw={rep['result'][4]}deg")
        print(f"  {'veh->goal':>9} {'traj_n':>6} {'len_m':>6} {'end_local':>12} {'end->goal':>9} {'yaw_err':>7} {'flips':>5}")
        for r in rep["rows"]:
            print(f"  {r['d_veh_goal']:>9} {r['traj_n']:>6} {r['traj_len_m']:>6} "
                  f"{str(r['end_local']):>12} {r['d_end_goal']:>9} {r['yaw_err']:>7} {r['flips']:>5}")


if __name__ == "__main__":
    main()