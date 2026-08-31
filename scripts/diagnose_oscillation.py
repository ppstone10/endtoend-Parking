"""方向 B 诊断：振荡样本重规划机制分析。

对选定 val 振荡样本重建任务，用 deployment 模型运行闭环，
捕获每次重规划的网络预测轨迹，分析终点航向偏差、方向切换与停止长度。
"""

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


def _traj_stats(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return {"n": pts.shape[0], "end_yaw": None, "flips": 0, "length_m": 0.0}
    seg = np.diff(pts[:, :2], axis=0)
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    flips = 0
    if headings.shape[0] >= 2:
        flips = int(np.sum(np.abs(np.diff(headings)) > np.pi / 2.0))
    length_m = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    return {
        "n": int(pts.shape[0]),
        "end_x": float(pts[-1, 0]),
        "end_y": float(pts[-1, 1]),
        "end_yaw": float(pts[-1, 2]),
        "flips": flips,
        "length_m": length_m,
    }


def analyze_index(
    index: int,
    data,
    manifest,
    checkpoint,
    model_name,
    *,
    control_seed: int = 0,
    replan_every: int = 10,
    max_steps: int = 600,
) -> dict:
    metadata = data["task_meta"][index]
    vehicle = VehicleConfig(**manifest["vehicle_model"])
    restored = reconstruct_dataset_task(
        metadata, root_seed=int(manifest["seed"]), vehicle=vehicle
    )
    planner, pipeline = build_task_components(restored.task, vehicle)
    loaded = load_model_checkpoint(checkpoint)
    validate_model_dataset(loaded.model, data)
    goal = restored.goal

    captured = []

    class HookedSource(NetworkSource):
        def next_trajectory(self, state):
            traj, ms = super().next_trajectory(state)
            captured.append(
                {
                    "step": engine_step[0],
                    "state": (state.x, state.y, state.yaw),
                    "traj": np.asarray(traj.points, copy=True),
                }
            )
            return traj, ms

    engine_step = [0]
    source = HookedSource(pipeline, loaded.model)
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
            dt=0.1, horizon=10, seed=control_seed + index, **vehicle.mpc_kwargs()
        ),
        source=source,
        terminal=TerminalChecker(restored.tol_pos, restored.tol_yaw),
        env=restored.task.scene.env,
        replan_every=replan_every,
        max_steps=max_steps,
        collision_checker=actual_collision_checker,
        **vehicle.collision_kwargs(),
    )

    # wrap engine.run to track step
    original_next = source.next_trajectory

    def stepping_next(state):
        traj, ms = original_next(state)
        captured[-1]["step"] = engine_step[0]
        return traj, ms

    source.next_trajectory = stepping_next

    start = VehicleState.from_array(np.asarray(data["states"])[index])
    result = engine.run(start, goal)

    # goal local to start
    dx = goal.x - start.x
    dy = goal.y - start.y
    cos_yaw, sin_yaw = np.cos(start.yaw), np.sin(start.yaw)
    goal_local = np.array(
        [
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
            float(
                np.arctan2(
                    np.sin(goal.yaw - start.yaw), np.cos(goal.yaw - start.yaw)
                )
            ),
        ]
    )

    replan_rows = []
    for cap in captured:
        traj_global = cap["traj"]
        stats = _traj_stats(traj_global)
        sx, sy, syaw = cap["state"]
        # transform global traj back to local of current state
        dxs = traj_global[:, 0] - sx
        dys = traj_global[:, 1] - sy
        c, s = np.cos(syaw), np.sin(syaw)
        local = np.empty_like(traj_global)
        local[:, 0] = c * dxs + s * dys
        local[:, 1] = -s * dxs + c * dys
        local[:, 2] = np.arctan2(
            np.sin(traj_global[:, 2] - syaw), np.cos(traj_global[:, 2] - syaw)
        )
        if stats["end_yaw"] is not None:
            end_yaw = local[-1, 2]
        else:
            end_yaw = None
        # goal in current state local frame
        gdx = goal.x - sx
        gdy = goal.y - sy
        gl = np.array(
            [
                c * gdx + s * gdy,
                -s * gdx + c * gdy,
                float(np.arctan2(np.sin(goal.yaw - syaw), np.cos(goal.yaw - syaw))),
            ]
        )
        replan_rows.append(
            {
                "step": cap["step"],
                "state": cap["state"],
                "traj_n": stats["n"],
                "traj_len_m": round(stats["length_m"], 2),
                "traj_end_local": (round(local[-1, 0], 2), round(local[-1, 1], 2)) if stats["end_yaw"] is not None else None,
                "traj_end_yaw_local": round(float(end_yaw), 2) if end_yaw is not None else None,
                "goal_local": (round(float(gl[0]), 2), round(float(gl[1]), 2), round(float(gl[2]), 2)),
                "yaw_err_to_goal": round(float(np.arctan2(np.sin(end_yaw - gl[2]), np.cos(end_yaw - gl[2]))), 2) if end_yaw is not None else None,
                "flips": stats["flips"],
            }
        )

    return {
        "index": index,
        "task_id": restored.task.task_id,
        "scene": restored.task.scene_name,
        "task_type": restored.task.task_type.value,
        "maneuver": metadata["difficulty"]["maneuver"],
        "result": {
            "success": result.success,
            "failure": result.failure,
            "steps": result.steps,
            "final_pos_err": round(result.final_pos_err, 3),
            "final_yaw_err_deg": round(np.degrees(result.final_yaw_err), 1),
        },
        "goal_local_from_start": [round(v, 2) for v in goal_local],
        "replans": replan_rows,
    }


def main() -> None:
    data_path = r"data\task_dataset\tracked_pivot_v7_3000\val.npz"
    checkpoint = r"runs\training\v9-safety-v1\net-v1\deployment.pt"
    indices = [10, 20, 48, 66, 140]
    from pathlib import Path
    data = DatasetGenerator.load(data_path)
    manifest = load_dataset_manifest(Path(data_path))
    for idx in indices:
        print("=" * 100)
        rep = analyze_index(
            idx, data, manifest, checkpoint, "net-v1"
        )
        print(f"index={idx} {rep['task_id']} {rep['maneuver']}")
        print(
            f"  result: success={rep['result']['success']} failure={rep['result']['failure']} "
            f"steps={rep['result']['steps']} pos={rep['result']['final_pos_err']}m "
            f"yaw={rep['result']['final_yaw_err_deg']}deg"
        )
        print(f"  goal_from_start(local) = {rep['goal_local_from_start']}")
        print(
            f"  {'step':>5} {'traj_n':>6} {'len_m':>6} {'end_local':>14} {'end_yaw':>7} "
            f"{'yaw_err':>7} {'flips':>5}"
        )
        for row in rep["replans"]:
            print(
                f"  {row['step']:>5} {row['traj_n']:>6} {row['traj_len_m']:>6} "
                f"{str(row['traj_end_local']):>14} {row['traj_end_yaw_local']:>7} "
                f"{row['yaw_err_to_goal']:>7} {row['flips']:>5}"
            )


if __name__ == "__main__":
    main()