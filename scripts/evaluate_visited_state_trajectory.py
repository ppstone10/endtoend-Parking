"""L2 闭环访问状态轨迹质量评测：网络 vs 专家在闭环诱导状态上的对比报告。

对数据集任务复原场景后，用 deployment 模型跑网络闭环，在每次重规划时
记录当前车辆状态、网络预测轨迹与专家在同一状态的重规划轨迹，输出：

- 访问状态开环误差（网络 vs 专家：ADE/FDE/航向 MAE）；
- 近端质量（距目标 <3m：网络预测长度/终点距目标/终点航向/方向切换）；
- 跨重规划一致性（相邻重规划的终点航向跳变与长度跳变）；
- 按场景/任务/机动/噪声分组。

运行：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/evaluate_visited_state_trajectory.py \
        --data data/task_dataset/tracked_pivot_v7_3000/val.npz \
        --model runs/training/v9-safety-v1/net-v1/deployment.pt \
        --samples 34 --output runs/visited-state/L2-v1/report.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller import MPCController
from dataset import DatasetGenerator, build_task_components
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
    select_evaluation_indices,
)
from interfaces import GoalPose, VehicleState
from metrics.visited_state import (
    VisitedStateRecord,
    analyze_visited_state_records,
)
from planner import RectangleFootprintCollisionChecker
from runtime import ClosedLoopEngine, NetworkSource, ReplanningExpertSource, TerminalChecker
from sim import DifferentialDriveModel, VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.data import validate_model_dataset
from training.reporting import atomic_write_json
from viz.visited_state import save_visited_state_report


class _HookedNetworkSource(NetworkSource):
    """网络轨迹源 + 同时采集专家重规划与网络预测的 hook。"""

    def __init__(self, pipeline, model, expert_source, goal, records) -> None:
        super().__init__(pipeline, model)
        self.expert_source = expert_source
        self._goal = goal
        self._records = records

    def next_trajectory(self, state: VehicleState):
        network_traj, ms = super().next_trajectory(state)
        try:
            expert_traj, _ = self.expert_source.next_trajectory(state)
            expert_points = np.asarray(expert_traj.points, dtype=np.float64)
        except (RuntimeError, ValueError):
            expert_points = None
        self._records.append(
            VisitedStateRecord(
                step=len(self._records),
                state=np.asarray([state.x, state.y, state.yaw]),
                network_points=np.asarray(network_traj.points, dtype=np.float64),
                expert_points=expert_points,
                goal=np.asarray([self._goal.x, self._goal.y, self._goal.yaw]),
                meta={},
            )
        )
        return network_traj, ms


def run_visited_state_evaluation(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    samples: int = 0,
    selection: str = "stratified",
    max_steps: int = 600,
    replan_every: int = 10,
    control_seed: int = 0,
    near_threshold_m: float = 3.0,
) -> dict:
    """在数据集任务上采集闭环访问状态并输出轨迹质量报告。"""
    data_source = Path(data_path).resolve()
    checkpoint_source = Path(checkpoint_path).resolve()
    data = DatasetGenerator.load(data_source)
    metadata = data.get("task_meta")
    if int(data.get("schema_version", -1)) != 2 or not isinstance(metadata, list):
        raise ValueError("L2 评测要求 schema v2 数据集与 task_meta")
    manifest = load_dataset_manifest(data_source)
    try:
        vehicle = VehicleConfig(**manifest["vehicle_model"])
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest vehicle_model 无效") from exc
    loaded = load_model_checkpoint(checkpoint_source)
    validate_model_dataset(loaded.model, data)
    indices = select_evaluation_indices(metadata, samples=samples, strategy=selection)
    if not indices:
        raise ValueError("没有可评测样本")

    all_records: list[VisitedStateRecord] = []
    for ordinal, index in enumerate(indices, start=1):
        restored = reconstruct_dataset_task(
            metadata[index], root_seed=int(manifest["seed"]), vehicle=vehicle
        )
        state = VehicleState.from_array(np.asarray(data["states"])[index])
        goal = restored.goal
        planner, pipeline = build_task_components(restored.task, vehicle)
        records: list[VisitedStateRecord] = []
        expert_source = ReplanningExpertSource(planner)
        expert_source.begin(state, goal)
        source = _HookedNetworkSource(pipeline, loaded.model, expert_source, goal, records)
        actual_collision_checker = RectangleFootprintCollisionChecker(
            restored.task.scene.env,
            vehicle_length=vehicle.length,
            vehicle_width=vehicle.width,
            collision_margin=0.0,
            resolution=vehicle.collision_check_resolution,
        )
        difficulty = metadata[index]["difficulty"]
        episode_meta = {
            "dataset_index": index,
            "task_id": restored.task.task_id,
            "scene_name": restored.task.scene_name,
            "task_type": restored.task.task_type.value,
            "maneuver": difficulty["maneuver"],
            "noise_level": difficulty["noise_level"],
            "adjacent_occupancy": int(difficulty["adjacent_occupancy"]),
        }
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
            meta=episode_meta,
            collision_checker=actual_collision_checker,
            **vehicle.collision_kwargs(),
        )
        result = engine.run(state, goal)
        for record in records:
            record.meta = dict(episode_meta)
            record.meta["result_success"] = result.success
            record.meta["result_failure"] = result.failure
        all_records.extend(records)
        print(
            f"[{ordinal}/{len(indices)}] {episode_meta['task_id']} "
            f"{'成功' if result.success else f'失败({result.failure})'} "
            f"重规划 {len(records)} 次",
            flush=True,
        )

    report, rows = analyze_visited_state_records(
        all_records, near_threshold_m=near_threshold_m
    )
    report = {
        "schema_version": 1,
        "task": "L2_visited_state_trajectory",
        "data": str(data_source),
        "checkpoint": str(checkpoint_source),
        "model_name": loaded.model_name,
        "checkpoint_epoch": loaded.epoch,
        "selection": selection,
        "requested_samples": samples,
        "selected_indices": indices,
        "max_steps": max_steps,
        "replan_every": replan_every,
        "control_seed": control_seed,
        "near_threshold_m": near_threshold_m,
        "vehicle_model": vehicle.to_metadata(),
        "overall": report["overall"],
        "groups": report["groups"],
        "replans": rows,
    }
    if output_path is not None:
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, report)
        plot_paths = save_visited_state_report(
            report,
            destination.with_name(destination.stem + "_degradation"),
        )
        report["artifacts"] = {
            "plot_png": plot_paths[0],
            "plot_pdf": plot_paths[1],
        }
        atomic_write_json(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="schema v2 数据集 NPZ")
    parser.add_argument("--model", required=True, help="deployment checkpoint")
    parser.add_argument("--samples", type=int, default=0, help="<=0 表示全部")
    parser.add_argument("--selection", choices=["stratified", "head"], default="stratified")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--near-threshold", type=float, default=3.0)
    parser.add_argument("--output", default="runs/visited-state/L2/report.json")
    args = parser.parse_args()
    report = run_visited_state_evaluation(
        args.data,
        args.model,
        output_path=args.output,
        samples=args.samples,
        selection=args.selection,
        max_steps=args.max_steps,
        replan_every=args.replan_every,
        control_seed=args.seed,
        near_threshold_m=args.near_threshold,
    )
    print(f"报告：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()