"""L3 单周期 MPC 跟踪复核：给定参考轨迹，验证控制器能否在单个重规划周期内跟踪。

目标：把"控制层"与"规划层"分离验证。对每个数据集任务：
- 参考轨迹来源可选：专家轨迹（数据集已存）或网络预测（deployment 在起点推理）。
- 只运行一个重规划周期（replan_every 个控制步），测量跟踪横向 RMS、
  参考段末端偏差与进度推进，不进入多周期滚动，也不判最终到达。

指标（逐样本 + 聚合）：
- tracking_rms_m：相对参考轨迹的横向偏差 RMS；
- end_offset_m：单周期结束位姿相对目标位姿的偏差；
- progress_m / ref_length_m / reach_ratio：实际推进 / 参考段总长。

运行：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/evaluate_single_cycle_tracking.py \
        --data data/task_dataset/tracked_pivot_v7_3000/val.npz \
        --samples 34 --source expert --output runs/single-cycle/L3-expert/report.json
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/evaluate_single_cycle_tracking.py \
        --data data/task_dataset/tracked_pivot_v7_3000/val.npz \
        --model runs/training/v7-flow-v3/net-v1/deployment.pt \
        --samples 34 --source network --output runs/single-cycle/L3-network/report.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller import MPCController
from dataset import DatasetGenerator, build_task_components
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
    select_evaluation_indices,
)
from interfaces import Trajectory, VehicleState
from planner import RectangleFootprintCollisionChecker
from runtime import ClosedLoopEngine, NetworkSource, TerminalChecker
from sim import DifferentialDriveModel, VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.data import validate_model_dataset
from training.reporting import atomic_write_json


class _FixedReferenceSource:
    """单周期用固定参考轨迹的轨迹源（不重规划）。"""

    def __init__(self, reference: Trajectory) -> None:
        self._reference = reference

    def begin(self, start_state, goal) -> None:
        self._goal = goal

    def next_trajectory(self, state):
        return self._reference, 0.0


def _dataset_expert_trajectory(data: dict, index: int) -> Trajectory:
    points = np.asarray(data["trajs"][index], dtype=np.float64)
    mask = np.asarray(data["masks"][index])
    n = int(mask.sum())
    if n < 2:
        raise ValueError(f"样本 {index} 专家轨迹长度不足")
    dt = float(np.asarray(data["dt"]).reshape(-1)[0])
    return Trajectory(points=points[:n], dt=dt)


def run_single_cycle_tracking(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    samples: int = 0,
    selection: str = "stratified",
    replan_every: int = 10,
    control_seed: int = 0,
    source: str = "expert",
) -> dict:
    """对数据集任务运行单周期 MPC 跟踪复核。"""
    data_source = Path(data_path).resolve()
    data = DatasetGenerator.load(data_source)
    metadata = data.get("task_meta")
    if int(data.get("schema_version", -1)) != 2 or not isinstance(metadata, list):
        raise ValueError("L3 评测要求 schema v2 数据集与 task_meta")
    manifest = load_dataset_manifest(data_source)
    try:
        vehicle = VehicleConfig(**manifest["vehicle_model"])
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest vehicle_model 无效") from exc
    indices = select_evaluation_indices(metadata, samples=samples, strategy=selection)
    if not indices:
        raise ValueError("没有可评测样本")

    loaded = None
    if source == "network":
        loaded = load_model_checkpoint(Path(checkpoint_path).resolve())
        validate_model_dataset(loaded.model, data)

    rows = []
    for ordinal, index in enumerate(indices, start=1):
        restored = reconstruct_dataset_task(
            metadata[index], root_seed=int(manifest["seed"]), vehicle=vehicle
        )
        start = VehicleState.from_array(np.asarray(data["states"])[index])
        planner, pipeline = build_task_components(restored.task, vehicle)
        if source == "expert":
            reference = _dataset_expert_trajectory(data, index)
            ref_source = "expert"
        else:
            network_source = NetworkSource(pipeline, loaded.model)
            network_source.begin(start, restored.goal)
            reference, _ = network_source.next_trajectory(start)
            ref_source = "network"

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
            source=_FixedReferenceSource(reference),
            terminal=TerminalChecker(restored.tol_pos, restored.tol_yaw),
            env=restored.task.scene.env,
            replan_every=replan_every,
            max_steps=replan_every,
            collision_checker=actual_collision_checker,
            **vehicle.collision_kwargs(),
        )
        result = engine.run(start, restored.goal)
        ref_pts = np.asarray(reference.points, dtype=np.float64)
        ref_seg = np.diff(ref_pts[:, :2], axis=0)
        ref_length = float(np.sum(np.hypot(ref_seg[:, 0], ref_seg[:, 1]))) if ref_seg.shape[0] else 0.0
        # 周期末状态对参考轨迹的最近点偏差（跟踪到位程度，而非距目标）。
        end_offset_ref = 0.0
        if result.record is not None and result.record.states:
            final_state = result.record.states[-1]
            d = np.hypot(ref_pts[:, 0] - final_state.x, ref_pts[:, 1] - final_state.y)
            end_offset_ref = round(float(d.min()), 3)
        difficulty = metadata[index]["difficulty"]
        rows.append(
            {
                "index": index,
                "task_id": restored.task.task_id,
                "scene": restored.task.scene_name,
                "task_type": restored.task.task_type.value,
                "maneuver": difficulty["maneuver"],
                "noise_level": difficulty["noise_level"],
                "source": ref_source,
                "tracking_rms_m": round(result.tracking_rms, 4),
                "steps": result.steps,
                "progress_m": round(result.path_length, 3),
                "ref_length_m": round(ref_length, 3),
                "reach_ratio": round(result.path_length / ref_length, 3) if ref_length > 0 else 0.0,
                "end_offset_ref_m": end_offset_ref,
                "collision": bool(result.collision),
            }
        )
        print(
            f"[{ordinal}/{len(indices)}] {restored.task.task_id} "
            f"RMS={result.tracking_rms:.3f}m reach={rows[-1]['reach_ratio']:.2f} "
            f"end_ref={rows[-1]['end_offset_ref_m']:.2f}m",
            flush=True,
        )

    overall = _aggregate(rows)
    report = {
        "schema_version": 1,
        "task": "L3_single_cycle_tracking",
        "data": str(data_source),
        "checkpoint": str(Path(checkpoint_path).resolve()) if source == "network" else None,
        "source": source,
        "selection": selection,
        "requested_samples": samples,
        "selected_indices": indices,
        "replan_every": replan_every,
        "control_seed": control_seed,
        "vehicle_model": vehicle.to_metadata(),
        "overall": overall,
        "samples": rows,
    }
    if output_path is not None:
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination, report)
    return report


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    tracking = [r["tracking_rms_m"] for r in rows]
    reach = [r["reach_ratio"] for r in rows]
    end_off = [r["end_offset_ref_m"] for r in rows]
    collisions = [r["collision"] for r in rows]
    return {
        "samples": len(rows),
        "tracking_rms_mean": round(float(np.mean(tracking)), 4),
        "tracking_rms_std": round(float(np.std(tracking)), 4),
        "reach_ratio_mean": round(float(np.mean(reach)), 3),
        "end_offset_ref_mean": round(float(np.mean(end_off)), 3),
        "collision_rate": round(float(np.mean(collisions)), 3),
        "tracking_rms_p90": round(float(np.percentile(tracking, 90)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="schema v2 数据集 NPZ")
    parser.add_argument("--model", default="", help="network 源时提供 deployment checkpoint")
    parser.add_argument("--samples", type=int, default=0, help="<=0 表示全部")
    parser.add_argument("--selection", choices=["stratified", "head"], default="stratified")
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["expert", "network"], default="expert")
    parser.add_argument("--output", default="runs/single-cycle/L3/report.json")
    args = parser.parse_args()
    if args.source == "network" and not args.model:
        parser.error("network 源要求 --model")
    report = run_single_cycle_tracking(
        args.data,
        args.model,
        output_path=args.output,
        samples=args.samples,
        selection=args.selection,
        replan_every=args.replan_every,
        control_seed=args.seed,
        source=args.source,
    )
    print(f"报告：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()