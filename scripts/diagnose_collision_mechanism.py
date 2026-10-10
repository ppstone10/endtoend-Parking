"""诊断纯网络闭环碰撞的机制：参考轨迹穿障 vs 执行偏离安全参考。

纯网络口径下每 replan_every 个控制周期由 ``NetworkSource`` 输出一条全局参考轨迹，
MPC 跟踪它。碰撞可能来自两条完全不同的路径：

1. **参考不可行**：网络输出的轨迹本身穿过障碍（或与障碍的间隙小于规划余量），
   MPC 忠实跟踪就会撞。这类碰撞可以被"轨迹级几何过滤"直接拦住。
2. **执行偏离**：参考轨迹是可行的，但 MPC 跟踪误差/车辆动力学把实际状态推离参考，
   实际状态进入障碍。这类碰撞过滤参考轨迹无效，必须在状态转移层拦截。

两类碰撞需要完全不同的对策，因此本脚本先量化二者的占比，再决定过滤器的落点。

用法：

    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/diagnose_collision_mechanism.py \
        --experiment E3 --indices-file runs/validation/v13/indices-shared-scenes.json \
        --output runs/diagnostics/collision-mechanism-E3.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator
from experiments import validation_matrix
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
)
from planner import RectangleFootprintCollisionChecker
from runtime.engine import ClosedLoopEngine
from runtime.recorder import EpisodeRecord
from sim import VehicleConfig
from training.reporting import atomic_write_json

CAPTURED: dict[int, EpisodeRecord] = {}


class _CapturingEngine(ClosedLoopEngine):
    """在引擎交出 EpisodeResult 前扣下逐步记录，供事后机制分析。"""

    def _build_result(self, state, goal, traj, record, inference_times, collision,
                      safety_stop=False, planning_failure=None):
        CAPTURED[int(self.meta.get("dataset_index", -1))] = record
        return super()._build_result(
            state, goal, traj, record, inference_times, collision, safety_stop,
            planning_failure,
        )


def _make_checker(env, vehicle: VehicleConfig, margin: float):
    return RectangleFootprintCollisionChecker(
        env,
        vehicle_length=vehicle.length,
        vehicle_width=vehicle.width,
        collision_margin=margin,
        resolution=vehicle.collision_check_resolution,
    )


def _first_blocked_index(checker, start_pose, points) -> int | None:
    """从 start_pose 起沿 points 逐点扫掠，返回第一个被判定为不安全的点下标。

    返回 None 表示整条轨迹（含从起点到首点的扫掠段）都安全。
    """
    previous = np.asarray(start_pose, dtype=np.float64)
    for index, point in enumerate(points):
        if not checker.swept_segment_free(previous, point):
            return index
        previous = point
    return None


def _path_arc(points) -> np.ndarray:
    if points.shape[0] < 2:
        return np.zeros(points.shape[0], dtype=np.float64)
    segments = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    return np.concatenate([[0.0], np.cumsum(segments)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/task_dataset/tracked_pivot_v7_3000/val.npz")
    parser.add_argument(
        "--model", default="runs/training/v26-targeted-dagger/net-v1/deployment.pt"
    )
    parser.add_argument("--experiment", default="E3")
    parser.add_argument(
        "--indices-file",
        default="runs/validation/v13/indices-shared-scenes.json",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--filter-margin",
        type=float,
        default=None,
        help="加装轨迹级几何过滤时要求的净空（米）；不给则按 --experiment 的原始口径跑",
    )
    parser.add_argument("--filter-max-offset", type=float, default=None)
    parser.add_argument("--filter-max-rounds", type=int, default=None)
    parser.add_argument(
        "--dump-blocked-plans",
        default="",
        help="把每条被判定不可行的重规划参考（发出位姿 + 轨迹点）落盘，供离线调参",
    )
    parser.add_argument("--output", default="runs/diagnostics/collision-mechanism.json")
    args = parser.parse_args()

    from scripts.run_validation_suite import _load_indices

    indices = _load_indices(args.indices_file)
    validation_matrix.ClosedLoopEngine = _CapturingEngine

    filter_kwargs: dict[str, Any] = {}
    if args.filter_margin is not None:
        filter_kwargs["filter_required_margin"] = args.filter_margin
    if args.filter_max_offset is not None:
        filter_kwargs["filter_max_offset"] = args.filter_max_offset
    if args.filter_max_rounds is not None:
        filter_kwargs["filter_max_rounds"] = args.filter_max_rounds

    report = validation_matrix.run_validation_experiment(
        args.experiment,
        data_path=args.data,
        checkpoint_path=args.model,
        samples=0,
        max_steps=args.max_steps,
        replan_every=args.replan_every,
        control_seed=args.seed,
        indices=indices,
        **filter_kwargs,
    )
    overall = report["overall"]
    print(
        f"重跑基线：{overall['evaluated_samples']} 条，成功 {overall['success_rate']:.1%}，"
        f"碰撞 {overall['collision_rate']:.1%}，失败 {overall['failures']}",
        flush=True,
    )

    data_path = Path(args.data).resolve()
    data = DatasetGenerator.load(data_path)
    manifest = load_dataset_manifest(data_path)
    vehicle = VehicleConfig(**manifest["vehicle_model"])

    episodes = report["episodes"]
    episode_meta = {int(e["meta"]["dataset_index"]): e["meta"] for e in episodes}
    collision_indices = [int(e["meta"]["dataset_index"]) for e in episodes if e["collision"]]

    # 逐回合需要的探针：参考轨迹在**发出时刻**是否可行、可行前缀有多长。
    # 参考轨迹的历史在 record.plans 里逐步快照，重规划点即相邻快照发生变化之处。
    per_episode: list[dict[str, Any]] = []
    plan_checks: Counter = Counter()
    collision_mech: Counter = Counter()
    safe_prefix_margin02: list[float] = []
    safe_prefix_margin00: list[float] = []
    collision_plan_arc_to_hit: list[float] = []
    deviation_at_collision: list[float] = []
    blocked_plans: list[dict[str, Any]] = []

    for index, meta in episode_meta.items():
        record = CAPTURED.get(index)
        if record is None or not record.states:
            continue
        restored = reconstruct_dataset_task(
            data["task_meta"][index], root_seed=int(manifest["seed"]), vehicle=vehicle
        )
        env = restored.task.scene.env
        checker02 = _make_checker(env, vehicle, 0.2)
        checker00 = _make_checker(env, vehicle, 0.0)

        # 重规划点 = 参考轨迹快照发生变化的步。
        replan_steps: list[int] = []
        previous_plan: np.ndarray | None = None
        for step, plan in enumerate(record.plans):
            if previous_plan is None or plan.shape != previous_plan.shape or not np.array_equal(plan, previous_plan):
                replan_steps.append(step)
                previous_plan = plan

        # 每个重规划点的参考可行性（与门禁同口径：margin=0.2）。
        plan_rows: list[dict[str, Any]] = []
        for step in replan_steps:
            plan = np.asarray(record.plans[step], dtype=np.float64)
            start = (
                np.asarray([record.states[step - 1].x, record.states[step - 1].y, record.states[step - 1].yaw])
                if step > 0
                else np.asarray([record.states[0].x, record.states[0].y, record.states[0].yaw])
            )
            blocked02 = _first_blocked_index(checker02, start, plan)
            blocked00 = _first_blocked_index(checker00, start, plan)
            if blocked02 is not None and args.dump_blocked_plans:
                blocked_plans.append(
                    {
                        "dataset_index": index,
                        "scene_name": meta["scene_name"],
                        "start_pose": [float(value) for value in start],
                        "points": plan.tolist(),
                    }
                )
            arc = _path_arc(plan)
            plan_checks["total"] += 1
            if blocked02 is not None:
                plan_checks["unsafe_margin02"] += 1
                safe_prefix_margin02.append(float(arc[blocked02]))
            if blocked00 is not None:
                plan_checks["unsafe_margin00"] += 1
                safe_prefix_margin00.append(float(arc[blocked00]))
            plan_rows.append(
                {
                    "step": step,
                    "n_points": int(plan.shape[0]),
                    "arc_total_m": float(arc[-1]) if arc.shape[0] else 0.0,
                    "blocked_index_margin02": blocked02,
                    "blocked_index_margin00": blocked00,
                    "safe_prefix_margin02_m": None if blocked02 is None else float(arc[blocked02]),
                    "safe_prefix_margin00_m": None if blocked00 is None else float(arc[blocked00]),
                }
            )

        collision_steps = [step for step, flag in enumerate(record.collisions) if flag]
        collision_row: dict[str, Any] | None = None
        if collision_steps:
            hit_step = collision_steps[0]
            # 生效中的参考 = 最近一次不晚于 hit_step 的重规划。
            effective = max((step for step in replan_steps if step <= hit_step), default=0)
            plan = np.asarray(record.plans[effective], dtype=np.float64)
            arc = _path_arc(plan)
            # 发出该参考时的车辆状态：重规划发生在该步推进之前，即上一步末状态。
            issue_state = record.states[effective - 1] if effective > 0 else record.states[0]
            blocked02 = _first_blocked_index(
                checker02,
                [issue_state.x, issue_state.y, issue_state.yaw],
                plan,
            )
            blocked00 = _first_blocked_index(
                checker00,
                [issue_state.x, issue_state.y, issue_state.yaw],
                plan,
            )
            deviation = float(
                np.min(np.hypot(plan[:, 0] - record.states[hit_step].x,
                                plan[:, 1] - record.states[hit_step].y))
            )
            deviation_at_collision.append(deviation)
            if blocked00 is not None:
                collision_mech["reference_unsafe_margin00"] += 1
                collision_plan_arc_to_hit.append(float(arc[blocked00]))
            elif blocked02 is not None:
                collision_mech["reference_unsafe_margin02_only"] += 1
                collision_plan_arc_to_hit.append(float(arc[blocked02]))
            else:
                collision_mech["reference_safe_execution_drift"] += 1
            collision_row = {
                "hit_step": hit_step,
                "effective_replan_step": effective,
                "plan_points": int(plan.shape[0]),
                "plan_arc_total_m": float(arc[-1]) if arc.shape[0] else 0.0,
                "blocked_index_margin02": blocked02,
                "blocked_index_margin00": blocked00,
                "plan_arc_to_first_blocked_m": (
                    None
                    if (blocked00 if blocked00 is not None else blocked02) is None
                    else float(arc[blocked00 if blocked00 is not None else blocked02])
                ),
                "state_deviation_from_plan_m": deviation,
                "steps_since_replan": hit_step - effective,
            }

        per_episode.append(
            {
                "dataset_index": index,
                "scene_name": meta["scene_name"],
                "task_type": meta["task_type"],
                "collided": bool(collision_steps),
                "steps": len(record.states),
                "replan_steps": len(replan_steps),
                "unsafe_replans_margin02": sum(
                    1 for row in plan_rows if row["blocked_index_margin02"] is not None
                ),
                "unsafe_replans_margin00": sum(
                    1 for row in plan_rows if row["blocked_index_margin00"] is not None
                ),
                "collision": collision_row,
            }
        )

    def _stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
            "max": float(array.max()),
        }

    payload = {
        "schema_version": 1,
        "protocol": {
            "data": str(data_path),
            "model": str(Path(args.model).resolve()),
            "experiment": args.experiment,
            "indices_file": str(args.indices_file),
            "replan_every": args.replan_every,
            "max_steps": args.max_steps,
            "shield_checker_margin": 0.2,
            "evaluation_collision_margin": 0.0,
        },
        "overall": {
            "evaluated_samples": overall["evaluated_samples"],
            "success_rate": overall["success_rate"],
            "collision_rate": overall["collision_rate"],
            "failures": overall["failures"],
        },
        "trajectory_filter": overall.get("trajectory_filter"),
        "filter": {
            "required_margin_m": args.filter_margin,
            "max_offset_m": args.filter_max_offset,
            "max_rounds": args.filter_max_rounds,
        },
        "plan_check_summary": dict(plan_checks),
        "collision_mechanism": dict(collision_mech),
        "safe_prefix_margin02_m": _stats([value for value in safe_prefix_margin02]),
        "safe_prefix_margin00_m": _stats([value for value in safe_prefix_margin00]),
        "collision_plan_arc_to_first_blocked_m": _stats(collision_plan_arc_to_hit),
        "collision_state_deviation_from_plan_m": _stats(deviation_at_collision),
        "scene_collision_counts": dict(
            sorted(
                Counter(
                    row["scene_name"] for row in per_episode if row["collided"]
                ).items()
            )
        ),
        "episodes": per_episode,
    }
    atomic_write_json(Path(args.output).resolve(), payload)
    if args.dump_blocked_plans:
        atomic_write_json(
            Path(args.dump_blocked_plans).resolve(),
            {
                "schema_version": 1,
                "data": str(data_path),
                "model": str(Path(args.model).resolve()),
                "scene_envs": {
                    str(index): meta["scene_name"]
                    for index, meta in episode_meta.items()
                },
                "plans": blocked_plans,
            },
        )
        print(f"不可行参考落盘 {len(blocked_plans)} 条：{Path(args.dump_blocked_plans).resolve()}")

    print("== 参考轨迹可行性（每个重规划点一条，与门禁同口径 margin=0.2）==")
    total = plan_checks["total"] or 1
    print(
        f"  重规划点 {plan_checks['total']} 个：margin0.2 不安全 "
        f"{plan_checks['unsafe_margin02']} 个（{plan_checks['unsafe_margin02']/total:.1%}），"
        f"margin0.0 不安全 {plan_checks['unsafe_margin00']} 个"
        f"（{plan_checks['unsafe_margin00']/total:.1%}）"
    )
    print("== 碰撞机制分解 ==")
    for key, value in collision_mech.items():
        print(f"  {key}: {value}")
    print("== 参考轨迹安全前缀长度（米）==")
    print(f"  margin0.2: {payload['safe_prefix_margin02_m']}")
    print(f"  margin0.0: {payload['safe_prefix_margin00_m']}")
    print("== 碰撞时实际状态到参考轨迹的最近距离（米）==")
    print(f"  {payload['collision_state_deviation_from_plan_m']}")
    print(f"报告：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
