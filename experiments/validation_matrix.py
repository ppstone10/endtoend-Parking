"""闭环五组对照实验统一 runner（E1–E5）。

按 ``docs/closed_loop_five_experiment_plan.md`` 的矩阵，把闭环链路拆成四个可独立
替换的环节，并对任意组合跑同一套协议与指标：

- ``data_source``：数据与任务几何来源（当前为 schema v2 数据集任务复原）；
- ``bev_source``：感知来源 —— 传感器 BEV（点云+图像）或 GT BEV（几何真值）；
- ``trajectory_source``：轨迹生成 —— 传统规划（Hybrid A*）或端到端网络；
- ``executor``：执行 —— MPC + 仿真矿卡，或理想执行（按预测轨迹推进）。

五组实验的对应关系：

====  ==================  ==================  ================  ==================
实验   bev_source          trajectory_source   executor          说明
====  ==================  ==================  ================  ==================
E1    gt                  expert              mpc_vehicle      控制与车辆模型地基
E2    gt                  network             ideal_path       网络自持滚动（最重要）
E3    gt                  network             mpc_vehicle      网络轨迹可执行性
E4    sensor              expert              mpc_vehicle      BEV 误差影响
E5    sensor              network             mpc_vehicle      最终系统
====  ==================  ==================  ================  ==================

另提供 ``bev_source=sensor`` + ``trajectory_source=network`` + ``executor=ideal_path``
（E2a）作为"生产感知 + 理想执行"的对照口径，用于区分"网络自身不收敛"与
"GT BEV 分布偏移"两种解释。

统一协议：同一份数据的同一批索引、同一 seed、同一 replan_every、同一 max_steps、
同一终止容差、同一车辆配置、同一碰撞检查器、同一指标函数；报告头部写出全部协议
字段以便跨实验核对。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from controller import MPCController
from dataset import GroundTruthBEVPipeline, build_task_components
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
    select_evaluation_indices,
)
from interfaces import GoalPose, Trajectory, VehicleState
from metrics import EpisodeResult, summarize
from metrics.rollout import CycleTracker, analyze_cycle_samples
from planner import RectangleFootprintCollisionChecker
from runtime import (
    ClosedLoopEngine,
    ExpertSource,
    FootprintTrajectorySafetyChecker,
    HierarchicalPlanningSource,
    IdealPathExecutor,
    NetworkSource,
    ReplanningExpertSource,
    SafetyShieldSource,
    TerminalChecker,
)
from sim import DifferentialDriveModel, VehicleConfig
from training.checkpoint import load_model_checkpoint
from training.data import validate_model_dataset
from training.reporting import atomic_write_json

VALID_BEV_SOURCES = ("sensor", "gt")
VALID_TRAJECTORY_SOURCES = ("expert", "expert_replan", "network")
VALID_EXECUTORS = ("mpc_vehicle", "ideal_path")
VALID_SAFETY_MODES = ("none", "expert_fallback", "hierarchical")


@dataclass(frozen=True)
class ExperimentSpec:
    """一组实验的环节组合。"""

    name: str
    bev_source: str
    trajectory_source: str
    executor: str
    safety_mode: str = "none"
    description: str = ""

    def validate(self) -> None:
        if self.bev_source not in VALID_BEV_SOURCES:
            raise ValueError(f"未知 bev_source：{self.bev_source}")
        if self.trajectory_source not in VALID_TRAJECTORY_SOURCES:
            raise ValueError(f"未知 trajectory_source：{self.trajectory_source}")
        if self.executor not in VALID_EXECUTORS:
            raise ValueError(f"未知 executor：{self.executor}")
        if self.safety_mode not in VALID_SAFETY_MODES:
            raise ValueError(f"未知 safety_mode：{self.safety_mode}")
        if self.trajectory_source in {"expert", "expert_replan"} and self.safety_mode != "none":
            raise ValueError("传统规划轨迹源不使用安全门禁/分层模式")
        if self.trajectory_source == "network" and self.executor == "ideal_path":
            return  # 理想执行口径只对网络轨迹有验证意义，但也允许专家参数化自检
        if self.trajectory_source in {"expert", "expert_replan"} and self.bev_source == "gt":
            raise ValueError("传统规划不消费 BEV，gt 与 sensor 对 E1/E4 无区别，请用 sensor")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bev_source": self.bev_source,
            "trajectory_source": self.trajectory_source,
            "executor": self.executor,
            "safety_mode": self.safety_mode,
            "description": self.description,
        }


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "E1": ExperimentSpec(
        name="E1",
        bev_source="sensor",
        trajectory_source="expert",
        executor="mpc_vehicle",
        description="GT 环境 → 传统规划 → MPC → 仿真矿卡",
    ),
    "E1r": ExperimentSpec(
        name="E1r",
        bev_source="sensor",
        trajectory_source="expert_replan",
        executor="mpc_vehicle",
        description="GT 环境 → 传统滚动重规划 → MPC → 仿真矿卡",
    ),
    "E2": ExperimentSpec(
        name="E2",
        bev_source="gt",
        trajectory_source="network",
        executor="ideal_path",
        description="GT BEV → NN → 理想执行 → 新状态 → NN",
    ),
    "E2a": ExperimentSpec(
        name="E2a",
        bev_source="sensor",
        trajectory_source="network",
        executor="ideal_path",
        description="传感器 BEV → NN → 理想执行（GT BEV 分布偏移对照口径）",
    ),
    "E3": ExperimentSpec(
        name="E3",
        bev_source="gt",
        trajectory_source="network",
        executor="mpc_vehicle",
        description="GT BEV → NN → MPC → 仿真矿卡",
    ),
    "E4": ExperimentSpec(
        name="E4",
        bev_source="sensor",
        trajectory_source="expert",
        executor="mpc_vehicle",
        description="点云+图像 → BEV → 传统规划 → MPC → 仿真矿卡",
    ),
    "E5": ExperimentSpec(
        name="E5",
        bev_source="sensor",
        trajectory_source="network",
        executor="mpc_vehicle",
        description="点云+图像 → BEV → NN → MPC → 仿真矿卡（最终系统）",
    ),
    "E5h": ExperimentSpec(
        name="E5h",
        bev_source="sensor",
        trajectory_source="network",
        executor="mpc_vehicle",
        safety_mode="hierarchical",
        description="E5 分层局部规划口径",
    ),
    "E5s": ExperimentSpec(
        name="E5s",
        bev_source="sensor",
        trajectory_source="network",
        executor="mpc_vehicle",
        safety_mode="expert_fallback",
        description="E5 安全门禁口径",
    ),
}


@dataclass
class EpisodeBundle:
    """单回合成品：指标 + 过程证据。"""

    result: EpisodeResult
    cycles: dict[str, Any] = field(default_factory=dict)
    cycle_rows: list[dict[str, Any]] = field(default_factory=list)
    executor_audit: dict[str, Any] = field(default_factory=dict)


def _control_steps_per_point(ideal_steps_per_point: float | None, horizon_dt: float, mpc_dt: float) -> float:
    """理想执行推进速度：默认与网络轨迹 dt 对齐。"""
    if ideal_steps_per_point is not None:
        if ideal_steps_per_point <= 0.0:
            raise ValueError("ideal_steps_per_point 必须为正")
        return float(ideal_steps_per_point)
    if horizon_dt > 0.0 and mpc_dt > 0.0:
        return max(horizon_dt / mpc_dt, 1.0)
    return 1.0


def _build_pipeline(spec: ExperimentSpec, task, vehicle: VehicleConfig):
    """按 bev_source 构造感知管道；返回 (planner, pipeline)。"""
    planner, sensor_pipeline = build_task_components(task, vehicle)
    if spec.bev_source == "sensor":
        return planner, sensor_pipeline
    gt_pipeline = GroundTruthBEVPipeline(
        task.scene.env,
        task.scene.bev_config,
        vehicle_length=vehicle.length,
        vehicle_width=vehicle.width,
    )
    return planner, gt_pipeline


def _build_source(
    spec: ExperimentSpec,
    planner,
    pipeline,
    model,
    *,
    hierarchical_lookahead: float,
):
    """按 trajectory_source 与 safety_mode 构造轨迹源。"""
    if spec.trajectory_source == "expert":
        return ExpertSource(planner)
    expert_source = ReplanningExpertSource(planner)
    if spec.trajectory_source == "expert_replan":
        return expert_source
    network_source = NetworkSource(pipeline, model)
    if spec.safety_mode == "expert_fallback":
        return SafetyShieldSource(
            network_source,
            expert_source,
            FootprintTrajectorySafetyChecker(planner._collision_checker),
        )
    if spec.safety_mode == "hierarchical":
        return HierarchicalPlanningSource(
            network_source,
            planner,
            lookahead=hierarchical_lookahead,
            safety_checker=FootprintTrajectorySafetyChecker(planner._collision_checker),
        )
    return network_source


def _cycles_from_record(
    result: EpisodeResult, tracker: CycleTracker, *, reference_dt: float
) -> None:
    """用引擎逐步记录重建重规划周期序列。

    引擎的 ``EpisodeRecord`` 已逐步保存状态与当前参考轨迹快照，重规划点即
    "参考轨迹快照发生变化"的步；据此把过程证据按周期切开，不需要引擎改动。
    """
    record = result.record
    if record is None or not record.states:
        return
    previous_plan: np.ndarray | None = None
    cumulative = 0.0
    previous_xy: tuple[float, float] | None = None
    for step, (state, plan) in enumerate(zip(record.states, record.plans)):
        if previous_xy is not None:
            cumulative += float(
                np.hypot(state.x - previous_xy[0], state.y - previous_xy[1])
            )
        previous_xy = (state.x, state.y)
        changed = (
            previous_plan is None
            or plan.shape != previous_plan.shape
            or not np.array_equal(plan, previous_plan)
        )
        if changed:
            tracker.start_cycle(
                state, Trajectory(np.array(plan, copy=True), dt=reference_dt)
            )
            previous_plan = plan
        tracker.sample(state, path_length=cumulative, time_s=0.1 * step)


def run_validation_experiment(
    spec: ExperimentSpec | str,
    *,
    data_path: str | Path,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    samples: int = 0,
    selection: str = "stratified",
    max_steps: int = 600,
    replan_every: int = 10,
    control_seed: int = 0,
    indices: list[int] | None = None,
    ideal_steps_per_point: float | None = None,
    ideal_point_spacing_m: float | None = None,
    hierarchical_lookahead: float = 3.0,
    progress: Callable[[int, int, EpisodeBundle], None] | None = None,
) -> dict[str, Any]:
    """按实验组合执行闭环评测并返回报告。"""
    if isinstance(spec, str):
        if spec not in EXPERIMENT_SPECS:
            raise ValueError(f"未知实验组合：{spec}；可选 {sorted(EXPERIMENT_SPECS)}")
        spec = EXPERIMENT_SPECS[spec]
    spec.validate()
    if max_steps <= 0 or replan_every <= 0:
        raise ValueError("max_steps 与 replan_every 必须为正")

    data_source = Path(data_path).resolve()
    from dataset import DatasetGenerator

    data = DatasetGenerator.load(data_source)
    metadata = data.get("task_meta")
    if int(data.get("schema_version", -1)) != 2 or not isinstance(metadata, list):
        raise ValueError("验证矩阵要求 schema v2 数据集与 task_meta")
    manifest = load_dataset_manifest(data_source)
    try:
        vehicle = VehicleConfig(**manifest["vehicle_model"])
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest vehicle_model 无效") from exc

    needs_model = spec.trajectory_source == "network"
    loaded = None
    if needs_model:
        if not checkpoint_path:
            raise ValueError("network 轨迹源要求提供 checkpoint")
        loaded = load_model_checkpoint(Path(checkpoint_path).resolve())
        validate_model_dataset(loaded.model, data)
    elif checkpoint_path:
        loaded = load_model_checkpoint(Path(checkpoint_path).resolve())

    if indices is None:
        selected = select_evaluation_indices(metadata, samples=samples, strategy=selection)
    else:
        selected = [int(index) for index in indices]
    if not selected:
        raise ValueError("没有可评测样本")

    horizon_dt = float(loaded.model.dt) if loaded is not None and hasattr(loaded.model, "dt") else 0.0
    mpc_dt = 0.1
    ideal_factor = _control_steps_per_point(ideal_steps_per_point, horizon_dt, mpc_dt)

    started = time.perf_counter()
    bundles: list[EpisodeBundle] = []
    reconstruct_failures: list[dict[str, Any]] = []
    for ordinal, index in enumerate(selected, start=1):
        try:
            restored = reconstruct_dataset_task(
                metadata[index], root_seed=int(manifest["seed"]), vehicle=vehicle
            )
        except ValueError as exc:
            reconstruct_failures.append({"dataset_index": index, "reason": str(exc)})
            continue
        state = VehicleState.from_array(np.asarray(data["states"])[index])
        goal = restored.goal
        planner, pipeline = _build_pipeline(spec, restored.task, vehicle)
        source = _build_source(
            spec,
            planner,
            pipeline,
            loaded.model if loaded is not None else None,
            hierarchical_lookahead=hierarchical_lookahead,
        )
        collision_checker = RectangleFootprintCollisionChecker(
            restored.task.scene.env,
            vehicle_length=vehicle.length,
            vehicle_width=vehicle.width,
            collision_margin=0.0,
            resolution=vehicle.collision_check_resolution,
        )
        mpc = MPCController(
            dt=mpc_dt,
            horizon=10,
            seed=control_seed + index,
            **vehicle.mpc_kwargs(),
        )
        executor = (
            IdealPathExecutor(
                control_steps_per_point=ideal_factor,
                point_spacing_m=ideal_point_spacing_m,
            )
            if spec.executor == "ideal_path"
            else None
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
            "spot_id": restored.goal_meta["spot_id"],
            "tol_pos": restored.tol_pos,
            "tol_yaw": restored.tol_yaw,
            "start_x": float(state.x),
            "start_y": float(state.y),
            "goal_x": float(restored.goal.x),
            "goal_y": float(restored.goal.y),
            "experiment": spec.name,
        }
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
            mpc=mpc,
            source=source,
            terminal=TerminalChecker(restored.tol_pos, restored.tol_yaw),
            env=restored.task.scene.env,
            replan_every=replan_every,
            max_steps=max_steps,
            meta=episode_meta,
            collision_checker=collision_checker,
            executor=executor,
            **vehicle.collision_kwargs(),
        )
        result = engine.run(state, goal)
        tracker = CycleTracker()
        tracker.begin(goal, state)
        _cycles_from_record(result, tracker, reference_dt=horizon_dt)
        cycle_report = analyze_cycle_samples(tracker)
        result.record = None
        bundles.append(
            EpisodeBundle(
                result=result,
                cycles=cycle_report,
                cycle_rows=tracker.cycles(),
                executor_audit=engine.executor_audit(),
            )
        )
        if progress is not None:
            progress(ordinal, len(selected), bundles[-1])

    if not bundles:
        raise ValueError(
            f"全部样本都未能复原任务几何（{len(reconstruct_failures)} 条失败）："
            "当前代码几何与数据集身份不一致"
        )

    results = [bundle.result for bundle in bundles]
    overall = summarize(results)
    overall["elapsed_sec"] = time.perf_counter() - started
    overall["evaluated_samples"] = len(bundles)
    overall["requested_indices"] = len(selected)
    overall["reconstruct_failures"] = len(reconstruct_failures)
    overall.update(_aggregate_cycles(bundles))
    if spec.executor == "ideal_path":
        overall.update(_aggregate_ideal_executor(bundles))

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "experiment": spec.to_metadata(),
        "protocol": {
            "data": str(data_source),
            "checkpoint": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
            "model_name": loaded.model_name if loaded is not None else None,
            "model_config": loaded.model_config if loaded is not None else None,
            "selection": selection,
            "requested_samples": samples,
            "selected_indices": selected,
            "evaluated_indices": [bundle.result.meta["dataset_index"] for bundle in bundles],
            "max_steps": max_steps,
            "replan_every": replan_every,
            "control_seed": control_seed,
            "mpc_dt": mpc_dt,
            "ideal_steps_per_point": ideal_factor if spec.executor == "ideal_path" else None,
            "ideal_point_spacing_m": (
                IdealPathExecutor.DEFAULT_POINT_SPACING_M
                if spec.executor == "ideal_path" and ideal_point_spacing_m is None
                else (ideal_point_spacing_m if spec.executor == "ideal_path" else None)
            ),
            "hierarchical_lookahead": (
                hierarchical_lookahead if spec.safety_mode == "hierarchical" else None
            ),
        },
        "vehicle_model": vehicle.to_metadata(),
        "overall": overall,
        "groups": _group_summaries(bundles),
        "episodes": [_episode_payload(bundle) for bundle in bundles],
        "reconstruct_failures": reconstruct_failures,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path).resolve(), report)
    return report


def _aggregate_cycles(bundles: list[EpisodeBundle]) -> dict[str, Any]:
    """跨样本聚合过程级滚动指标。"""
    keys_float = (
        "divergence_events",
        "goal_approach_monotonic_ratio",
        "drift_slope",
        "pred_dist_to_traj_m",
        "d_goal_end_m",
        "heading_divergence_events",
        "heading_approach_monotonic_ratio",
        "heading_drift_slope_deg",
        "yaw_err_end_deg",
    )
    aggregated: dict[str, Any] = {}
    for key in keys_float:
        values = [
            bundle.cycles[key]
            for bundle in bundles
            if bundle.cycles.get(key) is not None
        ]
        if values:
            array = np.asarray(values, dtype=np.float64)
            aggregated[f"cycles_{key}_mean"] = float(array.mean())
            aggregated[f"cycles_{key}_std"] = float(array.std())
    diverging = sum(1 for b in bundles if b.cycles.get("divergence_events", 0) > 0)
    aggregated["cycles_samples_with_divergence"] = diverging
    aggregated["cycles_divergence_rate"] = (
        diverging / len(bundles) if bundles else 0.0
    )
    slopes = [b.cycles["drift_slope"] for b in bundles if b.cycles.get("drift_slope") is not None]
    aggregated["cycles_samples_with_positive_drift"] = sum(1 for s in slopes if s > 0.0)
    return aggregated


def _aggregate_ideal_executor(bundles: list[EpisodeBundle]) -> dict[str, Any]:
    """理想执行器的聚合诊断（前进失败次数等）。"""
    failures = sum(int(b.executor_audit.get("advance_failures", 0)) for b in bundles)
    return {"ideal_executor_advance_failures": failures}


def _episode_payload(bundle: EpisodeBundle) -> dict[str, Any]:
    payload = bundle.result.to_dict()
    payload["final_yaw_err_deg"] = float(np.degrees(bundle.result.final_yaw_err))
    payload["cycles"] = bundle.cycles
    payload["cycle_rows"] = bundle.cycle_rows
    payload["executor_audit"] = bundle.executor_audit
    return payload


def _group_summaries(bundles: list[EpisodeBundle]) -> dict[str, dict[str, dict]]:
    dimensions = {
        "scene": lambda meta: meta["scene_name"],
        "task_type": lambda meta: meta["task_type"],
        "maneuver": lambda meta: meta["maneuver"],
        "noise_level": lambda meta: meta["noise_level"],
        "adjacent_occupancy": lambda meta: str(meta["adjacent_occupancy"]),
    }
    grouped: dict[str, dict[str, dict]] = {}
    for dimension, key_fn in dimensions.items():
        buckets: dict[str, list[EpisodeBundle]] = defaultdict(list)
        for bundle in bundles:
            buckets[str(key_fn(bundle.result.meta))].append(bundle)
        grouped[dimension] = {
            key: {
                **summarize([b.result for b in values]),
                **_aggregate_cycles(values),
                "success_count": sum(1 for b in values if b.result.success),
                "failure_counts": _failure_counts(values),
            }
            for key, values in sorted(buckets.items())
        }
    return grouped


def _failure_counts(bundles: list[EpisodeBundle]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bundle in bundles:
        if bundle.result.failure is not None:
            counts[bundle.result.failure] = counts.get(bundle.result.failure, 0) + 1
    return dict(sorted(counts.items()))
