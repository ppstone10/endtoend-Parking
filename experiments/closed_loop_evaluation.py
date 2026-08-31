"""按 schema v2 数据集任务复原并执行网络闭环评测。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np

from controller import MPCController
from dataset import DatasetGenerator, build_task_components
from interfaces import GoalPose, VehicleState
from metrics import EpisodeResult, summarize
from planner import RectangleFootprintCollisionChecker
from runtime import (
    ClosedLoopEngine,
    FootprintTrajectorySafetyChecker,
    NetworkSource,
    ReplanningExpertSource,
    SafetyShieldSource,
    TerminalChecker,
)
from sim import (
    DifferentialDriveModel,
    Maneuver,
    NoiseLevel,
    Task,
    TaskSampler,
    TaskType,
    VehicleConfig,
)
from training.checkpoint import load_model_checkpoint
from training.data import validate_model_dataset
from training.reporting import atomic_write_json


@dataclass(frozen=True)
class ReconstructedDatasetTask:
    """从持久元数据确定性复原的闭环任务与已选目标。"""

    task: Task
    goal: GoalPose
    goal_meta: dict[str, Any]
    tol_pos: float
    tol_yaw: float


def _task_sample_index(metadata: dict[str, Any]) -> int:
    scene_name = str(metadata.get("scene_name", ""))
    task_type = str(metadata.get("task_type", ""))
    task_id = str(metadata.get("task_id", ""))
    pattern = rf"{re.escape(scene_name)}-{re.escape(task_type)}-(\d+)-[0-9a-fA-F]{{8}}"
    match = re.fullmatch(pattern, task_id)
    if match is None:
        raise ValueError(f"任务身份格式无效：{task_id}")
    return int(match.group(1))


def reconstruct_dataset_task(
    metadata: dict[str, Any],
    *,
    root_seed: int,
    vehicle: VehicleConfig,
) -> ReconstructedDatasetTask:
    """使用根 seed、样本序号与难度坐标复原原始任务场景。"""
    if not isinstance(metadata, dict) or int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("闭环评测要求 task metadata schema v1")
    difficulty = metadata.get("difficulty")
    dataset_meta = metadata.get("dataset")
    if not isinstance(difficulty, dict) or not isinstance(dataset_meta, dict):
        raise ValueError("任务元数据缺少 difficulty 或 dataset")
    if dataset_meta.get("vehicle_model") != vehicle.to_metadata():
        raise ValueError("任务车辆模型与 manifest 不一致")

    sample_index = _task_sample_index(metadata)
    sampler = TaskSampler(
        seed=root_seed,
        vehicle_length=vehicle.length,
        vehicle_width=vehicle.width,
        collision_margin=vehicle.collision_margin,
    )
    try:
        task = sampler.sample(
            str(metadata["scene_name"]),
            TaskType(str(metadata["task_type"])),
            sample_index,
            maneuver=Maneuver(str(difficulty["maneuver"])),
            adjacent_occupancy=int(difficulty["adjacent_occupancy"]),
            noise_level=NoiseLevel(str(difficulty["noise_level"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"任务身份无法复原：{metadata.get('task_id', '<unknown>')}") from exc

    regenerated = task.to_metadata()
    if regenerated != {key: metadata.get(key) for key in regenerated}:
        raise ValueError(f"任务身份复原漂移：{metadata.get('task_id', '<unknown>')}")

    goal_meta = dataset_meta.get("selected_goal")
    required_goal = {"spot_id", "x", "y", "yaw", "tol_pos", "tol_yaw", "kind"}
    if not isinstance(goal_meta, dict) or not required_goal.issubset(goal_meta):
        raise ValueError("任务元数据缺少完整 selected_goal")
    candidates = (task.goal,) if task.goal is not None else task.candidate_goals
    selected = next(
        (candidate for candidate in candidates if candidate.spot_id == goal_meta["spot_id"]),
        None,
    )
    if selected is None or selected.to_metadata() != goal_meta:
        raise ValueError("selected_goal 不属于复原任务或位姿已漂移")
    return ReconstructedDatasetTask(
        task=task,
        goal=selected.as_goal_pose(),
        goal_meta=dict(goal_meta),
        tol_pos=float(selected.tol_pos),
        tol_yaw=float(selected.tol_yaw),
    )


def load_dataset_manifest(data_path: Path) -> dict[str, Any]:
    manifest_path = data_path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取数据集 manifest：{manifest_path}") from exc
    if not isinstance(manifest, dict) or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("闭环评测要求 manifest schema v1")
    if not isinstance(manifest.get("vehicle_model"), dict):
        raise ValueError("manifest 缺少 vehicle_model")
    return manifest


def select_evaluation_indices(
    metadata: list[dict[str, Any]],
    *,
    samples: int,
    strategy: str = "stratified",
) -> list[int]:
    """选择评测索引；samples<=0 表示全部，stratified 按场景×任务轮询。"""
    total = len(metadata)
    if samples <= 0 or samples >= total:
        return list(range(total))
    if strategy == "head":
        return list(range(samples))
    if strategy != "stratified":
        raise ValueError("selection strategy 必须为 stratified 或 head")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[(str(item.get("scene_name")), str(item.get("task_type")))].append(index)
    selected: list[int] = []
    round_index = 0
    ordered_keys = sorted(groups)
    while len(selected) < samples:
        added = False
        for key in ordered_keys:
            values = groups[key]
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) == samples:
                    break
        if not added:
            break
        round_index += 1
    return sorted(selected)


def _episode_public(result: EpisodeResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload["final_yaw_err_deg"] = float(np.degrees(result.final_yaw_err))
    return payload


def _group_summaries(results: list[EpisodeResult]) -> dict[str, dict[str, dict]]:
    dimensions = {
        "scene": lambda result: result.meta["scene_name"],
        "task_type": lambda result: result.meta["task_type"],
        "maneuver": lambda result: result.meta["maneuver"],
        "noise_level": lambda result: result.meta["noise_level"],
        "adjacent_occupancy": lambda result: str(result.meta["adjacent_occupancy"]),
    }
    grouped: dict[str, dict[str, dict]] = {}
    for dimension, key_fn in dimensions.items():
        buckets: dict[str, list[EpisodeResult]] = defaultdict(list)
        for result in results:
            buckets[str(key_fn(result))].append(result)
        grouped[dimension] = {
            key: summarize(values) for key, values in sorted(buckets.items())
        }
    return grouped


def run_dataset_network_evaluation(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    samples: int = 0,
    selection: str = "stratified",
    max_steps: int = 600,
    replan_every: int = 10,
    control_seed: int = 0,
    progress: Callable[[int, int, EpisodeResult], None] | None = None,
    safety_mode: str = "none",
) -> dict[str, Any]:
    """在数据集原始场景中执行 deployment→MPC 网络闭环评测。"""
    if max_steps <= 0 or replan_every <= 0:
        raise ValueError("max_steps 与 replan_every 必须为正")
    if safety_mode not in {"none", "expert_fallback"}:
        raise ValueError("safety_mode 必须为 none 或 expert_fallback")
    data_source = Path(data_path).resolve()
    checkpoint_source = Path(checkpoint_path).resolve()
    data = DatasetGenerator.load(data_source)
    metadata = data.get("task_meta")
    if int(data.get("schema_version", -1)) != 2 or not isinstance(metadata, list):
        raise ValueError("网络闭环评测要求 schema v2 数据集与 task_meta")
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

    started = time.perf_counter()
    results: list[EpisodeResult] = []
    for ordinal, index in enumerate(indices, start=1):
        restored = reconstruct_dataset_task(
            metadata[index], root_seed=int(manifest["seed"]), vehicle=vehicle
        )
        state = VehicleState.from_array(np.asarray(data["states"])[index])
        stored_goal = np.asarray(data["goals"])[index]
        actual_goal = np.asarray([restored.goal.x, restored.goal.y, restored.goal.yaw])
        yaw_delta = np.arctan2(
            np.sin(stored_goal[2] - actual_goal[2]),
            np.cos(stored_goal[2] - actual_goal[2]),
        )
        if not np.allclose(stored_goal[:2], actual_goal[:2], atol=1e-7) or abs(yaw_delta) > 1e-7:
            raise ValueError(f"样本 {index} 的 goals 数组与 selected_goal 不一致")

        planner, pipeline = build_task_components(restored.task, vehicle)
        stored_bev = data["bev_meta"]
        if (
            not np.isclose(pipeline.bev_config.resolution, float(stored_bev["resolution"]))
            or list(pipeline.bev_config.extent) != list(stored_bev["extent"])
            or list(pipeline.bev_config.shape) != list(stored_bev["shape"][-2:])
        ):
            raise ValueError(f"样本 {index} 的场景 BEV 配置与数据集不一致")
        network_source = NetworkSource(pipeline, loaded.model)
        if safety_mode == "expert_fallback":
            source = SafetyShieldSource(
                network_source,
                ReplanningExpertSource(planner),
                FootprintTrajectorySafetyChecker(planner._collision_checker),
            )
        else:
            source = network_source
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
            "spot_id": restored.goal_meta["spot_id"],
            "tol_pos": restored.tol_pos,
            "tol_yaw": restored.tol_yaw,
            "dynamic_event_applied": False,
        }
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
            mpc=MPCController(
                dt=0.1,
                horizon=10,
                seed=control_seed + index,
                **vehicle.mpc_kwargs(),
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
        result = engine.run(state, restored.goal)
        result.record = None
        results.append(result)
        if progress is not None:
            progress(ordinal, len(indices), result)

    overall = summarize(results)
    overall["elapsed_sec"] = time.perf_counter() - started
    shield_stats = None
    if safety_mode == "expert_fallback":
        checks = sum(result.meta["safety_shield"]["checks"] for result in results)
        transition_checks = sum(
            result.meta["safety_shield"].get("transition_checks", 0)
            for result in results
        )
        interventions = sum(
            result.meta["safety_shield"]["interventions"] for result in results
        )
        fallback_failures = sum(
            result.meta["safety_shield"]["fallback_failures"] for result in results
        )
        prevented_transitions = sum(
            result.meta["safety_shield"].get("prevented_transitions", 0)
            for result in results
        )
        safety_stops = sum(
            result.meta["safety_shield"].get("safety_stops", 0)
            for result in results
        )
        reasons: dict[str, int] = defaultdict(int)
        for result in results:
            for reason, count in result.meta["safety_shield"]["reasons"].items():
                reasons[reason] += int(count)
        shield_stats = {
            "checks": checks,
            "transition_checks": transition_checks,
            "interventions": interventions,
            "intervention_rate": interventions / checks if checks else 0.0,
            "prevented_transitions": prevented_transitions,
            "transition_prevention_rate": (
                prevented_transitions / transition_checks
                if transition_checks
                else 0.0
            ),
            "fallback_failures": fallback_failures,
            "safety_stops": safety_stops,
            "reasons": dict(sorted(reasons.items())),
        }
    report = {
        "schema_version": 1,
        "status": "completed",
        "evaluation": {
            "data": str(data_source),
            "checkpoint": str(checkpoint_source),
            "model_name": loaded.model_name,
            "model_config": loaded.model_config,
            "checkpoint_epoch": loaded.epoch,
            "selection": selection,
            "requested_samples": samples,
            "selected_indices": indices,
            "max_steps": max_steps,
            "replan_every": replan_every,
            "control_seed": control_seed,
            "dynamic_events_applied": False,
            "safety_mode": safety_mode,
        },
        "vehicle_model": vehicle.to_metadata(),
        "overall": overall,
        "groups": _group_summaries(results),
        "episodes": [_episode_public(result) for result in results],
        "safety_shield": shield_stats,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path).resolve(), report)
    return report
