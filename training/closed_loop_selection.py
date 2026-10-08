"""训练期闭环选型：用"闭环成功率"而非开环验证损失挑选模型。

动机（本项目的确定性证据）：开环 ADE/FDE 与闭环成功率严重脱节——
实测五次训练的开环 ADE 只落在 0.289–0.372m（跨度 29%），闭环成功率却跨越 **0%–35.3%**；
开环最好的一次闭环仅 8.1%。因此"按 val_loss 选 best"会稳定选错模型。

做法：在训练循环里周期性用**当前权重**跑一小批真实闭环回合（NN → MPC → 车辆模型），
以闭环成功率为选型指标，另存 `best_closed_loop.pt`，并由调用方决定是否用它做部署与早停。

安全边界：评估前 ``deepcopy`` 模型，评估在副本上进行（含 ``stop_threshold`` 等属性改写），
训练器持有的权重与优化器状态完全不受影响。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from controller import MPCController
from dataset import DatasetGenerator, build_task_components
from experiments.closed_loop_evaluation import (
    load_dataset_manifest,
    reconstruct_dataset_task,
)
from interfaces import VehicleState
from metrics import summarize
from planner import RectangleFootprintCollisionChecker
from runtime import (
    ClosedLoopEngine,
    FootprintTrajectorySafetyChecker,
    HierarchicalPlanningSource,
    NetworkSource,
    ReplanningExpertSource,
    SafetyShieldSource,
    TerminalChecker,
)
from sim import DifferentialDriveModel, VehicleConfig

from .checkpoint import load_model_checkpoint

__all__ = ["ClosedLoopSelectionConfig", "ClosedLoopSelector"]


@dataclass(frozen=True)
class ClosedLoopSelectionConfig:
    """闭环选型配置（训练 YAML 的可选 `closed_loop_selection` 段）。"""

    enabled: bool = False
    #: 用于闭环评估的数据分片；缺省时用训练配置里的 val 分片。
    data: str = ""
    #: 每多少 epoch 评估一次（1 = 每轮）。
    every_epochs: int = 2
    #: 评估样本数（按场景分层取前 N 条；<=0 表示全部）。
    samples: int = 60
    #: 闭环失败后不再提升的容忍轮数；<=0 表示不据此早停。
    patience: int = 6
    max_steps: int = 600
    replan_every: int = 10
    control_seed: int = 0
    safety_mode: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "data": self.data,
            "every_epochs": self.every_epochs,
            "samples": self.samples,
            "patience": self.patience,
            "max_steps": self.max_steps,
            "replan_every": self.replan_every,
            "control_seed": self.control_seed,
            "safety_mode": self.safety_mode,
        }

    def __post_init__(self) -> None:
        if self.every_epochs < 1:
            raise ValueError("closed_loop_selection.every_epochs 必须为正")
        if self.samples < 0:
            raise ValueError("closed_loop_selection.samples 不能为负")
        if self.max_steps <= 0 or self.replan_every <= 0:
            raise ValueError("closed_loop_selection 的 max_steps/replan_every 必须为正")
        if self.safety_mode not in {"none", "expert_fallback", "hierarchical"}:
            raise ValueError(
                "closed_loop_selection.safety_mode 必须是 none/expert_fallback/hierarchical"
            )


@dataclass(frozen=True)
class _Episode:
    """一条已复原的闭环评估任务。"""

    index: int
    state: VehicleState
    goal: Any
    task: Any
    goal_meta: dict[str, Any]
    tol_pos: float
    tol_yaw: float


class ClosedLoopSelector:
    """按固定任务集合用当前权重做闭环评估，返回选型分数。"""

    def __init__(
        self,
        config: ClosedLoopSelectionConfig,
        *,
        data_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.data_path = Path(data_path).resolve()
        self.device = device
        self._episodes: list[_Episode] | None = None
        self.vehicle: VehicleConfig | None = None
        self.selected_indices: list[int] = []

    # ------------------------------------------------------------------
    # 任务集合：与训练无关，初始化一次后复用，保证逐轮可比
    # ------------------------------------------------------------------
    def _ensure_episodes(self) -> list[_Episode]:
        if self._episodes is not None:
            return self._episodes
        data = DatasetGenerator.load(self.data_path)
        metadata = data.get("task_meta")
        if int(data.get("schema_version", -1)) != 2 or not isinstance(metadata, list):
            raise ValueError("闭环选型要求 schema v2 数据集与 task_meta")
        manifest = load_dataset_manifest(self.data_path)
        vehicle = VehicleConfig(**manifest["vehicle_model"])
        self.vehicle = vehicle
        states = np.asarray(data["states"])

        indices = list(range(len(metadata)))
        if self.config.samples > 0:
            indices = _stratified_indices(metadata, indices, self.config.samples)
        episodes: list[_Episode] = []
        skipped: list[int] = []
        for index in indices:
            try:
                restored = reconstruct_dataset_task(
                    metadata[index], root_seed=int(manifest["seed"]), vehicle=vehicle
                )
            except ValueError:
                skipped.append(index)
                continue
            episodes.append(
                _Episode(
                    index=index,
                    state=VehicleState.from_array(states[index]),
                    goal=restored.goal,
                    task=restored.task,
                    goal_meta=restored.goal_meta,
                    tol_pos=restored.tol_pos,
                    tol_yaw=restored.tol_yaw,
                )
            )
        if not episodes:
            raise ValueError(
                f"闭环选型在 {self.data_path} 上取不到可复原样本（跳过 {len(skipped)} 条）"
            )
        self.selected_indices = [episode.index for episode in episodes]
        self._episodes = episodes
        return episodes

    def evaluate(self, model) -> dict[str, Any]:
        """用给定模型的**副本**跑闭环，返回选型分数与附带指标。"""
        episodes = self._ensure_episodes()
        assert self.vehicle is not None
        vehicle = self.vehicle
        working = copy.deepcopy(model)
        working.eval()
        try:
            working.to(self.device)
        except (AttributeError, RuntimeError):
            pass

        results = []
        planning_failures = 0
        for episode in episodes:
            planner, pipeline = build_task_components(
                episode.task, vehicle, model_config=getattr(model, "model_config", None)
            )
            source, expert_source = _build_source(self.config.safety_mode, pipeline, planner, working)
            collision_checker = RectangleFootprintCollisionChecker(
                episode.task.scene.env,
                vehicle_length=vehicle.length,
                vehicle_width=vehicle.width,
                collision_margin=0.0,
                resolution=vehicle.collision_check_resolution,
            )
            engine = ClosedLoopEngine(
                vehicle_model=DifferentialDriveModel(**vehicle.vehicle_model_kwargs()),
                mpc=MPCController(
                    dt=0.1, horizon=10, seed=self.config.control_seed, **vehicle.mpc_kwargs()
                ),
                source=source,
                terminal=TerminalChecker(episode.tol_pos, episode.tol_yaw),
                env=episode.task.scene.env,
                replan_every=self.config.replan_every,
                max_steps=self.config.max_steps,
                meta=dict(episode.goal_meta, dataset_index=episode.index),
                collision_checker=collision_checker,
                **vehicle.collision_kwargs(),
            )
            result = engine.run(episode.state, episode.goal)
            result.record = None
            if result.meta.get("planning_failure"):
                planning_failures += 1
            results.append(result)

        summary = summarize(results)
        return {
            "score": float(summary["success_rate"]),
            "samples": summary["episodes"],
            "success_rate": float(summary["success_rate"]),
            "collision_rate": float(summary["collision_rate"]),
            "final_pos_err_mean": float(summary["final_pos_err_mean"]),
            "final_yaw_err_mean": float(summary["final_yaw_err_mean"]),
            "tracking_rms_mean": float(summary["tracking_rms_mean"]),
            "failure_counts": dict(summary.get("failures") or {}),
            "planning_failures": planning_failures,
        }


def _stratified_indices(metadata, indices: list[int], samples: int) -> list[int]:
    """按「场景×任务类型」轮询取前 N 条，保证各单元都有代表。"""
    groups: dict[tuple[str, str], list[int]] = {}
    for index in indices:
        item = metadata[index]
        key = (str(item.get("scene_name")), str(item.get("task_type")))
        groups.setdefault(key, []).append(index)
    selected: list[int] = []
    round_index = 0
    while len(selected) < samples:
        added = False
        for key in sorted(groups):
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


def _build_source(safety_mode: str, pipeline, planner, model):
    """按安全模式构造轨迹源；显式传入模型，避免依赖 checkpoint 文件。"""
    network_source = NetworkSource(pipeline, model)
    if safety_mode == "none":
        return network_source, None
    expert_source = ReplanningExpertSource(planner)
    if safety_mode == "expert_fallback":
        return (
            SafetyShieldSource(
                network_source,
                expert_source,
                FootprintTrajectorySafetyChecker(planner._collision_checker),
            ),
            expert_source,
        )
    return (
        HierarchicalPlanningSource(
            network_source,
            planner,
            safety_checker=FootprintTrajectorySafetyChecker(planner._collision_checker),
        ),
        expert_source,
    )


def load_contract_from_checkpoint(checkpoint_path: str | Path) -> dict:
    """读取 checkpoint 的输入契约（供外部对齐感知链路时使用）。"""
    return dict(load_model_checkpoint(checkpoint_path).model_config)
