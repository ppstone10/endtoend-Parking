"""训练数据集生成与加载。

批量生成样本（BEV + 目标位姿 + 状态 → 专家轨迹）。样本坐标约定：
- BEV 为车辆中心局部系；
- goal 与 expert_trajectory 为全局坐标（世界系），网络侧按需转换。

落盘格式：单个 .npz 文件，键为各字段数组（见 save）。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState
from sim.noise import get_noise_profile
from sim.tasks import Task, TaskGoal

from .maneuver import (
    DEFAULT_MIN_REQUESTED_DISTANCE_RATIO,
    ManeuverAudit,
    audit_maneuver_consistency,
    validate_minimum_requested_distance_ratio,
)


class TaskGenerationError(RuntimeError):
    """某个 Task 无法生成专家样本，并保留稳定任务标识。"""

    def __init__(self, task_id: str, reason: str, *, code: str | None = None) -> None:
        self.task_id = task_id
        self.reason = reason
        self.code = reason if code is None else code
        super().__init__(f"任务 {task_id} 生成失败：{reason}")


@dataclass
class TrainingSample:
    """单条训练样本。"""

    bev: BEVTensor
    goal: GoalPose
    state: VehicleState
    expert_trajectory: Trajectory
    task_meta: dict[str, Any] | None = None


class DatasetGenerator:
    """训练样本生成器。

    可兼容旧式环境内随机采样，也可把 Task 逐项转换为专家轨迹和融合 BEV。
    """

    def __init__(
        self,
        env=None,
        planner=None,
        sensor_pipeline=None,
        seed: int = 0,
        min_distance: float = 3.0,
        max_distance: float = 12.0,
        *,
        component_factory: Callable[[Task], tuple[Any, Any]] | None = None,
        goal_selector: Callable[[Task, Any], TaskGoal] | None = None,
        enforce_maneuver_consistency: bool = True,
        minimum_requested_distance_ratio: float = (
            DEFAULT_MIN_REQUESTED_DISTANCE_RATIO
        ),
    ) -> None:
        self.env = env
        self.planner = planner
        # sensor_pipeline: 提供 capture_bev(x, y, yaw) -> BEVTensor 的适配器。
        self.sensor_pipeline = sensor_pipeline
        self.rng = np.random.default_rng(seed)
        # 泊车场景轨迹通常较短；限制起终点距离保证规划快速收敛。
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.component_factory = component_factory
        self.goal_selector = goal_selector
        self.enforce_maneuver_consistency = bool(enforce_maneuver_consistency)
        self.minimum_requested_distance_ratio = (
            validate_minimum_requested_distance_ratio(
                minimum_requested_distance_ratio
            )
        )

    def generate(self, count: int | Iterable[Task]) -> list[TrainingSample]:
        """生成随机样本，或按输入顺序把 Task 转为专家样本。"""
        if isinstance(count, (int, np.integer)):
            return self._generate_random(int(count))
        return [self._generate_task(task) for task in count]

    def _generate_random(self, count: int) -> list[TrainingSample]:
        """兼容旧入口：随机采样并跳过规划失败，直到凑够 count 条。"""
        if self.env is None or self.planner is None or self.sensor_pipeline is None:
            raise ValueError("随机生成要求 env、planner 与 sensor_pipeline")
        if count < 0:
            raise ValueError("样本数量不能为负")
        samples: list[TrainingSample] = []
        attempts = 0
        max_attempts = count * 20 + 100
        while len(samples) < count and attempts < max_attempts:
            attempts += 1
            start, goal = self._random_pose_pair()
            if start is None:
                continue
            try:
                trajectory = self.planner.plan(start, goal)
            except (RuntimeError, ValueError):
                continue
            bev = self.sensor_pipeline.capture_bev(start.x, start.y, start.yaw)
            samples.append(
                TrainingSample(bev=bev, goal=goal, state=start, expert_trajectory=trajectory)
            )
        if len(samples) < count:
            raise RuntimeError(
                f"仅生成 {len(samples)}/{count} 条样本，尝试 {attempts} 次后仍未凑齐"
            )
        return samples

    def _generate_task(self, task: Task) -> TrainingSample:
        planner, pipeline = self._components_for(task)
        if getattr(pipeline, "bev_config", None) != task.scene.bev_config:
            raise TaskGenerationError(task.task_id, "传感器管道 BEV 配置与场景不一致")

        goal, trajectory, goal_policy, maneuver_audit = self._resolve_goal(task, planner)
        if hasattr(pipeline, "set_target_goals"):
            pipeline.set_target_goals([goal.as_goal_pose()])
        try:
            bev = pipeline.capture_bev(task.start.x, task.start.y, task.start.yaw)
        except (RuntimeError, ValueError) as exc:
            raise TaskGenerationError(task.task_id, f"BEV 采集失败：{exc}") from exc

        task_meta = task.to_metadata()
        task_meta["noise_profile"] = get_noise_profile(
            task.difficulty.noise_level
        ).to_metadata()
        task_meta["dataset"] = {
            "goal_policy": goal_policy,
            "selected_goal": goal.to_metadata(),
            "maneuver_audit": maneuver_audit.to_metadata(),
        }
        return TrainingSample(
            bev=bev,
            goal=goal.as_goal_pose(),
            state=task.start,
            expert_trajectory=trajectory,
            task_meta=task_meta,
        )

    def _components_for(self, task: Task) -> tuple[Any, Any]:
        if self.component_factory is not None:
            return self.component_factory(task)
        if self.planner is None or self.sensor_pipeline is None:
            raise TaskGenerationError(task.task_id, "未配置 task 组件工厂或默认组件")
        return self.planner, self.sensor_pipeline

    def _resolve_goal(
        self, task: Task, planner: Any
    ) -> tuple[TaskGoal, Trajectory, str, ManeuverAudit]:
        if task.goal is not None:
            goals = (task.goal,)
            policy = "task_goal"
        elif self.goal_selector is not None:
            selected = self.goal_selector(task, planner)
            if selected not in task.candidate_goals:
                raise TaskGenerationError(task.task_id, "goal_selector 返回了候选集外目标")
            goals = (selected,)
            policy = "goal_selector"
        else:
            goals = task.candidate_goals
            policy = (
                "first_consistent_plannable_candidate"
                if self.enforce_maneuver_consistency
                else "first_plannable_candidate"
            )

        failures: list[str] = []
        inconsistent_count = 0
        invalid_audit_count = 0
        for goal in goals:
            try:
                trajectory = planner.plan(task.start, goal.as_goal_pose())
            except (RuntimeError, ValueError) as exc:
                failures.append(f"{goal.spot_id}: {exc}")
                continue
            try:
                audit = audit_maneuver_consistency(
                    trajectory.points,
                    task.difficulty.maneuver,
                    minimum_requested_distance_ratio=(
                        self.minimum_requested_distance_ratio
                    ),
                )
            except ValueError as exc:
                invalid_audit_count += 1
                failures.append(f"{goal.spot_id}: 机动审计失败（{exc}）")
                continue
            if self.enforce_maneuver_consistency and not audit.consistent:
                inconsistent_count += 1
                failures.append(f"{goal.spot_id}: {_maneuver_failure_detail(audit)}")
                continue
            return goal, trajectory, policy, audit
        detail = "; ".join(failures) or "没有可用目标"
        if inconsistent_count:
            code = "maneuver_inconsistent"
        elif invalid_audit_count:
            code = "maneuver_audit_invalid"
        else:
            code = None
        raise TaskGenerationError(
            task.task_id,
            f"所有目标均不可用于专家监督（{detail}）",
            code=code,
        )

    def save(self, samples: list[TrainingSample], path: str | Path) -> None:
        """按 schema v2 写入 npz（轨迹补零，元数据使用 Unicode JSON）。"""
        if not samples:
            raise ValueError("不能保存空数据集")
        bev_meta = samples[0].bev.to_metadata()
        if any(sample.bev.to_metadata() != bev_meta for sample in samples[1:]):
            raise ValueError("同一数据集内所有样本必须使用一致的 BEV 元数据")
        dt = samples[0].expert_trajectory.dt
        if any(
            not np.isclose(sample.expert_trajectory.dt, dt)
            for sample in samples[1:]
        ):
            raise ValueError("同一数据集内所有专家轨迹必须使用一致的 dt")

        max_horizon = max(s.expert_trajectory.horizon for s in samples)
        bevs = np.stack([s.bev.data for s in samples]).astype(np.float32)
        goals = np.array([[s.goal.x, s.goal.y, s.goal.yaw] for s in samples])
        states = np.array([s.state.to_array() for s in samples])
        trajs = np.zeros((len(samples), max_horizon, 3), dtype=np.float32)
        masks = np.zeros((len(samples), max_horizon), dtype=np.float32)
        for i, s in enumerate(samples):
            n = s.expert_trajectory.horizon
            trajs[i, :n] = s.expert_trajectory.points
            masks[i, :n] = 1.0
        bev_meta_json = self._encode_metadata(bev_meta, "bev_meta")
        task_meta_json = np.asarray(
            [self._encode_metadata(sample.task_meta or {}, "task_meta") for sample in samples],
            dtype=np.str_,
        )
        np.savez_compressed(
            path,
            schema_version=np.asarray(2, dtype=np.uint16),
            bev_meta=np.asarray(bev_meta_json, dtype=np.str_),
            task_meta=task_meta_json,
            bevs=bevs,
            goals=goals,
            states=states,
            trajs=trajs,
            masks=masks,
            dt=np.array([dt]),
        )

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """加载 v1/v2 npz；v2 元数据解码为 Python 对象。"""
        with np.load(path, allow_pickle=False) as data:
            loaded = {key: data[key] for key in data.files}

        if "schema_version" not in loaded:
            loaded["schema_version"] = 1
            loaded["bev_meta"] = None
            loaded["task_meta"] = None
            return loaded

        version = int(np.asarray(loaded.pop("schema_version")).item())
        if version != 2:
            raise ValueError(f"不支持的数据集 schema 版本：{version}")
        try:
            bev_meta = json.loads(str(np.asarray(loaded["bev_meta"]).item()))
            task_meta = [json.loads(str(value)) for value in loaded["task_meta"].tolist()]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("schema v2 元数据缺失或 JSON 无效") from exc
        if not isinstance(bev_meta, dict) or not all(
            key in bev_meta for key in ("resolution", "extent", "channels", "shape")
        ):
            raise ValueError("schema v2 的 bev_meta 字段不完整")
        if "bevs" not in loaded or list(loaded["bevs"].shape[1:]) != bev_meta["shape"]:
            raise ValueError("schema v2 的 bev_meta.shape 与 bevs 数组不一致")
        if len(task_meta) != loaded["bevs"].shape[0] or any(
            not isinstance(metadata, dict) for metadata in task_meta
        ):
            raise ValueError("schema v2 的 task_meta 必须与样本逐项对齐")
        loaded["schema_version"] = version
        loaded["bev_meta"] = bev_meta
        loaded["task_meta"] = task_meta
        return loaded

    @staticmethod
    def _encode_metadata(metadata: dict[str, Any], field_name: str) -> str:
        try:
            return json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} 必须可 JSON 序列化") from exc

    def _random_pose_pair(self) -> tuple[VehicleState | None, GoalPose | None]:
        """随机采样无碰撞的起始/目标位姿，要求两点间距足够。"""
        half = self.env.world_size / 2.0 - 1.0
        for _ in range(50):
            sx = self.rng.uniform(-half, half)
            sy = self.rng.uniform(-half, half)
            syaw = self.rng.uniform(-np.pi, np.pi)
            gx = self.rng.uniform(-half, half)
            gy = self.rng.uniform(-half, half)
            gyaw = self.rng.uniform(-np.pi, np.pi)
            start = VehicleState(float(sx), float(sy), float(syaw))
            goal = GoalPose(float(gx), float(gy), float(gyaw))
            if not (self._pose_free(sx, sy, syaw) and self._pose_free(gx, gy, gyaw)):
                continue
            dist = np.hypot(gx - sx, gy - sy)
            if dist < self.min_distance or dist > self.max_distance:
                continue
            return start, goal
        return None, None

    def _pose_free(self, x: float, y: float, yaw: float) -> bool:
        """判断车辆矩形四角是否全部位于自由空间（与规划器一致）。"""
        half_l = self.planner.vehicle_length / 2.0
        half_w = self.planner.vehicle_width / 2.0
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        corners = [
            (x + half_l * cos_yaw - half_w * sin_yaw, y + half_l * sin_yaw + half_w * cos_yaw),
            (x + half_l * cos_yaw + half_w * sin_yaw, y + half_l * sin_yaw - half_w * cos_yaw),
            (x - half_l * cos_yaw - half_w * sin_yaw, y - half_l * sin_yaw + half_w * cos_yaw),
            (x - half_l * cos_yaw + half_w * sin_yaw, y - half_l * sin_yaw - half_w * cos_yaw),
        ]
        return all(self.env.is_free(cx, cy) for cx, cy in corners)


def _maneuver_failure_detail(audit: ManeuverAudit) -> str:
    return (
        f"机动不一致：请求 {audit.requested_maneuver.value}，"
        f"实际前进 {audit.forward_distance_ratio:.1%}、"
        f"倒车 {audit.reverse_distance_ratio:.1%}，"
        f"请求方向占比 {audit.requested_distance_ratio:.1%} < "
        f"{audit.minimum_requested_distance_ratio:.1%}"
    )
