"""T1–T5 任务模型与可复现分层采样。

任务层只描述场景、起点、目标与动态事件，不依赖或调用规划器。对当前场景
几何无法表达的类型，能力矩阵保留不支持原因，采样接口显式拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable

import numpy as np

from interfaces import GoalPose, VehicleState
from sim.footprint import rectangle_pose_is_free
from sim.noise import NoiseLevel
from sim.scenes import SCENE_REGISTRY, SceneBundle, build_scene
from sim.spots import ParkingSpot

__all__ = [
    "DistanceTier",
    "DynamicObstacleEvent",
    "Maneuver",
    "NoiseLevel",
    "Task",
    "TaskCapability",
    "TaskDifficulty",
    "TaskGoal",
    "TaskSampler",
    "TaskType",
    "UnsupportedTaskError",
]


class TaskType(str, Enum):
    """需求定义的五类任务。"""

    T1_NEAR = "T1"
    T2_MEDIUM = "T2"
    T3_LONG = "T3"
    T4_MULTI_SPOT = "T4"
    T5_DYNAMIC = "T5"


class DistanceTier(str, Enum):
    NEAR = "near"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def bounds(self) -> tuple[float, float]:
        return {
            DistanceTier.NEAR: (4.0, 8.0),
            DistanceTier.MEDIUM: (8.0, 15.0),
            DistanceTier.LONG: (15.0, 30.0),
        }[self]


class Maneuver(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class UnsupportedTaskError(ValueError):
    """场景几何或参数无法满足任务契约。"""


_TASK_DISTANCE = {
    TaskType.T1_NEAR: DistanceTier.NEAR,
    TaskType.T2_MEDIUM: DistanceTier.MEDIUM,
    TaskType.T3_LONG: DistanceTier.LONG,
    TaskType.T4_MULTI_SPOT: DistanceTier.MEDIUM,
    TaskType.T5_DYNAMIC: DistanceTier.MEDIUM,
}

_OCCUPANCY_SCENES = {
    "S1_parking_lot",
    "S2_diagonal_lot",
    "S4_dump_area",
    "S7_fuel_station",
    "S9_mine_complex",
}

_VEHICLE_SCALED_SCENES = {"S3_maintenance", "S4_dump_area", "S7_fuel_station", "S9_mine_complex"}

# 使用连续轴线入口的车位类型。紧 bay 需要避免随机大转角；普通垂直/斜列
# 车位也需要让请求的前进/倒车机动与车头朝向、入口侧保持一致。
_AXIAL_BAY_KINDS = {
    "crusher_slot",
    "maintenance_bay",
    "fuel_bay",
    "perpendicular_bay",
    "diagonal_bay",
}
_BIDIRECTIONAL_ENTRY_KINDS = {"perpendicular_bay", "diagonal_bay"}


@dataclass(frozen=True)
class TaskGoal:
    """可序列化的车位目标快照。"""

    spot_id: str
    x: float
    y: float
    yaw: float
    tol_pos: float
    tol_yaw: float
    kind: str

    @classmethod
    def from_spot(cls, spot: ParkingSpot) -> "TaskGoal":
        return cls(
            spot_id=spot.id,
            x=float(spot.pose.x),
            y=float(spot.pose.y),
            yaw=float(spot.pose.yaw),
            tol_pos=float(spot.tol_pos),
            tol_yaw=float(spot.tol_yaw),
            kind=spot.kind,
        )

    def as_goal_pose(self) -> GoalPose:
        return GoalPose(self.x, self.y, self.yaw)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "spot_id": self.spot_id,
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "tol_pos": self.tol_pos,
            "tol_yaw": self.tol_yaw,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class TaskDifficulty:
    """任务的正交难度坐标。"""

    distance_tier: DistanceTier
    maneuver: Maneuver
    adjacent_occupancy: int
    aisle_width: float | None
    noise_level: NoiseLevel
    scene_knobs: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.adjacent_occupancy not in (0, 1, 2):
            raise ValueError("adjacent_occupancy 必须是 0、1 或 2")
        if self.aisle_width is not None and self.aisle_width <= 0.0:
            raise ValueError("aisle_width 必须为正数或 None")
        normalized = tuple(sorted((str(key), _json_scalar(value)) for key, value in self.scene_knobs))
        object.__setattr__(self, "scene_knobs", normalized)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "distance_tier": self.distance_tier.value,
            "distance_bounds_m": list(self.distance_tier.bounds),
            "maneuver": self.maneuver.value,
            "adjacent_occupancy": self.adjacent_occupancy,
            "aisle_width_m": self.aisle_width,
            "noise_level": self.noise_level.value,
            "scene_knobs": dict(self.scene_knobs),
        }


@dataclass(frozen=True)
class DynamicObstacleEvent:
    """T5 的一次性进度触发障碍注入载荷。"""

    trigger_progress: float
    x: float
    y: float
    radius: float = 1.0
    obstacle_kind: str = "vehicle"
    action: str = "add_circle_obstacle"

    def __post_init__(self) -> None:
        if not 0.0 < self.trigger_progress < 1.0:
            raise ValueError("trigger_progress 必须位于 (0, 1)")
        if self.radius <= 0.0:
            raise ValueError("动态障碍半径必须为正")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "trigger": {"kind": "path_progress", "value": self.trigger_progress},
            "action": self.action,
            "obstacle": {
                "shape": "circle",
                "kind": self.obstacle_kind,
                "x": self.x,
                "y": self.y,
                "radius": self.radius,
            },
        }


@dataclass(frozen=True)
class Task:
    """一个具体场景实例上的可执行任务。"""

    task_id: str
    scene: SceneBundle
    task_type: TaskType
    seed: int
    start: VehicleState
    difficulty: TaskDifficulty
    goal: TaskGoal | None = None
    candidate_goals: tuple[TaskGoal, ...] = ()
    dynamic_event: DynamicObstacleEvent | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("任务 seed 不能为负")
        candidate_ids = [goal.spot_id for goal in self.candidate_goals]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("候选目标不能重复")
        if self.task_type == TaskType.T4_MULTI_SPOT:
            if self.goal is not None or not 3 <= len(self.candidate_goals) <= 6:
                raise ValueError("T4 必须无预选目标且携带 3–6 个候选目标")
        elif self.goal is None or self.candidate_goals:
            raise ValueError("T1/T2/T3/T5 必须恰有一个目标且无候选目标集")
        if (self.task_type == TaskType.T5_DYNAMIC) != (self.dynamic_event is not None):
            raise ValueError("仅 T5 且每个 T5 必须携带动态事件")

    @property
    def scene_name(self) -> str:
        return self.scene.name

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "scene_name": self.scene.name,
            "task_type": self.task_type.value,
            "seed": self.seed,
            "start": {
                "x": float(self.start.x),
                "y": float(self.start.y),
                "yaw": float(self.start.yaw),
                "v": float(self.start.v),
                "omega": float(self.start.omega),
            },
            "goal": None if self.goal is None else self.goal.to_metadata(),
            "candidate_goals": [goal.to_metadata() for goal in self.candidate_goals],
            "difficulty": self.difficulty.to_metadata(),
            "dynamic_event": (
                None if self.dynamic_event is None else self.dynamic_event.to_metadata()
            ),
        }


@dataclass(frozen=True)
class TaskCapability:
    scene_name: str
    task_type: TaskType
    supported: bool
    reason: str = ""


class TaskSampler:
    """按稳定坐标派生随机流的 T1–T5 采样器。"""

    def __init__(
        self,
        seed: int,
        *,
        vehicle_length: float = 6.0,
        vehicle_width: float = 3.0,
        collision_margin: float = 0.0,
        max_attempts: int = 4000,
    ) -> None:
        if int(seed) < 0:
            raise ValueError("根 seed 不能为负")
        if vehicle_length <= 0.0 or vehicle_width <= 0.0:
            raise ValueError("车辆尺寸必须为正")
        if collision_margin < 0.0:
            raise ValueError("collision_margin 不能为负")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须为正")
        self.seed = int(seed)
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.collision_margin = float(collision_margin)
        self.max_attempts = int(max_attempts)

    def _build_scene(self, scene_name: str, **kwargs) -> SceneBundle:
        """给依赖车体比例的场景注入与规划器一致的几何参数。"""
        if scene_name in _VEHICLE_SCALED_SCENES:
            kwargs.update(
                vehicle_length=self.vehicle_length,
                vehicle_width=self.vehicle_width,
                collision_margin=self.collision_margin,
            )
        return build_scene(scene_name, **kwargs)

    @staticmethod
    def supports_adjacent_occupancy(scene_name: str) -> bool:
        """返回场景构造器是否支持相邻占用难度轴。"""
        if scene_name not in SCENE_REGISTRY:
            raise ValueError(f"未知场景 {scene_name}")
        return scene_name in _OCCUPANCY_SCENES

    def adjacent_occupancy_levels(
        self, scene_name: str, task_type: TaskType | str
    ) -> tuple[int, ...]:
        """返回场景×任务类型实际可表达的相邻占用等级。"""
        kind = _as_task_type(task_type)
        if scene_name not in SCENE_REGISTRY:
            raise ValueError(f"未知场景 {scene_name}")
        if scene_name not in _OCCUPANCY_SCENES:
            return (0,)
        scene = self._build_scene(scene_name, seed=0)
        eligible = self._eligible_spot_indices(scene, _TASK_DISTANCE[kind])
        levels = [0]
        for count in (1, 2):
            patterns = self._valid_occupancy_patterns(
                scene_name, 0, scene, eligible, count
            )
            if kind == TaskType.T4_MULTI_SPOT:
                patterns = {
                    index: bundle for index, bundle in patterns.items()
                    if len(bundle.free_spots()) >= 3
                }
            if patterns:
                levels.append(count)
        return tuple(levels)

    def capability_matrix(
        self,
        scene_names: Iterable[str] | None = None,
        task_types: Iterable[TaskType | str] | None = None,
    ) -> tuple[TaskCapability, ...]:
        """返回稳定排序的完整能力矩阵，包括不支持单元。"""
        scenes = self._normalize_scenes(scene_names)
        kinds = self._normalize_task_types(task_types)
        cells: list[TaskCapability] = []
        for scene_name in scenes:
            bundle = self._build_scene(scene_name, seed=0)
            for task_type in kinds:
                reason = self._unsupported_reason(bundle, task_type)
                cells.append(TaskCapability(scene_name, task_type, not reason, reason))
        return tuple(cells)

    def sample_matrix(
        self,
        *,
        samples_per_cell: int = 1,
        strict: bool = False,
        scene_names: Iterable[str] | None = None,
        task_types: Iterable[TaskType | str] | None = None,
    ) -> tuple[Task, ...]:
        """对能力矩阵中的支持单元采样；严格模式拒绝任一不支持单元。"""
        if samples_per_cell <= 0:
            raise ValueError("samples_per_cell 必须为正")
        cells = self.capability_matrix(scene_names, task_types)
        unsupported = [cell for cell in cells if not cell.supported]
        if strict and unsupported:
            details = "; ".join(
                f"{cell.scene_name}/{cell.task_type.value}: {cell.reason}"
                for cell in unsupported[:5]
            )
            raise UnsupportedTaskError(f"严格矩阵包含不支持单元：{details}")
        tasks: list[Task] = []
        for cell in cells:
            if not cell.supported:
                continue
            for sample_index in range(samples_per_cell):
                tasks.append(self.sample(cell.scene_name, cell.task_type, sample_index))
        return tuple(tasks)

    def sample(
        self,
        scene_name: str,
        task_type: TaskType | str,
        sample_index: int = 0,
        *,
        maneuver: Maneuver | str | None = None,
        adjacent_occupancy: int = 0,
        noise_level: NoiseLevel | str = NoiseLevel.CLEAN,
    ) -> Task:
        """采样一个满足类型距离、候选数和 footprint 安全约束的任务。"""
        kind = _as_task_type(task_type)
        if scene_name not in SCENE_REGISTRY:
            raise ValueError(f"未知场景 {scene_name}，可选：{sorted(SCENE_REGISTRY)}")
        if sample_index < 0:
            raise ValueError("sample_index 不能为负")
        if adjacent_occupancy not in (0, 1, 2):
            raise ValueError("adjacent_occupancy 必须是 0、1 或 2")
        selected_maneuver = (
            self._default_maneuver(scene_name) if maneuver is None else Maneuver(maneuver)
        )
        selected_noise = NoiseLevel(noise_level)
        task_seed, scene_seed, rng = self._random_stream(scene_name, kind, sample_index)

        scene = self._build_scene(scene_name, seed=scene_seed)
        self._orient_spots_for_maneuver(scene, selected_maneuver)
        reason = self._unsupported_reason(scene, kind)
        if reason:
            raise UnsupportedTaskError(f"{scene_name}/{kind.value}: {reason}")

        tier = _TASK_DISTANCE[kind]
        eligible_indices = [
            index
            for index in self._eligible_spot_indices(scene, tier)
            if self._structured_approach_possible(
                scene, scene.spots[index], tier, selected_maneuver
            )
        ]
        if not eligible_indices:
            raise UnsupportedTaskError(
                f"{scene_name}/{kind.value}: 请求方向与距离档下无连续无碰撞入口走廊"
            )
        target_index = int(rng.choice(eligible_indices))
        if adjacent_occupancy:
            scene, target_index = self._with_adjacent_occupancy(
                scene_name, scene_seed, scene, eligible_indices,
                target_index, adjacent_occupancy, kind,
                tier, selected_maneuver, rng,
            )
            if kind == TaskType.T4_MULTI_SPOT and len(scene.free_spots()) < 3:
                raise UnsupportedTaskError(
                    f"{scene_name}/T4: 相邻占用后不足 3 个候选空闲车位"
                )

        reference_spot = scene.spots[target_index]
        start = self._sample_start(scene, reference_spot, tier, selected_maneuver, kind, rng)

        goal: TaskGoal | None
        candidates: tuple[TaskGoal, ...]
        if kind == TaskType.T4_MULTI_SPOT:
            candidate_spots = self._candidate_spots(scene, reference_spot, rng)
            goal = None
            candidates = tuple(TaskGoal.from_spot(spot) for spot in candidate_spots)
        else:
            goal = TaskGoal.from_spot(reference_spot)
            candidates = ()

        difficulty = TaskDifficulty(
            distance_tier=tier,
            maneuver=selected_maneuver,
            adjacent_occupancy=adjacent_occupancy,
            aisle_width=_optional_float(scene.difficulty_knobs.get("aisle_width")),
            noise_level=selected_noise,
            scene_knobs=tuple(scene.difficulty_knobs.items()),
        )
        event = None
        if kind == TaskType.T5_DYNAMIC:
            event = self._dynamic_event(scene, start, reference_spot.pose)
        task_id = f"{scene_name}-{kind.value}-{sample_index:04d}-{task_seed:08x}"
        return Task(
            task_id=task_id,
            scene=scene,
            task_type=kind,
            seed=task_seed,
            start=start,
            difficulty=difficulty,
            goal=goal,
            candidate_goals=candidates,
            dynamic_event=event,
        )

    def pose_is_free(self, env, x: float, y: float, yaw: float) -> bool:
        """使用与 Hybrid A* 相同的完整矩形 footprint 检查位姿。"""
        half_l = self.vehicle_length / 2.0 + self.collision_margin
        half_w = self.vehicle_width / 2.0 + self.collision_margin
        return rectangle_pose_is_free(
            env,
            x,
            y,
            yaw,
            half_length=half_l,
            half_width=half_w,
        )

    def _random_stream(
        self, scene_name: str, task_type: TaskType, sample_index: int
    ) -> tuple[int, int, np.random.Generator]:
        scenes = sorted(SCENE_REGISTRY)
        kinds = list(TaskType)
        coordinates = [self.seed, scenes.index(scene_name), kinds.index(task_type), sample_index]
        task_seq, scene_seq, sample_seq = np.random.SeedSequence(coordinates).spawn(3)
        task_seed = int(task_seq.generate_state(1, dtype=np.uint32)[0])
        scene_seed = int(scene_seq.generate_state(1, dtype=np.uint32)[0])
        return task_seed, scene_seed, np.random.default_rng(sample_seq)

    def _unsupported_reason(self, scene: SceneBundle, task_type: TaskType) -> str:
        if not scene.spawn_zones:
            return "场景没有起点采样区"
        free_spots = scene.free_spots()
        if not free_spots:
            return "场景没有空闲目标车位"
        if task_type == TaskType.T4_MULTI_SPOT and len(free_spots) < 3:
            return f"T4 需要至少 3 个空闲车位，当前只有 {len(free_spots)} 个"
        tier = _TASK_DISTANCE[task_type]
        if not any(
            self._start_footprint_possible(scene, spot.pose, tier, task_type)
            for spot in free_spots
        ):
            lower, upper = tier.bounds
            return f"起点区内无 footprint 安全位姿可满足 {lower:g}–{upper:g}m 距离"
        return ""

    def _eligible_spot_indices(self, scene: SceneBundle, tier: DistanceTier) -> list[int]:
        return [
            index for index, spot in enumerate(scene.spots)
            if not spot.occupied and self._distance_possible(scene.spawn_zones, spot.pose, tier)
        ]

    @staticmethod
    def _distance_possible(
        zones: Iterable[tuple[float, float, float, float]],
        goal: GoalPose,
        tier: DistanceTier,
    ) -> bool:
        lower, upper = tier.bounds
        for x_min, x_max, y_min, y_max in zones:
            dx_min = max(x_min - goal.x, 0.0, goal.x - x_max)
            dy_min = max(y_min - goal.y, 0.0, goal.y - y_max)
            nearest = math.hypot(dx_min, dy_min)
            farthest = max(
                math.hypot(x - goal.x, y - goal.y)
                for x in (x_min, x_max) for y in (y_min, y_max)
            )
            if nearest <= upper and farthest >= lower:
                return True
        return False

    def _start_footprint_possible(
        self,
        scene: SceneBundle,
        goal: GoalPose,
        tier: DistanceTier,
        task_type: TaskType,
    ) -> bool:
        """确定性探测距离区间是否包含完整车辆可落位姿。"""
        lower, upper = tier.bounds
        yaw_offsets = (0.0,)
        if task_type in (TaskType.T2_MEDIUM, TaskType.T3_LONG):
            yaw_offsets = tuple(math.radians(value) for value in (25.0, -25.0, 45.0, -45.0, 70.0, -70.0))
        for x_min, x_max, y_min, y_max in scene.spawn_zones:
            for x in np.linspace(x_min, x_max, 25):
                for y in np.linspace(y_min, y_max, 25):
                    distance = math.hypot(goal.x - x, goal.y - y)
                    if not lower <= distance <= upper:
                        continue
                    travel_yaw = math.atan2(goal.y - y, goal.x - x)
                    if any(
                        self.pose_is_free(scene.env, float(x), float(y), travel_yaw + offset)
                        for offset in yaw_offsets
                    ):
                        return True
        return False

    def _with_adjacent_occupancy(
        self,
        scene_name: str,
        scene_seed: int,
        scene: SceneBundle,
        eligible_indices: list[int],
        target_index: int,
        count: int,
        task_type: TaskType,
        tier: DistanceTier,
        maneuver: Maneuver,
        rng: np.random.Generator,
    ) -> tuple[SceneBundle, int]:
        if scene_name not in _OCCUPANCY_SCENES:
            raise UnsupportedTaskError(f"{scene_name} 不支持相邻占用参数")
        patterns = self._valid_occupancy_patterns(
            scene_name, scene_seed, scene, eligible_indices, count
        )
        for bundle in patterns.values():
            self._orient_spots_for_maneuver(bundle, maneuver)
        patterns = {
            index: bundle
            for index, bundle in patterns.items()
            if self._structured_approach_possible(
                bundle, bundle.spots[index], tier, maneuver
            )
        }
        if task_type == TaskType.T4_MULTI_SPOT:
            patterns = {
                index: bundle for index, bundle in patterns.items()
                if len(bundle.free_spots()) >= 3
            }
        valid_targets = list(patterns)
        if not valid_targets:
            raise UnsupportedTaskError(
                f"{scene_name}/{task_type.value} 无法表达 {count} 个相邻占用且保持目标安全"
            )
        if target_index not in patterns:
            target_index = int(rng.choice(valid_targets))
        return patterns[target_index], target_index

    def _valid_occupancy_patterns(
        self,
        scene_name: str,
        scene_seed: int,
        scene: SceneBundle,
        eligible_indices: list[int],
        count: int,
    ) -> dict[int, SceneBundle]:
        """枚举保持目标矿卡 footprint 安全的相邻占用场景。"""
        occupiable = list(range(len(scene.spots)))
        if scene_name == "S9_mine_complex":
            occupiable = [index for index, spot in enumerate(scene.spots) if spot.id.startswith("P")]
        patterns: dict[int, SceneBundle] = {}
        for index in eligible_indices:
            if index not in occupiable:
                continue
            neighbors = self._side_neighbors(scene, index, occupiable, count)
            if len(neighbors) != count:
                continue
            rebuilt = self._build_scene(
                scene_name, seed=scene_seed, occupied_pattern=neighbors
            )
            goal = rebuilt.spots[index].pose
            if self.pose_is_free(rebuilt.env, goal.x, goal.y, goal.yaw):
                patterns[index] = rebuilt
        return patterns

    @staticmethod
    def _side_neighbors(
        scene: SceneBundle, target_index: int, occupiable: list[int], count: int
    ) -> list[int]:
        target = scene.spots[target_index]
        family = target.id.rstrip("0123456789")
        same_row = [
            index for index in occupiable
            if index != target_index and scene.spots[index].id.rstrip("0123456789") == family
        ]
        if count == 1:
            return sorted(
                same_row,
                key=lambda index: (
                    abs(scene.spots[index].pose.x - target.pose.x), scene.spots[index].id
                ),
            )[:1]
        left = [index for index in same_row if scene.spots[index].pose.x < target.pose.x]
        right = [index for index in same_row if scene.spots[index].pose.x > target.pose.x]
        if not left or not right:
            return []
        nearest_left = max(left, key=lambda index: scene.spots[index].pose.x)
        nearest_right = min(right, key=lambda index: scene.spots[index].pose.x)
        return [nearest_left, nearest_right]

    def _sample_start(
        self,
        scene: SceneBundle,
        reference_spot: ParkingSpot,
        tier: DistanceTier,
        maneuver: Maneuver,
        task_type: TaskType,
        rng: np.random.Generator,
    ) -> VehicleState:
        if reference_spot.kind in _AXIAL_BAY_KINDS:
            axial = self._sample_axial_start(scene, reference_spot, tier, maneuver, rng)
            if axial is not None:
                return axial
        lower, upper = tier.bounds
        for _ in range(self.max_attempts):
            zone = scene.spawn_zones[int(rng.integers(0, len(scene.spawn_zones)))]
            x = float(rng.uniform(zone[0], zone[1]))
            y = float(rng.uniform(zone[2], zone[3]))
            distance = math.hypot(reference_spot.pose.x - x, reference_spot.pose.y - y)
            if distance < lower or distance > upper:
                continue
            travel_yaw = math.atan2(reference_spot.pose.y - y, reference_spot.pose.x - x)
            yaw = travel_yaw if maneuver == Maneuver.FORWARD else travel_yaw + math.pi
            if task_type in (TaskType.T2_MEDIUM, TaskType.T3_LONG):
                turn = float(rng.uniform(math.radians(25.0), math.radians(70.0)))
                yaw += turn if rng.random() < 0.5 else -turn
            yaw = _wrap_angle(yaw)
            if self.pose_is_free(scene.env, x, y, yaw):
                return VehicleState(x=x, y=y, yaw=yaw)
        # 极窄可行带（例如 S5/T3 接近距离上界）用确定性网格兜底；
        # 再由该任务随机流从可行点中选择，避免把低测度单元误报为不可采样。
        feasible: list[VehicleState] = []
        offsets = (0.0,)
        if task_type in (TaskType.T2_MEDIUM, TaskType.T3_LONG):
            offsets = tuple(
                math.radians(value) for value in (25.0, -25.0, 45.0, -45.0, 70.0, -70.0)
            )
        for x_min, x_max, y_min, y_max in scene.spawn_zones:
            for x in np.linspace(x_min, x_max, 41):
                for y in np.linspace(y_min, y_max, 41):
                    distance = math.hypot(reference_spot.pose.x - x, reference_spot.pose.y - y)
                    if not lower <= distance <= upper:
                        continue
                    travel_yaw = math.atan2(reference_spot.pose.y - y, reference_spot.pose.x - x)
                    base_yaw = travel_yaw if maneuver == Maneuver.FORWARD else travel_yaw + math.pi
                    for offset in offsets:
                        yaw = _wrap_angle(base_yaw + offset)
                        if self.pose_is_free(scene.env, float(x), float(y), yaw):
                            feasible.append(VehicleState(float(x), float(y), yaw))
        if feasible:
            return feasible[int(rng.integers(0, len(feasible)))]
        raise UnsupportedTaskError(
            f"{scene.name}/{task_type.value}: {self.max_attempts} 次内未找到满足距离与 footprint 的起点"
        )

    def _sample_axial_start(
        self,
        scene: SceneBundle,
        spot: ParkingSpot,
        tier: DistanceTier,
        maneuver: Maneuver,
        rng: np.random.Generator,
    ) -> VehicleState | None:
        """沿紧 bay 轴线对齐的起点：车辆与 bay 轴线平行（yaw=gyaw），
        按机动方向从轴线对应一侧 d 米外接近。

        真实矿区中卡车在维修 bay/破碎槽/加油位前先对齐再直线入位；远距离下
        随机大转角会导致解析候选在紧 bay 口全部碰撞、离散搜索难以收敛。
        倒车入位从鼻前侧（G+nose*d）接近、前进入位从鼻后侧（G-nose*d）接近；
        机动方向与 bay 几何不一致时返回 None 回退常规采样。
        """
        gx, gy, gyaw = spot.pose.x, spot.pose.y, spot.pose.yaw
        nx, ny = math.cos(gyaw), math.sin(gyaw)
        lower, upper = tier.bounds
        clear_limit = self._axial_clearance_limit(scene, spot, maneuver, upper)
        if clear_limit + 1e-9 < lower:
            return None
        distance = float(rng.uniform(lower, min(upper, clear_limit)))
        side = -1.0 if maneuver == Maneuver.FORWARD else 1.0
        return VehicleState(
            x=gx + side * nx * distance,
            y=gy + side * ny * distance,
            yaw=gyaw,
        )

    def _structured_approach_possible(
        self,
        scene: SceneBundle,
        spot: ParkingSpot,
        tier: DistanceTier,
        maneuver: Maneuver,
    ) -> bool:
        if spot.kind not in _AXIAL_BAY_KINDS:
            return True
        lower, upper = tier.bounds
        return self._axial_clearance_limit(scene, spot, maneuver, upper) + 1e-9 >= lower

    def _axial_clearance_limit(
        self,
        scene: SceneBundle,
        spot: ParkingSpot,
        maneuver: Maneuver,
        upper: float,
    ) -> float:
        """返回从目标沿请求入口侧连续保持 footprint 自由的最大轴向距离。"""
        gx, gy, gyaw = spot.pose.x, spot.pose.y, spot.pose.yaw
        nx, ny = math.cos(gyaw), math.sin(gyaw)
        side = -1.0 if maneuver == Maneuver.FORWARD else 1.0
        step = 0.1
        clear_limit = 0.0
        for distance in np.linspace(0.0, upper, int(math.ceil(upper / step)) + 1):
            if not self.pose_is_free(
                scene.env,
                gx + side * nx * float(distance),
                gy + side * ny * float(distance),
                gyaw,
            ):
                break
            clear_limit = float(distance)
        return clear_limit

    @staticmethod
    def _orient_spots_for_maneuver(scene: SceneBundle, maneuver: Maneuver) -> None:
        """普通车位的目标航向随入位方式变化：倒车时车头朝入口。"""
        if maneuver != Maneuver.REVERSE:
            return
        for spot in scene.spots:
            if spot.kind in _BIDIRECTIONAL_ENTRY_KINDS:
                spot.pose = GoalPose(
                    spot.pose.x,
                    spot.pose.y,
                    _wrap_angle(spot.pose.yaw + math.pi),
                )

    @staticmethod
    def _candidate_spots(
        scene: SceneBundle, reference_spot: ParkingSpot, rng: np.random.Generator
    ) -> list[ParkingSpot]:
        free = scene.free_spots()
        if len(free) < 3:
            raise UnsupportedTaskError("T4 需要至少 3 个候选空闲车位")
        max_count = min(6, len(free))
        count = int(rng.integers(3, max_count + 1))
        others = [spot for spot in free if spot.id != reference_spot.id]
        rng.shuffle(others)
        selected = [reference_spot, *others[: count - 1]]
        return selected

    def _dynamic_event(
        self, scene: SceneBundle, start: VehicleState, goal: GoalPose
    ) -> DynamicObstacleEvent:
        radius = 1.0
        dx, dy = goal.x - start.x, goal.y - start.y
        distance = math.hypot(dx, dy)
        normal_x, normal_y = (-dy / distance, dx / distance)
        for progress in (0.5, 0.4, 0.6, 0.3, 0.7):
            center_x = start.x + progress * dx
            center_y = start.y + progress * dy
            # 真实轨迹可能绕过静态设施；在同一进度截面横向搜索自由注入点，
            # 不把起终点直线误当作已规划路径。
            for lateral in (0.0, 2.5, -2.5, 5.0, -5.0, 7.5, -7.5):
                x = center_x + lateral * normal_x
                y = center_y + lateral * normal_y
                if self._circle_is_free(scene, x, y, radius):
                    return DynamicObstacleEvent(progress, float(x), float(y), radius)
        raise UnsupportedTaskError(f"{scene.name}/T5: 起终点之间无动态障碍自由注入点")

    @staticmethod
    def _circle_is_free(scene: SceneBundle, x: float, y: float, radius: float) -> bool:
        return all(
            scene.env.is_free(
                x + radius * math.cos(angle), y + radius * math.sin(angle)
            )
            for angle in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        ) and scene.env.is_free(x, y)

    @staticmethod
    def _default_maneuver(scene_name: str) -> Maneuver:
        if scene_name in {"S6_loading_face", "S8_weigh_station"}:
            return Maneuver.FORWARD
        return Maneuver.REVERSE

    @staticmethod
    def _normalize_scenes(scene_names: Iterable[str] | None) -> tuple[str, ...]:
        scenes = tuple(sorted(SCENE_REGISTRY) if scene_names is None else scene_names)
        unknown = [name for name in scenes if name not in SCENE_REGISTRY]
        if unknown:
            raise ValueError(f"未知场景：{unknown}")
        return scenes

    @staticmethod
    def _normalize_task_types(
        task_types: Iterable[TaskType | str] | None,
    ) -> tuple[TaskType, ...]:
        if task_types is None:
            return tuple(TaskType)
        return tuple(_as_task_type(value) for value in task_types)


def _as_task_type(value: TaskType | str) -> TaskType:
    if isinstance(value, TaskType):
        return value
    return TaskType(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"scene_knobs 仅允许 JSON 基础类型，收到 {type(value).__name__}")


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
