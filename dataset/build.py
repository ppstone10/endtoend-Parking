"""任务配额计划与同单元失败重采。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Iterable

from sim.tasks import (
    Maneuver,
    NoiseLevel,
    Task,
    TaskCapability,
    TaskSampler,
    TaskType,
    UnsupportedTaskError,
)

from .generator import DatasetGenerator, TaskGenerationError
from .splits import DatasetSplits, split_tasks


_EXPERT_UNREACHABLE_CELLS = {
    ("S5_crusher", TaskType.T2_MEDIUM),
    ("S5_crusher", TaskType.T3_LONG),
    ("S7_fuel_station", TaskType.T1_NEAR),
    ("S7_fuel_station", TaskType.T3_LONG),
    ("S7_fuel_station", TaskType.T5_DYNAMIC),
    ("S8_weigh_station", TaskType.T2_MEDIUM),
    ("S8_weigh_station", TaskType.T3_LONG),
    ("S9_mine_complex", TaskType.T3_LONG),
}
_EXPERT_MANEUVER_OVERRIDES = {
    **{
        ("S3_maintenance", task_type): (Maneuver.REVERSE,)
        for task_type in TaskType
    },
    **{
        ("S5_crusher", task_type): (Maneuver.REVERSE,)
        for task_type in (
            TaskType.T1_NEAR,
            TaskType.T5_DYNAMIC,
        )
    },
    ("S8_weigh_station", TaskType.T5_DYNAMIC): (Maneuver.FORWARD,),
    ("S9_mine_complex", TaskType.T1_NEAR): (Maneuver.FORWARD,),
    ("S9_mine_complex", TaskType.T5_DYNAMIC): (Maneuver.FORWARD,),
}


@dataclass(frozen=True)
class BuildReport:
    """一组期望 Task 的生成结果与重采证据。"""

    samples: tuple[Any, ...]
    failure_count: int
    failure_reasons: dict[str, int]
    replacements: tuple[Task, ...]


def build_task_plan(
    total_count: int,
    *,
    seed: int,
    test_scene: str = "S9_mine_complex",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    vehicle_length: float = 6.0,
    vehicle_width: float = 3.0,
    collision_margin: float = 0.0,
) -> DatasetSplits:
    """按能力矩阵构造目标比例任务，并将测试配额全部分给保留场景。"""
    if total_count < 10:
        raise ValueError("任务计划至少需要 10 条样本以表达 8:1:1")
    if len(ratios) != 3 or any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("train/val/test 比例必须包含三个正数")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/val/test 比例之和必须为 1")
    sampler = TaskSampler(
        seed=seed,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        collision_margin=collision_margin,
    )
    supported = [
        cell for cell in sampler.capability_matrix()
        if cell.supported and expert_maneuvers(cell.scene_name, cell.task_type)
    ]
    heldout_cells = [cell for cell in supported if cell.scene_name == test_scene]
    regular_cells = [cell for cell in supported if cell.scene_name != test_scene]
    if not heldout_cells:
        raise ValueError(f"测试保留场景 {test_scene} 没有支持的任务单元")

    test_count = round(total_count * ratios[2])
    val_count = round(total_count * ratios[1])
    if test_count <= 0 or val_count <= 0:
        raise ValueError("任务总量不足以分配 val/test")
    regular_count = total_count - test_count

    active_regular = regular_cells[: min(len(regular_cells), val_count)]
    active_heldout = heldout_cells[: min(len(heldout_cells), test_count)]
    regular_quotas = _balanced_quotas(regular_count, active_regular)
    heldout_quotas = _balanced_quotas(test_count, active_heldout)

    tasks = [
        *_sample_cells(sampler, active_regular, regular_quotas),
        *_sample_cells(sampler, active_heldout, heldout_quotas),
    ]
    splits = split_tasks(tasks, seed=seed, test_scene=test_scene, ratios=ratios)
    expected = (total_count - val_count - test_count, val_count, test_count)
    actual = (len(splits.train), len(splits.val), len(splits.test))
    if actual != expected:
        raise RuntimeError(f"任务划分数量 {actual} 与计划 {expected} 不一致")
    return splits


def generate_with_retries(
    tasks: Iterable[Task],
    *,
    generator: DatasetGenerator,
    seed: int,
    max_retries: int = 20,
    reserved_task_ids: Iterable[str] = (),
    task_sampler: TaskSampler | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BuildReport:
    """逐 Task 生成；失败时保持场景×类型×难度并增加样本索引。"""
    if max_retries <= 0:
        raise ValueError("max_retries 必须为正")
    task_list = tuple(tasks)
    reserved_ids = set(reserved_task_ids)
    reserved_ids.update(task.task_id for task in task_list)
    sampler = TaskSampler(seed=seed) if task_sampler is None else task_sampler
    if sampler.seed != int(seed):
        raise ValueError("task_sampler.seed 必须与 seed 一致")
    next_indices: dict[tuple[str, Any], int] = {}
    for task in task_list:
        key = (task.scene_name, task.task_type)
        next_indices[key] = max(next_indices.get(key, 0), _task_index(task) + 1)
    for task_id in reserved_ids:
        coordinates = _task_id_coordinates(task_id)
        if coordinates is None:
            continue
        scene_name, task_type, sample_index = coordinates
        key = (scene_name, task_type)
        next_indices[key] = max(next_indices.get(key, 0), sample_index + 1)

    samples: list[Any] = []
    replacements: list[Task] = []
    reasons: Counter[str] = Counter()
    failure_count = 0
    for original in task_list:
        current = original
        for retry in range(max_retries + 1):
            try:
                samples.extend(generator.generate([current]))
                break
            except TaskGenerationError as exc:
                failure_count += 1
                reasons[exc.code] += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "original_task_id": original.task_id,
                            "current_task_id": current.task_id,
                            "retry": retry + 1,
                            "max_attempts": max_retries + 1,
                            "failure_code": exc.code,
                        }
                    )
                if retry == max_retries:
                    raise RuntimeError(
                        f"{original.scene_name}/{original.task_type.value} 超过 "
                        f"{max_retries} 次重采：{exc.reason}"
                    ) from exc
                current = _replacement_task(
                    sampler, current, next_indices, reserved_ids
                )
                reserved_ids.add(current.task_id)
                replacements.append(current)

    return BuildReport(
        samples=tuple(samples),
        failure_count=failure_count,
        failure_reasons=dict(sorted(reasons.items())),
        replacements=tuple(replacements),
    )


def expert_maneuvers(
    scene_name: str, task_type: TaskType | str
) -> tuple[Maneuver, ...]:
    """返回当前 Hybrid A* 能稳定提供监督的任务机动方向。"""
    kind = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
    key = (scene_name, kind)
    if key in _EXPERT_UNREACHABLE_CELLS:
        return ()
    return _EXPERT_MANEUVER_OVERRIDES.get(
        key, (Maneuver.FORWARD, Maneuver.REVERSE)
    )


def _balanced_quotas(
    count: int, cells: list[TaskCapability]
) -> dict[tuple[str, Any], int]:
    if not cells:
        raise ValueError("没有可分配的任务单元")
    base, remainder = divmod(count, len(cells))
    return {
        (cell.scene_name, cell.task_type): base + int(index < remainder)
        for index, cell in enumerate(cells)
    }


def _sample_cells(
    sampler: TaskSampler,
    cells: list[TaskCapability],
    quotas: dict[tuple[str, Any], int],
) -> list[Task]:
    tasks: list[Task] = []
    for cell in cells:
        quota = quotas[(cell.scene_name, cell.task_type)]
        sample_index = 0
        adjacent_levels = _adjacent_levels(sampler, cell)
        maneuver_levels = expert_maneuvers(cell.scene_name, cell.task_type)
        if not maneuver_levels:
            raise ValueError(
                f"{cell.scene_name}/{cell.task_type.value} 不在专家可生成能力矩阵"
            )
        for ordinal in range(quota):
            noise_level = tuple(NoiseLevel)[ordinal % len(NoiseLevel)]
            maneuver = maneuver_levels[(ordinal // 3) % len(maneuver_levels)]
            adjacent = adjacent_levels[(ordinal // 6) % len(adjacent_levels)]
            for _ in range(100):
                try:
                    task = sampler.sample(
                        cell.scene_name,
                        cell.task_type,
                        sample_index=sample_index,
                        maneuver=maneuver,
                        adjacent_occupancy=adjacent,
                        noise_level=noise_level,
                    )
                except UnsupportedTaskError:
                    sample_index += 1
                    continue
                tasks.append(task)
                sample_index += 1
                break
            else:
                raise RuntimeError(
                    f"{cell.scene_name}/{cell.task_type.value} 无法完成任务采样配额"
                )
    return tasks


def _adjacent_levels(
    sampler: TaskSampler, cell: TaskCapability
) -> tuple[int, ...]:
    """返回该任务单元可表达的相邻占用等级。"""
    return sampler.adjacent_occupancy_levels(cell.scene_name, cell.task_type)


def _replacement_task(
    sampler: TaskSampler,
    task: Task,
    next_indices: dict[tuple[str, Any], int],
    reserved_task_ids: set[str],
) -> Task:
    key = (task.scene_name, task.task_type)
    for _ in range(100):
        sample_index = next_indices[key]
        next_indices[key] += 1
        try:
            replacement = sampler.sample(
                task.scene_name,
                task.task_type,
                sample_index=sample_index,
                maneuver=task.difficulty.maneuver,
                adjacent_occupancy=task.difficulty.adjacent_occupancy,
                noise_level=task.difficulty.noise_level,
            )
            if replacement.task_id in reserved_task_ids:
                continue
            return replacement
        except UnsupportedTaskError:
            continue
    raise RuntimeError(f"{task.scene_name}/{task.task_type.value} 无法重采同单元任务")


def _task_index(task: Task) -> int:
    coordinates = _task_id_coordinates(task.task_id)
    return 0 if coordinates is None else coordinates[2]


def _task_id_coordinates(
    task_id: str,
) -> tuple[str, TaskType, int] | None:
    match = re.match(r"^(.+)-(T[1-5])-(\d+)-[0-9a-f]+$", task_id)
    if match is None:
        return None
    return match.group(1), TaskType(match.group(2)), int(match.group(3))
