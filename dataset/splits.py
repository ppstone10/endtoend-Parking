"""任务级稳定分层与整场景泛化划分。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sim.tasks import Task


@dataclass(frozen=True)
class DatasetSplits:
    """互斥的 train/val/test Task 集合。"""

    train: tuple[Task, ...]
    val: tuple[Task, ...]
    test: tuple[Task, ...]

    def task_ids(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": tuple(task.task_id for task in self.train),
            "val": tuple(task.task_id for task in self.val),
            "test": tuple(task.task_id for task in self.test),
        }


def split_tasks(
    tasks: Iterable[Task],
    *,
    seed: int = 0,
    test_scene: str = "S9_mine_complex",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> DatasetSplits:
    """保留完整测试场景，并在其余任务内按场景×类型分层选择 val。"""
    task_list = tuple(tasks)
    _validate_inputs(task_list, seed, ratios)

    test = [task for task in task_list if task.scene_name == test_scene]
    remaining = [task for task in task_list if task.scene_name != test_scene]
    if not test:
        raise ValueError(f"测试保留场景 {test_scene} 没有样本")
    if len(remaining) < 2:
        raise ValueError("非测试任务不足，无法划分 train/val")

    val_target = round(len(task_list) * ratios[1])
    groups: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for task in remaining:
        groups[(task.scene_name, task.task_type.value)].append(task)
    quotas = _validation_quotas(groups, val_target)

    val_ids: set[str] = set()
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda task: task.task_id)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, *key[0].encode("utf-8"), *key[1].encode("utf-8")])
        )
        permutation = rng.permutation(len(ordered))
        val_ids.update(ordered[int(index)].task_id for index in permutation[: quotas[key]])

    train = [task for task in remaining if task.task_id not in val_ids]
    val = [task for task in remaining if task.task_id in val_ids]
    return DatasetSplits(
        train=tuple(sorted(train, key=lambda task: task.task_id)),
        val=tuple(sorted(val, key=lambda task: task.task_id)),
        test=tuple(sorted(test, key=lambda task: task.task_id)),
    )


def _validate_inputs(
    tasks: tuple[Task, ...], seed: int, ratios: tuple[float, float, float]
) -> None:
    if not tasks:
        raise ValueError("不能划分空 Task 集合")
    if seed < 0:
        raise ValueError("划分 seed 不能为负")
    if len(ratios) != 3 or any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("train/val/test 比例必须包含三个正数")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("train/val/test 比例之和必须为 1")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task ID 必须唯一，禁止跨 split 重复")


def _validation_quotas(
    groups: dict[tuple[str, str], list[Task]], val_target: int
) -> dict[tuple[str, str], int]:
    capacities = {key: max(0, len(value) - 1) for key, value in groups.items()}
    if val_target <= 0 or val_target > sum(capacities.values()):
        raise ValueError("目标 val 数量无法在保留 train 样本的前提下分配")

    total = sum(len(value) for value in groups.values())
    ideals = {key: val_target * len(value) / total for key, value in groups.items()}
    quotas = {
        key: min(capacities[key], int(np.floor(ideal))) for key, ideal in ideals.items()
    }
    if val_target >= len(groups):
        for key in groups:
            if capacities[key] > 0:
                quotas[key] = max(1, quotas[key])

    while sum(quotas.values()) < val_target:
        candidates = [key for key in groups if quotas[key] < capacities[key]]
        if not candidates:
            raise ValueError("无法完成 val 分层配额")
        key = max(
            candidates,
            key=lambda item: (ideals[item] - quotas[item], len(groups[item]), item),
        )
        quotas[key] += 1
    while sum(quotas.values()) > val_target:
        candidates = [key for key in groups if quotas[key] > 0]
        key = min(
            candidates,
            key=lambda item: (ideals[item] - quotas[item], -len(groups[item]), item),
        )
        quotas[key] -= 1
    return quotas
