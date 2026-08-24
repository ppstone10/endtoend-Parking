"""任务数据集稳定分层与泛化场景隔离测试。"""

import unittest

from dataset.splits import split_tasks
from sim.tasks import TaskSampler, TaskType


def _tasks():
    sampler = TaskSampler(seed=20260824)
    tasks = []
    for scene in ("S1_parking_lot", "S2_diagonal_lot"):
        for task_type in (TaskType.T1_NEAR, TaskType.T2_MEDIUM):
            tasks.extend(sampler.sample(scene, task_type, index) for index in range(10))
    for task_type in (TaskType.T1_NEAR, TaskType.T2_MEDIUM):
        tasks.extend(
            sampler.sample("S9_mine_complex", task_type, index) for index in range(3)
        )
    return tasks


class TestDatasetSplits(unittest.TestCase):
    def test_holdout_scene_is_exclusive_and_splits_cover_input(self):
        tasks = _tasks()
        splits = split_tasks(tasks, seed=9, test_scene="S9_mine_complex")
        train_ids = {task.task_id for task in splits.train}
        val_ids = {task.task_id for task in splits.val}
        test_ids = {task.task_id for task in splits.test}

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(train_ids | val_ids | test_ids, {task.task_id for task in tasks})
        self.assertTrue(all(task.scene_name == "S9_mine_complex" for task in splits.test))
        self.assertTrue(
            all(task.scene_name != "S9_mine_complex" for task in (*splits.train, *splits.val))
        )

    def test_same_seed_is_reproducible_and_each_main_stratum_reaches_val(self):
        tasks = _tasks()
        left = split_tasks(tasks, seed=11)
        right = split_tasks(list(reversed(tasks)), seed=11)
        self.assertEqual(
            {task.task_id for task in left.val},
            {task.task_id for task in right.val},
        )
        val_strata = {(task.scene_name, task.task_type) for task in left.val}
        self.assertEqual(
            val_strata,
            {
                ("S1_parking_lot", TaskType.T1_NEAR),
                ("S1_parking_lot", TaskType.T2_MEDIUM),
                ("S2_diagonal_lot", TaskType.T1_NEAR),
                ("S2_diagonal_lot", TaskType.T2_MEDIUM),
            },
        )

    def test_duplicate_ids_and_missing_holdout_are_rejected(self):
        tasks = _tasks()
        with self.assertRaises(ValueError):
            split_tasks([tasks[0], tasks[0], *tasks[1:]])
        with self.assertRaises(ValueError):
            split_tasks([task for task in tasks if task.scene_name != "S9_mine_complex"])


if __name__ == "__main__":
    unittest.main()
