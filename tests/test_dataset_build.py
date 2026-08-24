"""任务配额计划与失败重采测试。"""

import unittest

import numpy as np

from dataset.build import build_task_plan, expert_maneuvers, generate_with_retries
from dataset.generator import DatasetGenerator, TaskGenerationError
from scripts.build_dataset import build_components
from sim.tasks import Maneuver, TaskSampler, TaskType


class TestTaskPlan(unittest.TestCase):
    def test_plan_has_exact_ratios_and_s9_generalization_split(self):
        plan = build_task_plan(total_count=100, seed=123)
        self.assertEqual((len(plan.train), len(plan.val), len(plan.test)), (80, 10, 10))
        self.assertTrue(all(task.scene_name == "S9_mine_complex" for task in plan.test))
        self.assertTrue(
            all(task.scene_name != "S9_mine_complex" for task in (*plan.train, *plan.val))
        )
        self.assertEqual(
            {task.difficulty.noise_level.value for task in (*plan.train, *plan.val, *plan.test)},
            {"clean", "low", "high"},
        )

    def test_same_seed_produces_same_plan(self):
        left = build_task_plan(total_count=100, seed=321)
        right = build_task_plan(total_count=100, seed=321)
        self.assertEqual(left.task_ids(), right.task_ids())

    def test_invalid_ratios_are_rejected_explicitly(self):
        with self.assertRaises(ValueError):
            build_task_plan(total_count=100, seed=1, ratios=(0.9, 0.1))
        with self.assertRaises(ValueError):
            build_task_plan(total_count=100, seed=1, ratios=(0.7, 0.1, 0.1))

    def test_production_plan_covers_all_supported_cells(self):
        plan = build_task_plan(total_count=3000, seed=123)
        actual = {
            (task.scene_name, task.task_type)
            for task in (*plan.train, *plan.val, *plan.test)
        }
        expected = {
            (cell.scene_name, cell.task_type)
            for cell in TaskSampler(seed=123).capability_matrix()
            if cell.supported and expert_maneuvers(cell.scene_name, cell.task_type)
        }
        self.assertEqual(actual, expected)

    def test_expert_capability_excludes_persistently_unreachable_cells(self):
        self.assertEqual(
            expert_maneuvers("S5_crusher", TaskType.T3_LONG), ()
        )
        self.assertEqual(
            expert_maneuvers("S8_weigh_station", TaskType.T5_DYNAMIC),
            (Maneuver.FORWARD,),
        )
        plan = build_task_plan(total_count=3000, seed=123)
        s8_t5 = [
            task for task in (*plan.train, *plan.val)
            if task.scene_name == "S8_weigh_station"
            and task.task_type == TaskType.T5_DYNAMIC
        ]
        self.assertTrue(s8_t5)
        self.assertTrue(
            all(task.difficulty.maneuver == Maneuver.FORWARD for task in s8_t5)
        )

    def test_real_components_write_selected_goal_to_target_channel(self):
        task = TaskSampler(seed=321).sample(
            "S1_parking_lot",
            TaskType.T1_NEAR,
            maneuver=Maneuver.FORWARD,
        )
        sample = DatasetGenerator(component_factory=build_components).generate([task])[0]
        target_index = sample.bev.channels.index("target")
        self.assertGreater(np.count_nonzero(sample.bev.data[target_index] > 0.5), 0)
        self.assertGreater(sample.expert_trajectory.horizon, 1)
        self.assertTrue(sample.task_meta["dataset"]["feasibility_audit"]["feasible"])
        self.assertEqual(
            sample.task_meta["dataset"]["vehicle_model"]["model_version"],
            "tracked_pivot_v1",
        )


class _FailOnceGenerator:
    def __init__(self):
        self.failed = False

    def generate(self, tasks):
        task = list(tasks)[0]
        if not self.failed:
            self.failed = True
            raise TaskGenerationError(task.task_id, "测试失败")
        return [task.task_id]


class _FailConsistencyOnceGenerator(_FailOnceGenerator):
    def generate(self, tasks):
        task = list(tasks)[0]
        if not self.failed:
            self.failed = True
            raise TaskGenerationError(
                task.task_id,
                "请求 forward，实际前进 20.0%、倒车 80.0%",
                code="maneuver_inconsistent",
            )
        return [task.task_id]


class TestBuildRetries(unittest.TestCase):
    def test_failure_is_resampled_in_same_cell(self):
        plan = build_task_plan(total_count=20, seed=7)
        original = plan.train[0]
        report = generate_with_retries(
            [original],
            generator=_FailOnceGenerator(),
            seed=7,
            max_retries=2,
        )
        self.assertEqual(len(report.samples), 1)
        self.assertEqual(report.failure_count, 1)
        self.assertEqual(report.replacements[0].scene_name, original.scene_name)
        self.assertEqual(report.replacements[0].task_type, original.task_type)

    def test_replacement_skips_ids_reserved_by_other_splits(self):
        plan = build_task_plan(total_count=20, seed=7)
        original = plan.train[0]
        sampler = TaskSampler(seed=7)
        next_index = int(original.task_id.rsplit("-", 2)[1]) + 1
        blocked = sampler.sample(
            original.scene_name,
            original.task_type,
            sample_index=next_index,
            maneuver=original.difficulty.maneuver,
            adjacent_occupancy=original.difficulty.adjacent_occupancy,
            noise_level=original.difficulty.noise_level,
        )
        report = generate_with_retries(
            [original],
            generator=_FailOnceGenerator(),
            seed=7,
            max_retries=2,
            reserved_task_ids={blocked.task_id},
        )
        self.assertNotEqual(report.replacements[0].task_id, blocked.task_id)

    def test_maneuver_failures_use_stable_reason_and_preserve_difficulty(self):
        original = build_task_plan(total_count=20, seed=7).train[0]
        report = generate_with_retries(
            [original],
            generator=_FailConsistencyOnceGenerator(),
            seed=7,
            max_retries=2,
        )
        self.assertEqual(report.failure_reasons, {"maneuver_inconsistent": 1})
        self.assertEqual(
            report.replacements[0].difficulty.maneuver,
            original.difficulty.maneuver,
        )


if __name__ == "__main__":
    unittest.main()
