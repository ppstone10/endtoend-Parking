"""任务配额计划与失败重采测试。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset.build import build_task_plan, expert_maneuvers, generate_with_retries
from dataset.generator import DatasetGenerator, TaskGenerationError
from scripts.build_dataset import (
    _build_split_in_batches,
    _load_retry_state_or_default,
    _record_retry_failure,
    build_components,
)
from sim import MINING_DRILL_RIG
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
            expert_maneuvers("S7_fuel_station", TaskType.T1_NEAR), ()
        )
        self.assertEqual(
            expert_maneuvers("S9_mine_complex", TaskType.T3_LONG), ()
        )
        self.assertEqual(
            expert_maneuvers("S5_crusher", TaskType.T2_MEDIUM), ()
        )
        for task_type in (
            TaskType.T1_NEAR,
            TaskType.T5_DYNAMIC,
        ):
            with self.subTest(scene="S5_crusher", task_type=task_type):
                self.assertEqual(
                    expert_maneuvers("S5_crusher", task_type),
                    (Maneuver.REVERSE,),
                )
        for task_type in (TaskType.T1_NEAR, TaskType.T5_DYNAMIC):
            with self.subTest(scene="S9_mine_complex", task_type=task_type):
                self.assertEqual(
                    expert_maneuvers("S9_mine_complex", task_type),
                    (Maneuver.FORWARD,),
                )
        self.assertEqual(
            expert_maneuvers("S8_weigh_station", TaskType.T5_DYNAMIC),
            (Maneuver.FORWARD,),
        )
        for task_type in TaskType:
            with self.subTest(scene="S3_maintenance", task_type=task_type):
                self.assertEqual(
                    expert_maneuvers("S3_maintenance", task_type),
                    (Maneuver.REVERSE,),
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

        all_tasks = (*plan.train, *plan.val, *plan.test)
        excluded = {
            ("S7_fuel_station", TaskType.T1_NEAR),
            ("S9_mine_complex", TaskType.T3_LONG),
            ("S5_crusher", TaskType.T2_MEDIUM),
        }
        self.assertFalse(
            any((task.scene_name, task.task_type) in excluded for task in all_tasks)
        )
        for task in all_tasks:
            allowed = expert_maneuvers(task.scene_name, task.task_type)
            self.assertIn(task.difficulty.maneuver, allowed)

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
            "tracked_pivot_v4",
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


class _AlwaysFailGenerator:
    def __init__(self):
        self.attempted_task_ids = []

    def generate(self, tasks):
        task = list(tasks)[0]
        self.attempted_task_ids.append(task.task_id)
        raise TaskGenerationError(task.task_id, "持续测试失败")


class TestBuildRetries(unittest.TestCase):
    def test_retry_progress_callback_exposes_stable_failure_event(self):
        original = build_task_plan(total_count=20, seed=7).train[0]
        events = []
        generate_with_retries(
            [original],
            generator=_FailOnceGenerator(),
            seed=7,
            max_retries=2,
            progress_callback=events.append,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["original_task_id"], original.task_id)
        self.assertEqual(events[0]["retry"], 1)
        self.assertEqual(events[0]["failure_code"], "测试失败")
        self.assertEqual(events[0]["scene_name"], original.scene_name)
        self.assertEqual(events[0]["task_type"], original.task_type.value)
        self.assertGreater(events[0]["next_sample_index"], 0)

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

    def test_replacement_index_starts_after_all_reserved_plan_ids(self):
        plan = build_task_plan(total_count=20, seed=7)
        original = plan.train[0]
        reserved = {
            f"{original.scene_name}-{original.task_type.value}-{index:04d}-deadbeef"
            for index in range(121)
        }
        report = generate_with_retries(
            [original],
            generator=_FailOnceGenerator(),
            seed=7,
            max_retries=2,
            reserved_task_ids=reserved,
        )

        replacement_index = int(report.replacements[0].task_id.rsplit("-", 2)[1])
        self.assertGreaterEqual(replacement_index, 121)

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

    def test_resume_skips_a_previously_failed_original_task(self):
        original = build_task_plan(total_count=20, seed=7).train[0]
        report = generate_with_retries(
            [original],
            generator=_FailOnceGenerator(),
            seed=7,
            max_retries=2,
            excluded_task_ids={original.task_id},
            minimum_sample_indices={
                (original.scene_name, original.task_type): 121,
            },
        )

        first_attempt_id = report.replacements[0].task_id
        first_attempt_index = int(first_attempt_id.rsplit("-", 2)[1])
        self.assertNotEqual(first_attempt_id, original.task_id)
        self.assertGreaterEqual(first_attempt_index, 121)


class TestBatchRetryState(unittest.TestCase):
    def test_restarting_an_incomplete_batch_does_not_repeat_failed_ids(self):
        task = build_task_plan(total_count=20, seed=7).train[0]
        generator = _AlwaysFailGenerator()
        config = MINING_DRILL_RIG
        sampler = TaskSampler(
            seed=7,
            vehicle_length=config.length,
            vehicle_width=config.width,
            collision_margin=config.collision_margin,
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp, "dataset")
            checkpoint_root = output / ".checkpoints"
            checkpoint_root.mkdir(parents=True)
            arguments = {
                "split_name": "train",
                "tasks": (task,),
                "generator": generator,
                "vehicle_config": config,
                "checkpoint_root": checkpoint_root,
                "output": output,
                "seed": 7,
                "max_retries": 1,
                "batch_size": 1,
                "reserved_task_ids": {task.task_id},
                "task_sampler": sampler,
            }

            with self.assertRaisesRegex(RuntimeError, "请使用原命令续建"):
                _build_split_in_batches(**arguments)
            first_run_ids = tuple(generator.attempted_task_ids)
            with self.assertRaisesRegex(RuntimeError, "请使用原命令续建"):
                _build_split_in_batches(**arguments)
            second_run_ids = tuple(generator.attempted_task_ids[len(first_run_ids):])

            self.assertEqual(len(first_run_ids), 2)
            self.assertEqual(len(second_run_ids), 2)
            self.assertTrue(set(first_run_ids).isdisjoint(second_run_ids))
            retry_path = checkpoint_root / "train" / "part-00000.retry.json"
            state = json.loads(retry_path.read_text(encoding="utf-8"))
            self.assertEqual(state["failure_count"], 4)
            self.assertEqual(state["excluded_task_ids"], [*first_run_ids, *second_run_ids])

    def test_failure_state_is_atomic_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "part-00003.retry.json")
            state = _load_retry_state_or_default(
                path,
                split_name="train",
                batch_index=3,
                original_task_ids=("S4_dump_area-T3-0004-deadbeef",),
            )
            event = {
                "scene_name": "S4_dump_area",
                "task_type": "T3",
                "original_task_id": "S4_dump_area-T3-0004-deadbeef",
                "current_task_id": "S4_dump_area-T3-0123-feedface",
                "failure_code": "planner_timeout",
                "next_sample_index": 124,
                "retry": 11,
                "max_attempts": 11,
            }

            _record_retry_failure(path, state, event)
            restored = _load_retry_state_or_default(
                path,
                split_name="train",
                batch_index=3,
                original_task_ids=("S4_dump_area-T3-0004-deadbeef",),
            )

            self.assertEqual(restored["failure_count"], 1)
            self.assertEqual(restored["failure_reasons"], {"planner_timeout": 1})
            self.assertEqual(
                restored["excluded_task_ids"],
                ["S4_dump_area-T3-0123-feedface"],
            )
            self.assertEqual(
                restored["next_sample_indices"],
                {"S4_dump_area/T3": 124},
            )
            self.assertFalse(path.with_name(f"{path.name}.tmp").exists())

    def test_retry_state_rejects_a_different_batch_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "part-00003.retry.json")
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split_name": "train",
                        "batch_index": 3,
                        "original_task_ids": ["old-task"],
                        "failure_count": 0,
                        "failure_reasons": {},
                        "excluded_task_ids": [],
                        "next_sample_indices": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "retry 状态与当前批次不一致"):
                _load_retry_state_or_default(
                    path,
                    split_name="train",
                    batch_index=3,
                    original_task_ids=("new-task",),
                )


if __name__ == "__main__":
    unittest.main()
