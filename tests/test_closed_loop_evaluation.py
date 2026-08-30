"""数据集驱动网络闭环评测测试。"""

import unittest

from experiments.closed_loop_evaluation import (
    reconstruct_dataset_task,
    select_evaluation_indices,
)
from sim import Maneuver, NoiseLevel, TaskSampler, TaskType, MINING_DRILL_RIG


class TestDatasetTaskReconstruction(unittest.TestCase):
    def _sampler(self) -> TaskSampler:
        return TaskSampler(
            seed=20260824,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )

    def test_reconstructs_scene_occupancy_noise_and_selected_goal(self):
        task = self._sampler().sample(
            "S1_parking_lot",
            TaskType.T2_MEDIUM,
            17,
            maneuver=Maneuver.REVERSE,
            adjacent_occupancy=1,
            noise_level=NoiseLevel.HIGH,
        )
        metadata = task.to_metadata()
        metadata["dataset"] = {
            "selected_goal": task.goal.to_metadata(),
            "vehicle_model": MINING_DRILL_RIG.to_metadata(),
        }

        restored = reconstruct_dataset_task(
            metadata,
            root_seed=20260824,
            vehicle=MINING_DRILL_RIG,
        )

        self.assertEqual(restored.task.task_id, task.task_id)
        self.assertEqual(restored.task.difficulty.noise_level, NoiseLevel.HIGH)
        self.assertEqual(restored.task.difficulty.adjacent_occupancy, 1)
        self.assertEqual(restored.goal_meta["spot_id"], task.goal.spot_id)
        self.assertEqual(
            [spot.occupied for spot in restored.task.scene.spots],
            [spot.occupied for spot in task.scene.spots],
        )

    def test_reconstructs_t4_selected_candidate(self):
        task = self._sampler().sample(
            "S9_mine_complex",
            TaskType.T4_MULTI_SPOT,
            9,
            maneuver=Maneuver.FORWARD,
            adjacent_occupancy=0,
            noise_level=NoiseLevel.CLEAN,
        )
        selected = task.candidate_goals[1]
        metadata = task.to_metadata()
        metadata["dataset"] = {
            "selected_goal": selected.to_metadata(),
            "vehicle_model": MINING_DRILL_RIG.to_metadata(),
        }

        restored = reconstruct_dataset_task(
            metadata,
            root_seed=20260824,
            vehicle=MINING_DRILL_RIG,
        )

        self.assertEqual(restored.goal_meta["spot_id"], selected.spot_id)
        self.assertAlmostEqual(restored.goal.x, selected.x)
        self.assertAlmostEqual(restored.tol_pos, selected.tol_pos)
        self.assertAlmostEqual(restored.tol_yaw, selected.tol_yaw)

    def test_rejects_task_identity_drift(self):
        task = self._sampler().sample("S1_parking_lot", TaskType.T1_NEAR, 3)
        metadata = task.to_metadata()
        metadata["task_id"] = metadata["task_id"].replace("-0003-", "-0004-")
        metadata["dataset"] = {
            "selected_goal": task.goal.to_metadata(),
            "vehicle_model": MINING_DRILL_RIG.to_metadata(),
        }

        with self.assertRaisesRegex(ValueError, "任务身份"):
            reconstruct_dataset_task(
                metadata,
                root_seed=20260824,
                vehicle=MINING_DRILL_RIG,
            )


class TestEvaluationSelection(unittest.TestCase):
    def test_stratified_selection_covers_scene_task_groups_first(self):
        metadata = [
            {"scene_name": "S1", "task_type": "T1"},
            {"scene_name": "S1", "task_type": "T1"},
            {"scene_name": "S1", "task_type": "T2"},
            {"scene_name": "S2", "task_type": "T1"},
            {"scene_name": "S2", "task_type": "T1"},
        ]

        selected = select_evaluation_indices(
            metadata, samples=3, strategy="stratified"
        )

        self.assertEqual(selected, [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
