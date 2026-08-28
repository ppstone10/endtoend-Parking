"""任务层测试：T1–T5 契约、可复现采样与 9×5 能力矩阵。"""

from dataclasses import replace
import json
import math
import unittest

from sim.tasks import (
    Maneuver,
    NoiseLevel,
    TaskSampler,
    TaskType,
    UnsupportedTaskError,
)
from planner.hybrid_astar import HybridAStarPlanner
from sim import MINING_DRILL_RIG, ParkingEnvironment, RectangleObstacle


class TestTaskModel(unittest.TestCase):
    def setUp(self):
        self.sampler = TaskSampler(seed=20260824)

    def test_task_metadata_is_json_serializable(self):
        task = self.sampler.sample("S1_parking_lot", TaskType.T1_NEAR)
        encoded = json.dumps(task.to_metadata(), sort_keys=True)
        self.assertIn('"schema_version": 1', encoded)
        self.assertIn('"task_type": "T1"', encoded)

    def test_single_goal_and_candidate_goal_invariants(self):
        single = self.sampler.sample("S1_parking_lot", TaskType.T1_NEAR)
        self.assertIsNotNone(single.goal)
        self.assertEqual(single.candidate_goals, ())
        with self.assertRaises(ValueError):
            replace(single, goal=None)

        multi = self.sampler.sample("S1_parking_lot", TaskType.T4_MULTI_SPOT)
        self.assertIsNone(multi.goal)
        self.assertGreaterEqual(len(multi.candidate_goals), 3)
        self.assertLessEqual(len(multi.candidate_goals), 6)
        with self.assertRaises(ValueError):
            replace(multi, goal=multi.candidate_goals[0])


class TestTaskSampling(unittest.TestCase):
    def test_full_footprint_rejects_obstacle_between_center_and_corners(self):
        sampler = TaskSampler(seed=1, vehicle_length=6.0, vehicle_width=3.0)
        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(0.8, 1.2, -0.2, 0.2)],
        )

        self.assertFalse(sampler.pose_is_free(env, 0.0, 0.0, 0.0))

    def test_same_coordinates_have_same_metadata(self):
        left = TaskSampler(seed=1234).sample(
            "S2_diagonal_lot", TaskType.T2_MEDIUM, sample_index=7,
            maneuver=Maneuver.REVERSE, noise_level=NoiseLevel.LOW,
        )
        right = TaskSampler(seed=1234).sample(
            "S2_diagonal_lot", TaskType.T2_MEDIUM, sample_index=7,
            maneuver=Maneuver.REVERSE, noise_level=NoiseLevel.LOW,
        )
        self.assertEqual(left.to_metadata(), right.to_metadata())

    def test_sample_index_derives_independent_identity(self):
        sampler = TaskSampler(seed=1234)
        first = sampler.sample("S1_parking_lot", TaskType.T1_NEAR, sample_index=0)
        second = sampler.sample("S1_parking_lot", TaskType.T1_NEAR, sample_index=1)
        self.assertNotEqual(first.task_id, second.task_id)
        self.assertNotEqual(first.seed, second.seed)

    def test_distance_tiers_and_vehicle_footprint(self):
        sampler = TaskSampler(seed=42)
        cases = [
            ("S1_parking_lot", TaskType.T1_NEAR, 4.0, 8.0),
            ("S2_diagonal_lot", TaskType.T2_MEDIUM, 8.0, 15.0),
            ("S9_mine_complex", TaskType.T3_LONG, 15.0, 30.0),
        ]
        for scene, kind, lower, upper in cases:
            with self.subTest(scene=scene, kind=kind):
                task = sampler.sample(scene, kind)
                self.assertIsNotNone(task.goal)
                distance = math.hypot(task.start.x - task.goal.x, task.start.y - task.goal.y)
                self.assertGreaterEqual(distance, lower)
                self.assertLessEqual(distance, upper)
                self.assertTrue(
                    sampler.pose_is_free(task.scene.env, task.start.x, task.start.y, task.start.yaw)
                )

    def test_requested_difficulty_axes_are_recorded(self):
        task = TaskSampler(seed=9).sample(
            "S1_parking_lot", TaskType.T2_MEDIUM,
            maneuver=Maneuver.REVERSE, adjacent_occupancy=2,
            noise_level=NoiseLevel.HIGH,
        )
        self.assertEqual(task.difficulty.maneuver, Maneuver.REVERSE)
        self.assertEqual(task.difficulty.adjacent_occupancy, 2)
        self.assertEqual(task.difficulty.noise_level, NoiseLevel.HIGH)
        self.assertEqual(task.difficulty.aisle_width, 12.0)
        occupied = [spot for spot in task.scene.spots if spot.occupied]
        self.assertEqual(len(occupied), 2)
        target_x = task.goal.x
        self.assertTrue(any(spot.pose.x < target_x for spot in occupied))
        self.assertTrue(any(spot.pose.x > target_x for spot in occupied))

    def test_unrepresentable_adjacent_occupancy_is_rejected(self):
        with self.assertRaises(UnsupportedTaskError):
            TaskSampler(seed=9).sample(
                "S6_loading_face", TaskType.T2_MEDIUM, adjacent_occupancy=1
            )

    def test_t4_rejects_occupancy_that_leaves_too_few_candidates(self):
        with self.assertRaises(UnsupportedTaskError):
            TaskSampler(seed=9).sample(
                "S4_dump_area", TaskType.T4_MULTI_SPOT, adjacent_occupancy=2
            )

    def test_adjacent_occupancy_levels_respect_scene_and_t4_capacity(self):
        sampler = TaskSampler(seed=9)
        self.assertEqual(
            sampler.adjacent_occupancy_levels("S7_fuel_station", TaskType.T1_NEAR),
            (0, 1),
        )
        self.assertEqual(
            sampler.adjacent_occupancy_levels("S4_dump_area", TaskType.T4_MULTI_SPOT),
            (0, 1),
        )
        self.assertEqual(
            sampler.adjacent_occupancy_levels("S6_loading_face", TaskType.T1_NEAR),
            (0,),
        )
        self.assertEqual(
            sampler.adjacent_occupancy_levels("S2_diagonal_lot", TaskType.T1_NEAR),
            (0,),
        )

    def test_adjacent_occupancy_keeps_target_vehicle_footprint_safe(self):
        with self.assertRaisesRegex(UnsupportedTaskError, "保持目标安全"):
            TaskSampler(seed=9).sample(
                "S2_diagonal_lot", TaskType.T1_NEAR, adjacent_occupancy=1
            )


class TestTaskMatrix(unittest.TestCase):
    def setUp(self):
        self.sampler = TaskSampler(seed=77)

    def test_capability_matrix_contains_all_45_cells(self):
        cells = self.sampler.capability_matrix()
        self.assertEqual(len(cells), 45)
        self.assertEqual(len({(c.scene_name, c.task_type) for c in cells}), 45)
        unsupported = [cell for cell in cells if not cell.supported]
        self.assertTrue(unsupported)
        self.assertTrue(all(cell.reason for cell in unsupported))

    def test_non_strict_matrix_only_returns_supported_tasks(self):
        cells = self.sampler.capability_matrix()
        tasks = self.sampler.sample_matrix()
        self.assertEqual(len(tasks), sum(cell.supported for cell in cells))
        self.assertEqual(
            {(task.scene.name, task.task_type) for task in tasks},
            {(cell.scene_name, cell.task_type) for cell in cells if cell.supported},
        )

    def test_strict_matrix_rejects_unsupported_cells(self):
        with self.assertRaises(UnsupportedTaskError):
            self.sampler.sample_matrix(strict=True)

    def test_single_spot_scene_does_not_fake_t4(self):
        cell = next(
            c for c in self.sampler.capability_matrix()
            if c.scene_name == "S6_loading_face" and c.task_type == TaskType.T4_MULTI_SPOT
        )
        self.assertFalse(cell.supported)
        with self.assertRaises(UnsupportedTaskError):
            self.sampler.sample("S6_loading_face", TaskType.T4_MULTI_SPOT)


class TestTaskDynamicEvent(unittest.TestCase):
    def test_t5_has_one_progress_triggered_event_without_mutating_environment(self):
        task = TaskSampler(seed=55).sample("S8_weigh_station", TaskType.T5_DYNAMIC)
        self.assertIsNotNone(task.dynamic_event)
        event = task.dynamic_event
        self.assertGreater(event.trigger_progress, 0.0)
        self.assertLess(event.trigger_progress, 1.0)
        self.assertGreater(event.radius, 0.0)
        self.assertEqual(event.obstacle_kind, "vehicle")
        self.assertTrue(task.scene.env.is_free(event.x, event.y))
        self.assertFalse(any(getattr(obs, "x", None) == event.x for obs in task.scene.env.obstacles))


class TestTaskPlannerIntegration(unittest.TestCase):
    def test_representative_t1_to_t5_tasks_are_plannable(self):
        sampler = TaskSampler(seed=20260824)
        cases = [
            ("S1_parking_lot", TaskType.T1_NEAR),
            ("S3_maintenance", TaskType.T2_MEDIUM),
            ("S6_loading_face", TaskType.T3_LONG),
            ("S1_parking_lot", TaskType.T4_MULTI_SPOT),
            ("S8_weigh_station", TaskType.T5_DYNAMIC),
        ]
        for scene_name, task_type in cases:
            with self.subTest(scene=scene_name, task_type=task_type):
                task = sampler.sample(scene_name, task_type)
                target = task.goal or task.candidate_goals[0]
                planner = HybridAStarPlanner(
                    task.scene.env, vehicle_length=6.0, vehicle_width=3.0
                )
                trajectory = planner.plan(task.start, target.as_goal_pose())
                self.assertGreater(trajectory.horizon, 1)

    def test_axial_bay_long_distance_start_is_aligned_and_plannable(self):
        """紧 bay 长距起点必须与 bay 轴线对齐（yaw==gyaw）且可规划。"""
        sampler = TaskSampler(
            seed=20260824,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        cases = [
            ("S3_maintenance", TaskType.T3_LONG, Maneuver.REVERSE, (15.0, 30.0)),
            ("S5_crusher", TaskType.T2_MEDIUM, Maneuver.REVERSE, (8.0, 15.0)),
            ("S7_fuel_station", TaskType.T2_MEDIUM, Maneuver.FORWARD, (8.0, 15.0)),
        ]
        for scene_name, task_type, maneuver, bounds in cases:
            for sample_index in range(3):
                with self.subTest(
                    scene=scene_name, task_type=task_type,
                    maneuver=maneuver, sample_index=sample_index,
                ):
                    task = sampler.sample(
                        scene_name, task_type, sample_index=sample_index,
                        maneuver=maneuver, adjacent_occupancy=0,
                        noise_level=NoiseLevel.CLEAN,
                    )
                    goal = task.goal.as_goal_pose()
                    dist = math.hypot(goal.x - task.start.x, goal.y - task.start.y)
                    self.assertGreaterEqual(dist, bounds[0])
                    self.assertLessEqual(dist, bounds[1])
                    self.assertAlmostEqual(
                        math.sin(task.start.yaw - goal.yaw), 0.0, places=6
                    )
                    planner = HybridAStarPlanner(
                        task.scene.env, **MINING_DRILL_RIG.planner_kwargs()
                    )
                    pdir = 1 if maneuver == Maneuver.FORWARD else -1
                    trajectory = planner.plan(
                        task.start, goal, preferred_direction=pdir
                    )
                    self.assertGreater(trajectory.horizon, 1)

    def test_bidirectional_parking_goal_heading_matches_entry_direction(self):
        sampler = TaskSampler(
            seed=20260824,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        forward = sampler.sample(
            "S1_parking_lot", TaskType.T2_MEDIUM, sample_index=4,
            maneuver=Maneuver.FORWARD,
        )
        reverse = sampler.sample(
            "S1_parking_lot", TaskType.T2_MEDIUM, sample_index=4,
            maneuver=Maneuver.REVERSE,
        )
        self.assertEqual(forward.goal.spot_id, reverse.goal.spot_id)
        self.assertAlmostEqual(
            abs(math.sin(reverse.goal.yaw - forward.goal.yaw)), 0.0, places=6
        )
        self.assertAlmostEqual(
            math.cos(reverse.goal.yaw - forward.goal.yaw), -1.0, places=6
        )
        for task, expected_side in ((forward, -1.0), (reverse, 1.0)):
            goal = task.goal.as_goal_pose()
            nx, ny = math.cos(goal.yaw), math.sin(goal.yaw)
            axial_offset = (
                (task.start.x - goal.x) * nx + (task.start.y - goal.y) * ny
            )
            self.assertGreater(axial_offset * expected_side, 0.0)
            self.assertAlmostEqual(task.start.yaw, goal.yaw, places=6)

    def test_previously_failing_admitted_tasks_have_clear_plannable_approaches(self):
        sampler = TaskSampler(
            seed=20260824,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        cases = [
            ("S1_parking_lot", TaskType.T1_NEAR, 8, Maneuver.FORWARD, 2),
            ("S1_parking_lot", TaskType.T2_MEDIUM, 10, Maneuver.FORWARD, 1),
            ("S1_parking_lot", TaskType.T5_DYNAMIC, 11, Maneuver.REVERSE, 2),
            ("S7_fuel_station", TaskType.T2_MEDIUM, 3, Maneuver.REVERSE, 1),
            ("S7_fuel_station", TaskType.T2_MEDIUM, 9, Maneuver.REVERSE, 1),
            ("S7_fuel_station", TaskType.T2_MEDIUM, 11, Maneuver.REVERSE, 1),
            ("S9_mine_complex", TaskType.T2_MEDIUM, 0, Maneuver.REVERSE, 0),
            ("S9_mine_complex", TaskType.T2_MEDIUM, 1, Maneuver.REVERSE, 1),
            ("S9_mine_complex", TaskType.T2_MEDIUM, 6, Maneuver.REVERSE, 0),
            ("S9_mine_complex", TaskType.T2_MEDIUM, 7, Maneuver.REVERSE, 1),
            ("S9_mine_complex", TaskType.T2_MEDIUM, 8, Maneuver.REVERSE, 2),
            ("S9_mine_complex", TaskType.T5_DYNAMIC, 1, Maneuver.REVERSE, 1),
            ("S9_mine_complex", TaskType.T5_DYNAMIC, 4, Maneuver.REVERSE, 1),
            ("S9_mine_complex", TaskType.T5_DYNAMIC, 7, Maneuver.REVERSE, 1),
            ("S9_mine_complex", TaskType.T5_DYNAMIC, 8, Maneuver.REVERSE, 2),
        ]
        for scene_name, task_type, sample_index, maneuver, occupancy in cases:
            with self.subTest(
                scene=scene_name, task_type=task_type,
                sample_index=sample_index, maneuver=maneuver,
            ):
                task = sampler.sample(
                    scene_name, task_type, sample_index=sample_index,
                    maneuver=maneuver, adjacent_occupancy=occupancy,
                    noise_level=NoiseLevel.CLEAN,
                )
                goal = task.goal or task.candidate_goals[0]
                planner = HybridAStarPlanner(
                    task.scene.env, **MINING_DRILL_RIG.planner_kwargs()
                )
                trajectory = planner.plan(
                    task.start,
                    goal.as_goal_pose(),
                    preferred_direction=(1 if maneuver == Maneuver.FORWARD else -1),
                )
                self.assertGreater(trajectory.horizon, 1)

    def test_t4_reference_goal_is_first_and_plannable(self):
        sampler = TaskSampler(
            seed=20260824,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        cases = [
            ("S3_maintenance", 5, Maneuver.REVERSE, 0),
            ("S9_mine_complex", 5, Maneuver.REVERSE, 2),
            ("S9_mine_complex", 6, Maneuver.FORWARD, 0),
        ]
        for scene_name, sample_index, maneuver, occupancy in cases:
            with self.subTest(scene=scene_name, sample_index=sample_index):
                task = sampler.sample(
                    scene_name, TaskType.T4_MULTI_SPOT,
                    sample_index=sample_index, maneuver=maneuver,
                    adjacent_occupancy=occupancy,
                )
                goal = task.candidate_goals[0]
                planner = HybridAStarPlanner(
                    task.scene.env, **MINING_DRILL_RIG.planner_kwargs()
                )
                trajectory = planner.plan(
                    task.start,
                    goal.as_goal_pose(),
                    preferred_direction=(1 if maneuver == Maneuver.FORWARD else -1),
                )
                self.assertGreater(trajectory.horizon, 1)


if __name__ == "__main__":
    unittest.main()
