"""场景库测试：S1–S9 构造、自检与关键语义。"""

import math
import unittest

import numpy as np

from interfaces import VehicleState
from planner import HybridAStarPlanner
from sim import Maneuver, MINING_DRILL_RIG, NoiseLevel, TaskSampler, TaskType
from sim.scenes import SCENE_REGISTRY, build_scene
from sim.scenes_validate import validate_scene
from sim.spots import ParkingSpot


ALL_SCENES = [
    "S1_parking_lot", "S2_diagonal_lot", "S3_maintenance", "S4_dump_area",
    "S5_crusher", "S6_loading_face", "S7_fuel_station", "S8_weigh_station",
    "S9_mine_complex",
]


class TestSceneRegistry(unittest.TestCase):
    def test_scene_bev_configs_keep_common_grid_shape(self):
        for name in ALL_SCENES[:-1]:
            with self.subTest(scene=name):
                self.assertEqual(
                    build_scene(name).bev_config.extent,
                    (20.0, 20.0, 20.0, 20.0),
                )
        default_scene = build_scene("S1_parking_lot")
        long_range_scene = build_scene("S9_mine_complex")
        self.assertEqual(default_scene.bev_config.extent, (20.0, 20.0, 20.0, 20.0))
        self.assertEqual(long_range_scene.bev_config.extent, (40.0, 40.0, 40.0, 40.0))
        self.assertEqual(default_scene.bev_config.shape, (160, 160))
        self.assertEqual(long_range_scene.bev_config.shape, (160, 160))

    def test_all_nine_scenes_registered(self):
        for name in ALL_SCENES:
            self.assertIn(name, SCENE_REGISTRY)
        self.assertEqual(len([n for n in SCENE_REGISTRY if n.startswith("S")]), 9)

    def test_unknown_scene_raises(self):
        with self.assertRaises(ValueError):
            build_scene("S99_nothing")


class TestSceneValidation(unittest.TestCase):
    def test_all_scenes_pass_validation(self):
        for name in ALL_SCENES:
            with self.subTest(scene=name):
                bundle = build_scene(name)
                errors = validate_scene(bundle)
                self.assertEqual(errors, [], f"{name} 自检失败: {errors}")


class TestSceneSemantics(unittest.TestCase):
    def test_s1_occupied_spot_creates_obstacle(self):
        free = build_scene("S1_parking_lot")
        occ = build_scene("S1_parking_lot", occupied_pattern=[0, 2])
        self.assertEqual(len(free.env.obstacles) + 2, len(occ.env.obstacles))
        self.assertEqual(len(occ.free_spots()), len(free.free_spots()) - 2)

    def test_s4_cliff_behind_berm(self):
        """S4：挡墙挡射线，悬崖禁入不挡射线。"""
        bundle = build_scene("S4_dump_area")
        env = bundle.env
        # 挡墙在 y≈6：从场内朝 +y 的射线应打在挡墙 y=6 处（x 车道中心）。
        spot = bundle.free_spots()[0]
        dist = env.raycast(np.array([spot.pose.x, spot.pose.y]), np.pi / 2, 30.0)
        self.assertAlmostEqual(dist, 6.0 - spot.pose.y, delta=0.05)
        # 悬崖内部禁入。
        self.assertFalse(env.is_free(spot.pose.x, 6.8))

    def test_s5_slot_clearance(self):
        """S5：槽内目标位姿 6×3 车辆自由，槽底料口禁区。"""
        bundle = build_scene("S5_crusher")
        spot = bundle.spots[0]
        self.assertTrue(bundle.env.is_free(spot.pose.x, spot.pose.y))
        # 槽底（y > 8.0）为禁区。
        self.assertFalse(bundle.env.is_free(spot.pose.x, 8.6))

    def test_s8_line_marking_traversable(self):
        """S8：称重台标线可通行（目标位姿在标线上）。"""
        bundle = build_scene("S8_weigh_station")
        spot = bundle.spots[0]
        self.assertTrue(bundle.env.is_free(spot.pose.x, spot.pose.y))

    def test_s9_has_all_three_zones(self):
        bundle = build_scene("S9_mine_complex")
        kinds = {s.kind for s in bundle.spots}
        self.assertIn("spot", kinds)          # 停车场
        self.assertIn("berm_bay", kinds)      # 卸载区
        self.assertIn("crusher_slot", kinds)  # 破碎站
        self.assertGreaterEqual(len(bundle.spawn_zones), 2)

    def test_spot_boxes_contain_vehicle(self):
        """全部场景车位框必须能容纳 6×3 矿卡（长≥6、宽≥3，留余量）。"""
        for name in ALL_SCENES:
            with self.subTest(scene=name):
                bundle = build_scene(name)
                for s in bundle.spots:
                    length, width = s.size
                    self.assertGreaterEqual(
                        length, 6.0, f"{name}/{s.id} 车位长度 {length} 小于车长 6m"
                    )
                    self.assertGreaterEqual(
                        width, 3.0 + 0.25, f"{name}/{s.id} 车位宽度 {width} 余量不足"
                    )

    def test_s6_equipment_matches_truck_scale(self):
        """S6 装载设备为挖掘机/装载机量级：单体不超过矿卡量级太多（≤8m）。"""
        bundle = build_scene("S6_loading_face")
        for obs in bundle.env.obstacles:
            if obs.kind == "equipment":
                x0, x1, y0, y1 = obs.bbox
                self.assertLessEqual(max(x1 - x0, y1 - y0), 8.0)


class TestMineScaleContract(unittest.TestCase):
    def _dump_scene(self, name):
        return build_scene(
            name,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )

    def test_dump_bays_keep_vehicle_relative_spacing_and_clearance(self):
        required_rear_clearance = MINING_DRILL_RIG.collision_margin + 0.3
        for name, prefix, berm_y in (
            ("S4_dump_area", "B", 6.0),
            ("S9_mine_complex", "DB", 2.0),
        ):
            with self.subTest(scene=name):
                bundle = self._dump_scene(name)
                spots = [spot for spot in bundle.spots if spot.id.startswith(prefix)]
                self.assertGreaterEqual(len(spots), 2)
                self.assertGreaterEqual(
                    spots[1].pose.x - spots[0].pose.x,
                    3.0 * MINING_DRILL_RIG.width,
                )
                for spot in spots:
                    self.assertAlmostEqual(spot.pose.yaw, -np.pi / 2)
                    rear_y = spot.pose.y - (
                        MINING_DRILL_RIG.length / 2.0 * np.sin(spot.pose.yaw)
                    )
                    self.assertGreaterEqual(
                        berm_y - rear_y,
                        required_rear_clearance - 1e-9,
                    )
                self.assertEqual(
                    bundle.difficulty_knobs["geometry_profile"],
                    "vehicle_relative_v1",
                )

    def test_known_s4_t3_timeout_pose_plans_with_safety_margin(self):
        bundle = self._dump_scene("S4_dump_area")
        goal = next(spot.pose for spot in bundle.spots if spot.id == "B2")
        planner = HybridAStarPlanner(bundle.env, **MINING_DRILL_RIG.planner_kwargs())
        trajectory = planner.plan(
            VehicleState(-10.584733, -4.703514, 1.195006, 0.0, 0.0),
            goal,
            preferred_direction=1,
        )
        self.assertGreater(len(trajectory.points), 2)

    def test_s3_maintenance_bays_are_vehicle_relative(self):
        bundle = self._dump_scene("S3_maintenance")
        self.assertEqual(
            bundle.difficulty_knobs["geometry_profile"], "vehicle_relative_v1"
        )
        self.assertEqual(bundle.difficulty_knobs["clearance"], 0.6)
        self.assertGreaterEqual(len(bundle.spots), 3)
        for spot in bundle.spots:
            self.assertEqual(spot.kind, "maintenance_bay")
            self.assertAlmostEqual(spot.pose.yaw, -np.pi / 2)
            # bay 宽度 = 车宽 + 2×单侧净空。
            self.assertAlmostEqual(
                spot.size[1], MINING_DRILL_RIG.width + 2 * 0.6
            )
        # 作业道深度需支撑 T3(15–30m) 起点：spawn 区下缘与 bay 的距离上限 ≥ 20m。
        spawn = bundle.spawn_zones[0]
        farthest = max(
            math.hypot(s.pose.x - x, s.pose.y - y)
            for s in bundle.spots
            for x in (spawn[0], spawn[1])
            for y in (spawn[2], spawn[3])
        )
        self.assertGreaterEqual(farthest, 20.0)

    def test_s7_fuel_bay_keeps_margin_aware_island_clearance(self):
        bundle = build_scene(
            "S7_fuel_station",
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        island_top = 2.4 / 2.0
        for spot in bundle.spots:
            # 车体近岛侧边缘与岛面保持 ≥0.3m 物理净空（计入碰撞裕量）。
            near_edge = spot.pose.y - (
                MINING_DRILL_RIG.width / 2.0 + MINING_DRILL_RIG.collision_margin
            )
            self.assertGreaterEqual(near_edge - island_top, 0.3 - 1e-9)
        sampler = TaskSampler(
            seed=20260824,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        for sample_index in (296, 300):
            with self.subTest(sample_index=sample_index):
                task = sampler.sample(
                    "S4_dump_area",
                    TaskType.T3_LONG,
                    sample_index=sample_index,
                    maneuver=Maneuver.FORWARD,
                    adjacent_occupancy=0,
                    noise_level=NoiseLevel.CLEAN,
                )
                planner = HybridAStarPlanner(
                    task.scene.env,
                    **MINING_DRILL_RIG.planner_kwargs(),
                )
                trajectory = planner.plan(
                    task.start,
                    task.goal.as_goal_pose(),
                    preferred_direction=1,
                )
                self.assertGreater(len(trajectory.points), 2)


class TestParkingSpot(unittest.TestCase):
    def test_occupant_obstacle_bounds(self):
        spot = ParkingSpot(id="t", pose=__import__("interfaces").GoalPose(10.0, 5.0, np.pi / 2))
        occ = spot.occupant_obstacle(6.0, 3.0)
        # yaw=90°：车沿 y 向 → x 半宽 3/2、y 半长 6/2。
        self.assertAlmostEqual(occ.x_min, 10.0 - 1.5)
        self.assertAlmostEqual(occ.x_max, 10.0 + 1.5)
        self.assertAlmostEqual(occ.y_min, 5.0 - 3.0)
        self.assertAlmostEqual(occ.y_max, 5.0 + 3.0)


if __name__ == "__main__":
    unittest.main()
