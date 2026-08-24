"""场景库测试：S1–S9 构造、自检与关键语义。"""

import unittest

import numpy as np

from sim.scenes import SCENE_REGISTRY, build_scene
from sim.scenes_validate import validate_scene
from sim.spots import ParkingSpot


ALL_SCENES = [
    "S1_parking_lot", "S2_diagonal_lot", "S3_maintenance", "S4_dump_area",
    "S5_crusher", "S6_loading_face", "S7_fuel_station", "S8_weigh_station",
    "S9_mine_complex",
]


class TestSceneRegistry(unittest.TestCase):
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
