"""BEV 输入改造的验收测试：语义高度、几何 target、融合替换。"""

from __future__ import annotations

import unittest

import numpy as np

from dataset import GroundTruthBEVPipeline, build_planner_and_pipeline
from dataset.target_bev import rasterize_goal_target
from interfaces import BEVConfig, GoalPose
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import (
    KIND_BERM,
    KIND_ROCK,
    KIND_VEHICLE,
    KIND_WALL,
    MAX_SEMANTIC_HEIGHT_M,
    ParkingEnvironment,
    RectangleObstacle,
    SimulatedCamera,
    SimulatedLiDAR,
    normalized_height,
    semantic_height,
)
from interfaces import CameraIntrinsics
from dataset.pipeline import SensorBEVPipeline


def _config() -> BEVConfig:
    return BEVConfig(resolution=0.25, extent=(20.0, 20.0, 20.0, 20.0))


class TestSemanticHeight(unittest.TestCase):
    def test_kind_heights_are_ordered_and_normalized(self):
        self.assertGreater(semantic_height(KIND_WALL), semantic_height(KIND_BERM))
        self.assertGreater(semantic_height(KIND_BERM), semantic_height(KIND_ROCK))
        self.assertAlmostEqual(
            normalized_height(KIND_WALL), semantic_height(KIND_WALL) / MAX_SEMANTIC_HEIGHT_M
        )
        self.assertLessEqual(normalized_height(KIND_WALL), 1.0)

    def test_unknown_kind_falls_back_to_default(self):
        self.assertGreater(semantic_height("unknown-kind"), 0.0)

    def test_lidar_height_channel_carries_obstacle_kind(self):
        """回归：固定安装高度的平面 LiDAR 曾使 height 与 occupancy 完全重合。"""
        env = ParkingEnvironment(
            world_size=40.0,
            obstacles=[
                RectangleObstacle(x_min=4.0, x_max=6.0, y_min=-1.0, y_max=1.0, kind=KIND_WALL),
                RectangleObstacle(x_min=-6.0, x_max=-4.0, y_min=-1.0, y_max=1.0, kind=KIND_ROCK),
            ],
        )
        lidar = SimulatedLiDAR(env, beams=720, max_range=20.0)
        bev = LiDAR2BEV(config=_config()).to_bev(lidar.capture(0.0, 0.0, 0.0), 0.0, 0.0, 0.0)
        channels = list(bev.channels)
        height = np.asarray(bev.data)[channels.index("height")]
        occupancy = np.asarray(bev.data)[channels.index("occupancy")]
        occupied = height[occupancy > 0]
        self.assertGreater(occupied.size, 0)
        # 归一化语义高度上限为 1.0：通道不得出现米制长度（曾有 3.0/0.8 这类值）。
        self.assertLessEqual(float(occupied.max()), 1.0 + 1e-6)
        occupied_values = {round(float(value), 2) for value in occupied}
        self.assertIn(round(normalized_height(KIND_WALL), 2), occupied_values)
        self.assertIn(round(normalized_height(KIND_ROCK), 2), occupied_values)
        self.assertGreater(len(occupied_values), 1)


class TestGeometryTarget(unittest.TestCase):
    def test_goal_rect_is_rasterized_regardless_of_direction(self):
        config = _config()
        for goal in (GoalPose(8.0, 0.0, 0.0), GoalPose(0.0, 8.0, 0.0), GoalPose(-8.0, 0.0, 0.0)):
            with self.subTest(goal=goal):
                grid = rasterize_goal_target(goal, 6.0, 3.0, 0.0, 0.0, 0.0, config)
                # 6×3m / 0.25m² ≈ 288 格，允许边界取整误差。
                self.assertGreater(int(grid.sum()), 200)

    def test_goal_outside_extent_gives_empty_channel(self):
        config = _config()
        grid = rasterize_goal_target(GoalPose(35.0, 0.0, 0.0), 6.0, 3.0, 0.0, 0.0, 0.0, config)
        self.assertEqual(float(grid.sum()), 0.0)

    def test_sensor_pipeline_target_override_beats_camera(self):
        """侧方目标：相机链路为空，几何 target 仍应有值。"""
        env = ParkingEnvironment(world_size=40.0, obstacles=[], parking_spots=[])
        intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        )
        config = _config()

        def pipeline(target_override: bool) -> SensorBEVPipeline:
            return SensorBEVPipeline(
                lidar_sensor=SimulatedLiDAR(env, beams=360, max_range=20.0),
                camera_sensor=SimulatedCamera(
                    env, intrinsics, parking_area=(6.0, 3.0), view_yaws_deg=(0.0,)
                ),
                lidar2bev=LiDAR2BEV(config=config),
                camera2bev=Camera2BEV(config=config),
                bev_fusion=BEVFusion(vehicle_length=6.0, vehicle_width=3.0),
                target_override=target_override,
            )

        goal = GoalPose(0.0, 8.0, 0.0)  # 正左方，单前视看不到
        for override, expect_positive in ((False, False), (True, True)):
            with self.subTest(override=override):
                pipe = pipeline(override)
                pipe.set_target_goals([goal])
                bev = pipe.capture_bev(0.0, 0.0, 0.0)
                channels = list(bev.channels)
                target = np.asarray(bev.data)[channels.index("target")]
                if expect_positive:
                    self.assertGreater(int(target.sum()), 0)
                else:
                    self.assertEqual(int(target.sum()), 0)

    def test_geometry_override_leaves_other_channels_untouched(self):
        """改造只影响 target：同一次采集下其余通道逐格相同。"""
        env = ParkingEnvironment(world_size=40.0, obstacles=[], parking_spots=[])
        intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        )
        config = _config()

        def capture(target_override: bool) -> dict[str, np.ndarray]:
            pipe = SensorBEVPipeline(
                lidar_sensor=SimulatedLiDAR(env, beams=360, max_range=20.0, seed=3),
                camera_sensor=SimulatedCamera(
                    env, intrinsics, parking_area=(6.0, 3.0), seed=5
                ),
                lidar2bev=LiDAR2BEV(config=config),
                camera2bev=Camera2BEV(config=config),
                bev_fusion=BEVFusion(vehicle_length=6.0, vehicle_width=3.0),
                target_override=target_override,
            )
            pipe.set_target_goals([GoalPose(0.0, 8.0, 0.0)])  # 正左方：单前视看不到
            bev = pipe.capture_bev(0.0, 0.0, 0.0)
            channels = list(bev.channels)
            return {
                name: np.asarray(bev.data)[channels.index(name)] for name in channels
            }

        plain = capture(False)
        overridden = capture(True)
        for name in ("occupancy", "height", "density", "vehicle"):
            with self.subTest(channel=name):
                self.assertTrue(np.array_equal(plain[name], overridden[name]))
        self.assertFalse(np.array_equal(plain["target"], overridden["target"]))


class TestGroundTruthPipelineHeight(unittest.TestCase):
    def test_height_channel_reflects_obstacle_kind(self):
        env = ParkingEnvironment(
            world_size=40.0,
            obstacles=[
                RectangleObstacle(x_min=4.0, x_max=6.0, y_min=2.0, y_max=4.0, kind=KIND_BERM),
                RectangleObstacle(x_min=4.0, x_max=6.0, y_min=-4.0, y_max=-2.0, kind=KIND_VEHICLE),
            ],
        )
        pipeline = GroundTruthBEVPipeline(
            env, _config(), vehicle_length=6.0, vehicle_width=3.0
        )
        pipeline.set_target_goals([GoalPose(10.0, 0.0, 0.0)])
        bev = pipeline.capture_bev(0.0, 0.0, 0.0)
        channels = list(bev.channels)
        height = np.asarray(bev.data)[channels.index("height")]
        occupied = height[height > 0]
        self.assertGreater(occupied.size, 0)
        self.assertLessEqual(float(occupied.max()), 1.0 + 1e-6)
        values = {round(float(value), 2) for value in occupied}
        self.assertIn(round(normalized_height(KIND_BERM), 2), values)
        self.assertIn(round(normalized_height(KIND_VEHICLE), 2), values)


class _TaskStub:
    """`build_planner_and_pipeline` 需要的最小 Task 替身。"""

    def __init__(self) -> None:
        from sim import MINING_DRILL_RIG, NoiseLevel, SceneBundle  # noqa: F401

        self.scene = type("Scene", (), {})()
        self.scene.env = ParkingEnvironment(world_size=40.0, obstacles=[])
        self.scene.bev_config = _config()
        self.difficulty = type("Difficulty", (), {"noise_level": "clean"})()
        self.seed = 11


if __name__ == "__main__":
    unittest.main()
