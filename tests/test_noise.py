"""传感器噪声测试：三档配置、LiDAR/Camera 行为与 seed 复现。"""

import json
import unittest

import numpy as np

from interfaces import CameraIntrinsics, GoalPose
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR
from sim.noise import (
    CAMERA_NOISE_CLEAN,
    LIDAR_NOISE_CLEAN,
    CameraNoiseConfig,
    LiDARNoiseConfig,
    NoiseLevel,
    NoiseProfile,
    get_noise_profile,
)


def _lidar_env() -> ParkingEnvironment:
    return ParkingEnvironment(
        world_size=50.0,
        obstacles=[RectangleObstacle(8.0, 9.0, -20.0, 20.0)],
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=400.0, fy=400.0, cx=320.0, cy=240.0,
        image_width=640, image_height=480,
    )


def _camera_env(with_goal: bool = True) -> ParkingEnvironment:
    goals = [GoalPose(5.0, 0.0, 0.0)] if with_goal else []
    return ParkingEnvironment(world_size=30.0, parking_spots=goals)


class TestNoiseProfiles(unittest.TestCase):
    def test_builtin_profiles_are_monotonic_and_serializable(self):
        clean = get_noise_profile("clean")
        low = get_noise_profile(NoiseLevel.LOW)
        high = get_noise_profile("high")
        for field in ("range_std", "dropout_rate", "range_jitter_std"):
            self.assertLessEqual(getattr(clean.lidar, field), getattr(low.lidar, field))
            self.assertLessEqual(getattr(low.lidar, field), getattr(high.lidar, field))
        for field in ("pixel_std", "false_negative_rate", "false_positive_rate"):
            self.assertLessEqual(getattr(clean.camera, field), getattr(low.camera, field))
            self.assertLessEqual(getattr(low.camera, field), getattr(high.camera, field))
        json.dumps(high.to_metadata(), sort_keys=True)

    def test_invalid_values_and_unknown_level_are_rejected(self):
        with self.assertRaises(ValueError):
            LiDARNoiseConfig(dropout_rate=1.1)
        with self.assertRaises(ValueError):
            CameraNoiseConfig(pixel_std=-1.0)
        with self.assertRaises(ValueError):
            get_noise_profile("extreme")

    def test_task_and_sensor_layers_share_noise_level_enum(self):
        from sim.tasks import NoiseLevel as TaskNoiseLevel

        self.assertIs(TaskNoiseLevel, NoiseLevel)


class TestLiDARNoise(unittest.TestCase):
    def test_clean_default_is_backward_compatible(self):
        env = _lidar_env()
        default = SimulatedLiDAR(env, beams=180, max_range=20.0).capture(0.0, 0.0, 0.0)
        explicit = SimulatedLiDAR(
            env, beams=180, max_range=20.0, noise="clean", seed=999
        ).capture(0.0, 0.0, 0.0)
        np.testing.assert_array_equal(default.points, explicit.points)

    def test_distance_noise_perturbs_points_along_rays(self):
        profile = NoiseProfile(
            level=NoiseLevel.HIGH,
            lidar=LiDARNoiseConfig(range_std=0.5),
            camera=CAMERA_NOISE_CLEAN,
        )
        env = _lidar_env()
        clean = SimulatedLiDAR(env, beams=180, max_range=20.0).capture(0.0, 0.0, 0.0)
        noisy = SimulatedLiDAR(
            env, beams=180, max_range=20.0, noise=profile, seed=7
        ).capture(0.0, 0.0, 0.0)
        self.assertEqual(noisy.points.shape, clean.points.shape)
        self.assertGreater(np.abs(noisy.points[:, :2] - clean.points[:, :2]).sum(), 0.0)

    def test_dropout_can_return_variable_or_empty_point_cloud(self):
        high = SimulatedLiDAR(
            _lidar_env(), beams=1000, max_range=20.0, noise="high", seed=2
        ).capture(0.0, 0.0, 0.0)
        self.assertLess(high.count, 1000)
        drop_all = NoiseProfile(
            level=NoiseLevel.HIGH,
            lidar=LiDARNoiseConfig(dropout_rate=1.0),
            camera=CAMERA_NOISE_CLEAN,
        )
        empty = SimulatedLiDAR(
            _lidar_env(), beams=32, noise=drop_all, seed=3
        ).capture(0.0, 0.0, 0.0)
        self.assertEqual(empty.points.shape, (0, 4))

    def test_range_jitter_changes_frame_effective_range(self):
        profile = NoiseProfile(
            level=NoiseLevel.HIGH,
            lidar=LiDARNoiseConfig(range_jitter_std=1.0),
            camera=CAMERA_NOISE_CLEAN,
        )
        frame = SimulatedLiDAR(
            ParkingEnvironment(world_size=100.0),
            beams=90,
            max_range=20.0,
            noise=profile,
            seed=5,
        ).capture(0.0, 0.0, 0.0)
        radii = np.hypot(frame.points[:, 0], frame.points[:, 1])
        self.assertAlmostEqual(float(radii.std()), 0.0, places=5)
        self.assertNotAlmostEqual(float(radii.mean()), 20.0, places=3)


class TestCameraNoise(unittest.TestCase):
    def test_clean_default_is_backward_compatible(self):
        env = _camera_env()
        default = SimulatedCamera(env, _intrinsics()).capture(0.0, 0.0, 0.0)
        explicit = SimulatedCamera(
            env, _intrinsics(), noise=NoiseLevel.CLEAN, seed=999
        ).capture(0.0, 0.0, 0.0)
        np.testing.assert_array_equal(default.image, explicit.image)

    def test_false_negative_removes_real_target(self):
        profile = NoiseProfile(
            level=NoiseLevel.HIGH,
            lidar=LIDAR_NOISE_CLEAN,
            camera=CameraNoiseConfig(false_negative_rate=1.0),
        )
        frame = SimulatedCamera(
            _camera_env(), _intrinsics(), noise=profile, seed=1
        ).capture(0.0, 0.0, 0.0)
        self.assertEqual(int(frame.image.max()), 0)

    def test_false_positive_adds_target_without_ground_truth(self):
        profile = NoiseProfile(
            level=NoiseLevel.HIGH,
            lidar=LIDAR_NOISE_CLEAN,
            camera=CameraNoiseConfig(false_positive_rate=1.0),
        )
        frame = SimulatedCamera(
            _camera_env(with_goal=False), _intrinsics(), noise=profile, seed=1
        ).capture(0.0, 0.0, 0.0)
        self.assertGreater(int(frame.image.max()), 0)

    def test_pixel_noise_preserves_shape_and_uint8(self):
        frame = SimulatedCamera(
            _camera_env(), _intrinsics(), noise="high", seed=11
        ).capture(0.0, 0.0, 0.0)
        self.assertEqual(frame.image.shape, (480, 640, 1))
        self.assertEqual(frame.image.dtype, np.uint8)


class TestNoiseReproducibility(unittest.TestCase):
    def test_same_seed_reproduces_lidar_and_camera_sequences(self):
        lidar_a = SimulatedLiDAR(_lidar_env(), beams=120, noise="high", seed=42)
        lidar_b = SimulatedLiDAR(_lidar_env(), beams=120, noise="high", seed=42)
        camera_a = SimulatedCamera(_camera_env(), _intrinsics(), noise="high", seed=42)
        camera_b = SimulatedCamera(_camera_env(), _intrinsics(), noise="high", seed=42)
        for _ in range(2):
            np.testing.assert_array_equal(
                lidar_a.capture(0.0, 0.0, 0.0).points,
                lidar_b.capture(0.0, 0.0, 0.0).points,
            )
            np.testing.assert_array_equal(
                camera_a.capture(0.0, 0.0, 0.0).image,
                camera_b.capture(0.0, 0.0, 0.0).image,
            )

    def test_noise_does_not_consume_global_numpy_rng(self):
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        SimulatedLiDAR(_lidar_env(), beams=32, noise="high", seed=1).capture(
            0.0, 0.0, 0.0
        )
        actual = np.random.random(4)
        np.testing.assert_array_equal(actual, expected)

    def test_different_seed_changes_non_clean_observation(self):
        first = SimulatedCamera(
            _camera_env(), _intrinsics(), noise="high", seed=1
        ).capture(0.0, 0.0, 0.0)
        second = SimulatedCamera(
            _camera_env(), _intrinsics(), noise="high", seed=2
        ).capture(0.0, 0.0, 0.0)
        self.assertFalse(np.array_equal(first.image, second.image))


if __name__ == "__main__":
    unittest.main()
