"""数据集生成测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import DatasetGenerator, SensorBEVPipeline, TaskGenerationError, TrainingSample
from interfaces import BEVConfig, BEVTensor, CameraIntrinsics, GoalPose, Trajectory, VehicleState
from planner import HybridAStarPlanner
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR
from sim.tasks import NoiseLevel, TaskSampler, TaskType


def _build_generator(seed: int = 0) -> DatasetGenerator:
    env = ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=-6.0, y_max=-2.0),
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=2.0, y_max=6.0),
        ],
    )
    planner = HybridAStarPlanner(env=env)
    intrinsics = CameraIntrinsics(
        fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
    )
    pipeline = SensorBEVPipeline(
        lidar_sensor=SimulatedLiDAR(env, beams=360, max_range=20.0),
        camera_sensor=SimulatedCamera(env, intrinsics),
        lidar2bev=LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
        camera2bev=Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
        bev_fusion=BEVFusion(),
    )
    return DatasetGenerator(env=env, planner=planner, sensor_pipeline=pipeline, seed=seed)


def _sample(task_meta: dict | None = None) -> TrainingSample:
    config = BEVConfig()
    bev = BEVTensor(
        data=np.zeros((5, *config.shape), dtype=np.float32),
        resolution=config.resolution,
        extent=config.extent,
        channels=["occupancy", "height", "density", "target", "vehicle"],
    )
    return TrainingSample(
        bev=bev,
        goal=GoalPose(4.0, 0.0, 0.0),
        state=VehicleState(0.0, 0.0, 0.0),
        expert_trajectory=Trajectory(
            points=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
            dt=0.1,
        ),
        task_meta=task_meta,
    )


class TestDatasetGenerator(unittest.TestCase):
    def test_pipeline_rejects_mismatched_spatial_configs(self):
        env = ParkingEnvironment(world_size=40.0)
        intrinsics = CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        )
        with self.assertRaises(ValueError):
            SensorBEVPipeline(
                lidar_sensor=SimulatedLiDAR(env),
                camera_sensor=SimulatedCamera(env, intrinsics),
                lidar2bev=LiDAR2BEV(config=BEVConfig()),
                camera2bev=Camera2BEV(
                    config=BEVConfig(resolution=0.5, extent=(20.0, 20.0, 20.0, 20.0))
                ),
                bev_fusion=BEVFusion(),
            )

    def test_generate_samples(self):
        generator = _build_generator(seed=1)
        samples = generator.generate(count=3)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertEqual(sample.bev.shape[0], 5)  # 融合 BEV 5 通道
            self.assertEqual(sample.bev.channels[-1], "vehicle")
            self.assertGreater(sample.expert_trajectory.horizon, 2)
            self.assertIsInstance(sample.goal, GoalPose)
            self.assertIsInstance(sample.state, VehicleState)

    def test_sample_poses_free(self):
        generator = _build_generator(seed=2)
        samples = generator.generate(count=2)
        env = generator.env
        for sample in samples:
            # 起始与目标车辆矩形中心应在自由空间。
            self.assertTrue(env.is_free(sample.state.x, sample.state.y))
            self.assertTrue(env.is_free(sample.goal.x, sample.goal.y))
            # 专家轨迹各点自由。
            for px, py in sample.expert_trajectory.points[:, :2]:
                self.assertTrue(env.is_free(float(px), float(py)))

    def test_schema_v2_roundtrip_exposes_bev_and_task_metadata(self):
        generator = _build_generator()
        samples = [_sample({"task_type": "T1_short"}), _sample()]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.npz"
            generator.save(samples, path)
            with np.load(path, allow_pickle=False) as raw:
                self.assertEqual(raw["bev_meta"].dtype.kind, "U")
                self.assertEqual(raw["task_meta"].dtype.kind, "U")
            data = generator.load(path)

        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(data["bev_meta"]["resolution"], 0.25)
        self.assertEqual(data["bev_meta"]["extent"], [20.0, 20.0, 20.0, 20.0])
        self.assertEqual(data["bev_meta"]["shape"], [5, 160, 160])
        self.assertEqual(data["task_meta"], [{"task_type": "T1_short"}, {}])

    def test_v1_archive_still_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.npz"
            np.savez_compressed(path, bevs=np.zeros((1, 5, 4, 4)), dt=np.array([0.1]))
            data = DatasetGenerator.load(path)

        self.assertEqual(data["schema_version"], 1)
        self.assertIsNone(data["bev_meta"])
        self.assertIsNone(data["task_meta"])
        self.assertEqual(data["bevs"].shape, (1, 5, 4, 4))

    def test_unknown_explicit_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "future.npz"
            np.savez_compressed(path, schema_version=np.asarray(3, dtype=np.uint16))
            with self.assertRaises(ValueError):
                DatasetGenerator.load(path)

    def test_schema_v2_rejects_metadata_not_aligned_with_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "misaligned.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(2, dtype=np.uint16),
                bevs=np.zeros((2, 5, 4, 4), dtype=np.float32),
                bev_meta=np.asarray(
                    '{"channels":["a","b","c","d","e"],"extent":[1,1,1,1],'
                    '"resolution":0.5,"shape":[5,4,4]}',
                    dtype=np.str_,
                ),
                task_meta=np.asarray(["{}"], dtype=np.str_),
            )
            with self.assertRaises(ValueError):
                DatasetGenerator.load(path)

    def test_mixed_bev_metadata_is_rejected(self):
        generator = _build_generator()
        first = _sample()
        second = _sample()
        second.bev.channels[-1] = "other"
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                generator.save([first, second], Path(temp_dir) / "invalid.npz")


class _TaskPipeline:
    def __init__(self, task):
        self.bev_config = task.scene.bev_config

    def capture_bev(self, x: float, y: float, yaw: float) -> BEVTensor:
        config = self.bev_config
        return BEVTensor(
            data=np.zeros((5, *config.shape), dtype=np.float32),
            resolution=config.resolution,
            extent=config.extent,
            channels=["occupancy", "height", "density", "target", "vehicle"],
        )


class _TaskPlanner:
    def __init__(self, rejected_x: float | None = None, reject_all: bool = False):
        self.rejected_x = rejected_x
        self.reject_all = reject_all

    def plan(self, start: VehicleState, goal: GoalPose) -> Trajectory:
        if self.reject_all or self.rejected_x == goal.x:
            raise RuntimeError("测试规划失败")
        return Trajectory(
            points=np.array(
                [[start.x, start.y, start.yaw], [goal.x, goal.y, goal.yaw]],
                dtype=np.float32,
            ),
            dt=0.1,
        )


class TestTaskDrivenDataset(unittest.TestCase):
    def test_single_goal_task_becomes_self_describing_sample(self):
        task = TaskSampler(seed=20260824).sample(
            "S1_parking_lot", TaskType.T1_NEAR, noise_level=NoiseLevel.LOW
        )
        generator = DatasetGenerator(
            component_factory=lambda current: (_TaskPlanner(), _TaskPipeline(current))
        )
        sample = generator.generate([task])[0]

        self.assertEqual(sample.state, task.start)
        self.assertEqual(sample.goal, task.goal.as_goal_pose())
        self.assertEqual(sample.task_meta["task_id"], task.task_id)
        self.assertEqual(sample.task_meta["dataset"]["goal_policy"], "task_goal")
        self.assertEqual(sample.task_meta["noise_profile"]["level"], "low")

    def test_t4_uses_first_plannable_candidate_and_records_policy(self):
        task = TaskSampler(seed=77).sample("S1_parking_lot", TaskType.T4_MULTI_SPOT)
        rejected_x = task.candidate_goals[0].x
        generator = DatasetGenerator(
            component_factory=lambda current: (
                _TaskPlanner(rejected_x=rejected_x),
                _TaskPipeline(current),
            )
        )
        sample = generator.generate([task])[0]

        self.assertNotEqual(sample.goal.x, rejected_x)
        self.assertEqual(
            sample.task_meta["dataset"]["goal_policy"],
            "first_plannable_candidate",
        )
        self.assertEqual(
            sample.task_meta["dataset"]["selected_goal"]["x"], sample.goal.x
        )

    def test_task_failure_identifies_task(self):
        task = TaskSampler(seed=8).sample("S1_parking_lot", TaskType.T1_NEAR)
        generator = DatasetGenerator(
            component_factory=lambda current: (
                _TaskPlanner(reject_all=True),
                _TaskPipeline(current),
            )
        )
        with self.assertRaisesRegex(TaskGenerationError, task.task_id):
            generator.generate([task])


if __name__ == "__main__":
    unittest.main()
