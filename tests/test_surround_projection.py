"""CAM-PROJ: 独立几何真值约束相机方向与裁剪，而非只检查非空。"""
import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile

from interfaces import BEVConfig, BEVTensor, CameraFrame, CameraIntrinsics, CameraView, GoalPose, VehicleState, Trajectory
from interfaces.bev import CAMERA_PROJECTION_CURRENT
from dataset.target_bev import rasterize_goal_target
from sensor2bev import Camera2BEV
from sim import ParkingEnvironment, SimulatedCamera
from model import build_model
from training.data import validate_model_dataset


class TestSurroundGeometry(unittest.TestCase):
    def setUp(self):
        self.config = BEVConfig()
        self.intrinsics = CameraIntrinsics(400, 400, 320, 240, 640, 480)

    def capture(self, goal, pose=(0., 0., 0.), views=(0., 90., -90., 180.)):
        env = ParkingEnvironment(world_size=100, obstacles=[], parking_spots=[goal])
        camera = SimulatedCamera(env, self.intrinsics, view_yaws_deg=views)
        bev = Camera2BEV(config=self.config).to_bev(camera.capture(*pose), *pose)
        truth = rasterize_goal_target(goal, 6., 3., *pose, self.config)
        return bev, truth

    def test_four_directions_match_independent_truth(self):
        for goal in [GoalPose(8, 0, 0), GoalPose(-8, 0, 0),
                     GoalPose(0, 8, np.pi/2), GoalPose(0, -8, np.pi/2)]:
            with self.subTest(goal=goal):
                bev, truth = self.capture(goal)
                np.testing.assert_array_equal(bev.data[0], truth)

    def test_left_and_right_are_distinct(self):
        left, _ = self.capture(GoalPose(0, 8, np.pi/2))
        right, _ = self.capture(GoalPose(0, -8, np.pi/2))
        self.assertFalse(np.array_equal(left.data, right.data))

    def test_clipped_intersections_stay_at_positive_near_plane(self):
        camera = SimulatedCamera(ParkingEnvironment(world_size=100, obstacles=[]), self.intrinsics)
        polygon = [np.array([-3., 1.5]), np.array([3., 1.5]),
                   np.array([3., -1.5]), np.array([-3., -1.5])]
        points = camera._clip_to_camera_front(polygon)
        depths = np.cos(camera.pitch)*np.array(points)[:, 0]+camera.height*np.sin(camera.pitch)
        self.assertTrue(np.all(depths >= .001-1e-12))
        self.assertTrue(np.isfinite([camera.model.project(*p) for p in points]).all())

    def test_nonzero_pose_and_general_view_yaw_on_asymmetric_grid(self):
        self.config = BEVConfig(.25, (16., 12., 14., 10.))
        pose = (5., -3., .73)
        angle = np.deg2rad(37.)
        local = np.array([8*np.cos(angle), 8*np.sin(angle)])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        goal = GoalPose(pose[0]+c*local[0]-s*local[1],
                        pose[1]+s*local[0]+c*local[1], pose[2]+angle)
        bev, truth = self.capture(goal, pose, (37.,))
        mask = bev.data[0]>0
        self.assertGreater((mask&(truth>0)).sum()/(mask|(truth>0)).sum(), .94)

    def test_views_have_calibration_and_preview_is_not_projection_input(self):
        env = ParkingEnvironment(world_size=100, obstacles=[], parking_spots=[GoalPose(-8, 0, 0)])
        camera = SimulatedCamera(env, self.intrinsics)
        frame = camera.capture(0., 0., 0.)
        self.assertEqual(len(frame.views), 4)
        np.testing.assert_allclose([v.yaw for v in frame.views], np.deg2rad([0, 90, -90, 180]))
        frame.image[:] = 0
        bev = Camera2BEV().to_bev(frame, 0, 0, 0)
        np.testing.assert_array_equal(bev.data[0], rasterize_goal_target(GoalPose(-8, 0, 0), 6, 3, 0, 0, 0, self.config))
        self.assertEqual(bev.to_metadata()['camera_projection'], CAMERA_PROJECTION_CURRENT)

    def test_per_view_height_pitch_override_converter_defaults(self):
        env = ParkingEnvironment(world_size=100, obstacles=[], parking_spots=[GoalPose(8, 0, 0)])
        camera = SimulatedCamera(env, self.intrinsics, height=2., pitch=np.deg2rad(40.), view_yaws_deg=(0.,))
        bev = Camera2BEV().to_bev(camera.capture(0., 0., 0.), 0, 0, 0)
        truth = rasterize_goal_target(env.parking_spots[0], 6, 3, 0, 0, 0, self.config)>0
        mask = bev.data[0]>0
        self.assertGreater((mask&truth).sum()/(mask|truth).sum(), .94)

    def test_empty_outside_overlap_and_legacy_single_frame(self):
        empty, _ = self.capture(GoalPose(80, 0, 0))
        self.assertEqual(empty.data.sum(), 0)
        one, _ = self.capture(GoalPose(8, 0, 0), views=(0.,))
        overlap, _ = self.capture(GoalPose(8, 0, 0), views=(0., 0.))
        np.testing.assert_array_equal(one.data, overlap.data)
        env = ParkingEnvironment(world_size=100, obstacles=[], parking_spots=[])
        camera = SimulatedCamera(env, self.intrinsics)
        frame = camera.capture(0, 0, 0)
        self.assertFalse(any(v.image.any() for v in frame.views))
        old = CameraFrame(frame.image, self.intrinsics)
        self.assertNotIn('camera_projection', Camera2BEV().to_bev(old, 0, 0, 0).to_metadata())

    def test_noise_is_reproducible_per_view(self):
        env = ParkingEnvironment(world_size=100, obstacles=[], parking_spots=[GoalPose(0, 8, 0)])
        for level in ('low', 'high'):
            a = SimulatedCamera(env, self.intrinsics, noise=level, seed=31)
            b = SimulatedCamera(env, self.intrinsics, noise=level, seed=31)
            for _ in range(2):
                fa, fb = a.capture(0, 0, 0), b.capture(0, 0, 0)
                for va, vb in zip(fa.views, fb.views, strict=True):
                    np.testing.assert_array_equal(va.image, vb.image)
                ba, bb = Camera2BEV().to_bev(fa, 0, 0, 0), Camera2BEV().to_bev(fb, 0, 0, 0)
                np.testing.assert_array_equal(ba.data, bb.data)
                self.assertTrue(np.isin(ba.data, [0, 1]).all())

    def test_invalid_image_and_calibration_rejected(self):
        image = np.zeros((480, 640, 1), np.uint8)
        for kwargs in ({'height': 0}, {'yaw': np.nan}, {'pitch': np.inf}):
            with self.assertRaises(ValueError):
                CameraView(image, self.intrinsics, **kwargs)
        with self.assertRaises(ValueError):
            CameraView(np.zeros((10, 10, 1)), self.intrinsics)
        with self.assertRaises(ValueError):
            CameraView(image, CameraIntrinsics(0, 400, 320, 240, 640, 480))


class TestProjectionContract(unittest.TestCase):
    def test_all_models_reject_cross_version_inference_and_training(self):
        torch.set_num_threads(1)
        for name in ('net-v0', 'net-v1', 'net-v2'):
            base = {'horizon': 3, 'hidden_dim': 8} if name=='net-v0' else {'max_horizon': 3, 'hidden_dim': 8}
            if name=='net-v2':
                base.update(base_channels=4, attention_heads=2)
            old = build_model(name, base)
            new = build_model(name, {**base, 'camera_projection': CAMERA_PROJECTION_CURRENT})
            current = BEVTensor(np.zeros((5, 16, 16)), .5, (4, 4, 4, 4), ['a', 'b', 'c', 'target', 'vehicle'], CAMERA_PROJECTION_CURRENT)
            legacy = BEVTensor(current.data, .5, (4, 4, 4, 4), current.channels)
            state, goal = VehicleState(0, 0, 0, 0, 0), GoalPose(2, 0, 0)
            for model, bev in ((old, current), (new, legacy)):
                with self.subTest(model=name), self.assertRaisesRegex(ValueError, 'camera_projection'):
                    model.predict(bev, goal, state)
            self.assertGreater(len(new.predict(current, goal, state).points), 0)
            data = {'bevs': current.data[None], 'dt': [new.dt], 'bev_meta': current.to_metadata()}
            validate_model_dataset(new, data)
            with self.assertRaisesRegex(ValueError, 'camera_projection'):
                validate_model_dataset(old, data)
            data['bev_meta'] = legacy.to_metadata()
            with self.assertRaisesRegex(ValueError, 'camera_projection'):
                validate_model_dataset(new, data)

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            build_model('net-v1', {'camera_projection': 'unverified'})
        model = build_model('net-v1', {'target_channel': 'geometry'})
        with self.assertRaisesRegex(ValueError, '未知输入camera_projection'):
            validate_model_dataset(model, {'bevs': np.zeros((1, 5, 16, 16)), 'dt': [.1],
                                           'bev_meta': {'camera_projection': 'unverified'}})

    def test_geometry_model_cannot_hide_new_image_input_mismatch(self):
        model = build_model('net-v1', {'max_horizon': 3, 'hidden_dim': 8, 'target_channel': 'geometry'})
        bev = BEVTensor(np.zeros((5, 16, 16)), .5, (4, 4, 4, 4),
                        ['a', 'b', 'c', 'target', 'vehicle'], CAMERA_PROJECTION_CURRENT)
        with self.assertRaisesRegex(ValueError, 'target_channel'):
            model.predict(bev, GoalPose(1, 0, 0), VehicleState(0, 0, 0))
        with self.assertRaisesRegex(ValueError, 'target_channel'):
            validate_model_dataset(model, {'bevs': bev.data[None], 'dt': [.1], 'bev_meta': bev.to_metadata()})

    def test_training_pair_cannot_mix_projection_versions(self):
        from training.runner import _validate_dataset_pair
        base = {'bevs': np.zeros((1, 5, 16, 16)), 'dt': [.1]}
        with self.assertRaisesRegex(ValueError, 'camera_projection'):
            _validate_dataset_pair(base, {**base, 'bev_meta': {'camera_projection': CAMERA_PROJECTION_CURRENT}})

    def test_dataset_rejects_mixed_projection_versions(self):
        from dataset import DatasetGenerator
        from dataset.generator import TrainingSample
        bevs = [BEVTensor(np.zeros((1, 2, 2)), 1., (1, 1, 1, 1), ['target'], version)
                for version in (None, CAMERA_PROJECTION_CURRENT)]
        samples = [TrainingSample(bev, GoalPose(0, 0, 0), VehicleState(0, 0, 0),
                                  Trajectory(np.zeros((2, 3)), .1)) for bev in bevs]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, '元数据'):
            DatasetGenerator(None).save(samples, Path(tmp)/'mixed.npz')

    def test_l1_sampling_does_not_force_target_in_front(self):
        from scripts.evaluate_bev_fidelity import _sample_vehicle_pose
        from sim import MINING_DRILL_RIG
        from types import SimpleNamespace
        # 空场景隔离航向采样，避免狭窄spawn区碰撞拒绝引入条件分布偏差。
        scene = SimpleNamespace(env=ParkingEnvironment(world_size=100, obstacles=[]),
                                spawn_zones=[(-1, 1, -1, 1)], name='empty')
        goal = GoalPose(8, 0, 0)
        rng = np.random.default_rng(4)
        poses = [_sample_vehicle_pose(scene, MINING_DRILL_RIG, goal, rng) for _ in range(4)]
        self.assertTrue(any(abs(np.arctan2(np.sin(np.arctan2(goal.y-y, goal.x-x)-yaw),
                                          np.cos(np.arctan2(goal.y-y, goal.x-x)-yaw)))>np.pi/2
                            for x, y, yaw in poses))


if __name__ == '__main__':
    unittest.main()
