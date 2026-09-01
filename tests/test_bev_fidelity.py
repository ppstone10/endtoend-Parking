"""BEV 保真度指标与评测脚本测试（验证阶梯 L1）。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from interfaces import BEVConfig, BEVTensor, GoalPose
from metrics.bev_fidelity import (
    aggregate_bev_fidelity,
    compute_bev_fidelity_metrics,
    rasterize_ground_truth_occupancy,
    rasterize_ground_truth_target,
    rasterize_lidar_truth_occupancy,
)
from scripts.evaluate_bev_fidelity import run_bev_fidelity
from sim import ParkingEnvironment, RectangleObstacle


def _bev_config() -> BEVConfig:
    return BEVConfig(resolution=1.0, extent=(4.0, 4.0, 4.0, 4.0))


def _channel_bev(occupancy: np.ndarray, target: np.ndarray | None = None) -> BEVTensor:
    config = _bev_config()
    h, w = config.shape
    channels = ["occupancy"]
    channels_list = [occupancy.astype(np.float32)]
    if target is not None:
        channels.append("target")
        channels_list.append(target.astype(np.float32))
    data = np.stack(channels_list, axis=0)
    return BEVTensor(data=data, resolution=config.resolution, extent=config.extent, channels=channels)


class TestGroundTruthRasterization(unittest.TestCase):
    def setUp(self):
        self.config = _bev_config()
        self.env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(x_min=2.0, x_max=4.0, y_min=-1.0, y_max=1.0)],
        )

    def test_occupancy_rasterizes_emits_points_obstacle(self):
        # 车辆在原点朝向 +x，障碍位于前方 2~4m 处。
        truth = rasterize_ground_truth_occupancy(self.env, 0.0, 0.0, 0.0, self.config)
        self.assertEqual(truth.shape, self.config.shape)
        self.assertGreater(truth.sum(), 0.0)

    def test_occupancy_ignores_non_emitting_obstacle(self):
        from sim import PolygonObstacle

        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[PolygonObstacle(
                vertices=[(2.0, -1.0), (4.0, -1.0), (4.0, 1.0), (2.0, 1.0)],
                emits_points=False,
            )],
        )
        truth = rasterize_ground_truth_occupancy(env, 0.0, 0.0, 0.0, self.config)
        self.assertEqual(truth.sum(), 0.0)

    def test_target_rasterizes_goal_rectangle(self):
        truth = rasterize_ground_truth_target(
            GoalPose(3.0, 0.0, 0.0), length=6.0, width=3.0, x=0.0, y=0.0, yaw=0.0, bev_config=self.config
        )
        self.assertGreater(truth.sum(), 0.0)
        # 目标矩形 6×3m @1m 分辨率应覆盖约 18 个栅格。
        self.assertAlmostEqual(truth.sum(), 18.0, delta=4.0)

    def test_lidar_truth_matches_occupancy_for_front_obstacle(self):
        lidar_truth = rasterize_lidar_truth_occupancy(
            self.env, 0.0, 0.0, 0.0, self.config, beams=1800
        )
        self.assertEqual(lidar_truth.shape, self.config.shape)
        self.assertGreater(lidar_truth.sum(), 0.0)


class TestFidelityMetrics(unittest.TestCase):
    def test_perfect_match(self):
        truth = np.zeros((8, 8), dtype=np.float32)
        truth[2:6, 2:6] = 1.0
        bev = _channel_bev(occupancy=truth, target=truth)
        metrics = compute_bev_fidelity_metrics(bev, truth, truth)
        self.assertAlmostEqual(metrics.occupancy_iou, 1.0)
        self.assertAlmostEqual(metrics.occupancy_precision, 1.0)
        self.assertAlmostEqual(metrics.occupancy_recall, 1.0)
        self.assertAlmostEqual(metrics.target_hit_rate, 1.0)

    def test_missing_occupancy_zero_recall(self):
        truth = np.zeros((8, 8), dtype=np.float32)
        truth[2:6, 2:6] = 1.0
        bev = _channel_bev(occupancy=np.zeros((8, 8)), target=truth)
        metrics = compute_bev_fidelity_metrics(bev, truth, truth)
        self.assertAlmostEqual(metrics.occupancy_iou, 0.0)
        self.assertAlmostEqual(metrics.occupancy_recall, 0.0)
        self.assertAlmostEqual(metrics.occupancy_precision, 1.0)

    def test_extra_prediction_drops_precision(self):
        truth = np.zeros((8, 8), dtype=np.float32)
        truth[2:6, 2:6] = 1.0
        pred = truth.copy()
        pred[6, 6] = 1.0
        bev = _channel_bev(occupancy=pred, target=truth)
        metrics = compute_bev_fidelity_metrics(bev, truth, truth)
        self.assertLess(metrics.occupancy_precision, 1.0)
        self.assertLess(metrics.occupancy_iou, 1.0)

    def test_requires_occupancy_and_target_channels(self):
        with self.assertRaises(ValueError):
            compute_bev_fidelity_metrics(
                _channel_bev(occupancy=np.zeros((8, 8))), np.zeros((8, 8)), np.zeros((8, 8))
            )


class TestAggregate(unittest.TestCase):
    def test_aggregate_averages(self):
        truth = np.zeros((8, 8), dtype=np.float32)
        truth[2:6, 2:6] = 1.0
        bev = _channel_bev(occupancy=truth, target=truth)
        m1 = compute_bev_fidelity_metrics(bev, truth, truth)
        m2 = compute_bev_fidelity_metrics(bev, np.zeros((8, 8)), truth)
        agg = aggregate_bev_fidelity([m1, m2])
        self.assertEqual(agg.samples, 2)
        self.assertAlmostEqual(agg.occupancy_iou, m1.occupancy_iou / 2.0)

    def test_aggregate_empty_raises(self):
        with self.assertRaises(ValueError):
            aggregate_bev_fidelity([])


class TestEvalScriptSmoke(unittest.TestCase):
    def test_run_bev_fidelity_writes_report_and_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = run_bev_fidelity(
                ["S1_parking_lot"],
                output_dir=root / "out",
                poses_per_scene=1,
                noise_levels=["clean", "high"],
                seed=0,
            )
            self.assertIn("clean", report["overall"])
            self.assertIn("high", report["overall"])
            self.assertGreater(report["overall"]["clean"]["occupancy_iou"], 0.0)
            self.assertTrue((root / "out/report.json").exists())
            self.assertTrue((root / "out/bev_fidelity_degradation.png").exists())
            self.assertTrue((root / "out/bev_fidelity_degradation.pdf").exists())


if __name__ == "__main__":
    unittest.main()