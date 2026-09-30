"""GT BEV 管道测试：几何真值五通道与控制/生产 BEV 的同构性。"""

import unittest

import numpy as np

from dataset import GroundTruthBEVPipeline, ground_truth_vehicle_outline
from interfaces import BEVConfig, GoalPose
from sim import ParkingEnvironment, RectangleObstacle


def _environment() -> ParkingEnvironment:
    return ParkingEnvironment(
        world_size=40.0,
        obstacles=[RectangleObstacle(x_min=-10.0, x_max=10.0, y_min=4.0, y_max=6.0)],
    )


class TestGroundTruthBEVPipeline(unittest.TestCase):
    def setUp(self):
        self.config = BEVConfig(resolution=0.25, extent=(20.0, 20.0, 20.0, 20.0))
        self.pipeline = GroundTruthBEVPipeline(
            _environment(),
            self.config,
            vehicle_length=6.0,
            vehicle_width=3.0,
        )
        self.goal = GoalPose(5.0, 0.0, 0.0)
        self.pipeline.set_target_goals([self.goal])

    def test_channels_and_shape_match_production(self):
        bev = self.pipeline.capture_bev(0.0, 0.0, 0.0)
        self.assertEqual(
            list(bev.channels),
            ["occupancy", "height", "density", "target", "vehicle"],
        )
        self.assertEqual(bev.data.shape, (5, *self.config.shape))
        self.assertEqual(bev.resolution, self.config.resolution)
        self.assertEqual(bev.extent, self.config.extent)

    def test_occupancy_marks_wall_ahead(self):
        bev = self.pipeline.capture_bev(0.0, 0.0, 0.0)
        occupancy = bev.data[0]
        # 障碍位于全局 y∈[4,6]，车头朝 +x：局部系中落在左侧（列 > 中心）。
        front, back, left, right = self.config.extent
        del front, back
        col_center = int(right / self.config.resolution)
        self.assertGreater(occupancy[:, col_center + 10 : col_center + 20].sum(), 0.0)
        self.assertEqual(occupancy[:, : col_center - 20].sum(), 0.0)

    def test_height_and_density_follow_occupancy(self):
        bev = self.pipeline.capture_bev(0.0, 0.0, 0.0)
        occupancy, height, density = bev.data[0], bev.data[1], bev.data[2]
        occupied = occupancy > 0.0
        self.assertTrue(np.all(height[occupied] > 0.0))
        self.assertTrue(np.all(height[~occupied] == 0.0))
        self.assertTrue(np.all(density[occupied] == 1.0))
        self.assertTrue(np.all(density[~occupied] == 0.0))

    def test_target_channel_marks_goal_rectangle(self):
        bev = self.pipeline.capture_bev(0.0, 0.0, 0.0)
        target = bev.data[3]
        self.assertGreater(target.sum(), 0.0)
        # 目标在 +x 方向 5m 处，矩形应落在中心行之前（行号更小）。
        rows, cols = np.nonzero(target)
        center_row = int(self.config.extent[0] / self.config.resolution)
        self.assertLess(float(rows.mean()), float(center_row))

    def test_vehicle_channel_is_self_footprint(self):
        bev = self.pipeline.capture_bev(3.0, 2.0, 0.5)
        vehicle = bev.data[4]
        expected_area = (6.0 / 0.25) * (3.0 / 0.25)
        self.assertAlmostEqual(float(vehicle.sum()), expected_area, delta=expected_area * 0.1)

    def test_no_targets_means_empty_target_channel(self):
        pipeline = GroundTruthBEVPipeline(
            _environment(),
            self.config,
            vehicle_length=6.0,
            vehicle_width=3.0,
        )
        bev = pipeline.capture_bev(0.0, 0.0, 0.0)
        self.assertEqual(float(bev.data[3].sum()), 0.0)

    def test_invalid_vehicle_size_rejected(self):
        with self.assertRaises(ValueError):
            GroundTruthBEVPipeline(
                _environment(), self.config, vehicle_length=0.0, vehicle_width=3.0
            )


class TestVehicleOutline(unittest.TestCase):
    def test_centered_on_vehicle_origin(self):
        config = BEVConfig(resolution=0.25, extent=(20.0, 20.0, 20.0, 20.0))
        outline = ground_truth_vehicle_outline(config, 4.0, 2.0)
        rows, cols = np.nonzero(outline[0])
        self.assertAlmostEqual(float(rows.mean()), 79.5, delta=1.0)
        self.assertAlmostEqual(float(cols.mean()), 79.5, delta=1.0)


if __name__ == "__main__":
    unittest.main()
