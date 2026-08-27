"""开环轨迹指标口径测试。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from metrics.open_loop import compute_open_loop_metrics


class TestOpenLoopMetrics(unittest.TestCase):
    def test_ade_fde_and_wrapped_yaw_error(self):
        targets = np.asarray(
            [[[0.0, 0.0, math.pi - 0.1], [1.0, 0.0, -math.pi + 0.1]]],
            dtype=np.float32,
        )
        predictions = np.asarray(
            [[[0.0, 0.0, -math.pi + 0.1], [2.0, 0.0, math.pi - 0.1]]],
            dtype=np.float32,
        )
        metrics = compute_open_loop_metrics(
            predictions, targets, np.asarray([[1.0, 1.0]], dtype=np.float32)
        )

        self.assertEqual(metrics.samples, 1)
        self.assertEqual(metrics.valid_points, 2)
        self.assertAlmostEqual(metrics.ade_m, 0.5, places=6)
        self.assertAlmostEqual(metrics.fde_m, 1.0, places=6)
        self.assertAlmostEqual(metrics.yaw_mae_rad, 0.2, places=5)

    def test_rejects_prediction_shorter_than_valid_target(self):
        predictions = np.zeros((1, 2, 3), dtype=np.float32)
        targets = np.zeros((1, 3, 3), dtype=np.float32)
        masks = np.ones((1, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "horizon"):
            compute_open_loop_metrics(predictions, targets, masks)

    def test_rejects_non_prefix_mask(self):
        points = np.zeros((1, 3, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "连续前缀"):
            compute_open_loop_metrics(
                points, points, np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
