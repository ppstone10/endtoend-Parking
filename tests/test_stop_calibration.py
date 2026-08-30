"""变长停止阈值校准测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.stop_calibration import (
    calibrate_stop_threshold,
    write_deployment_checkpoint,
)


class TestStopCalibration(unittest.TestCase):
    def test_selects_threshold_with_lowest_length_mae(self):
        probabilities = np.asarray(
            [[0.1, 0.8, 0.9, 0.9], [0.1, 0.4, 0.7, 0.9]], dtype=np.float64
        )
        logits = np.log(probabilities / (1.0 - probabilities))
        masks = np.asarray(
            [[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.float32
        )

        result = calibrate_stop_threshold(
            logits, masks, thresholds=[0.3, 0.5, 0.75]
        )

        self.assertEqual(result["selected_threshold"], 0.5)
        self.assertEqual(result["selected"]["length_mae_points"], 0.0)
        self.assertEqual(result["selected"]["stop_found_rate"], 1.0)

    def test_rejects_invalid_threshold_grid(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            calibrate_stop_threshold(
                np.zeros((1, 2)), np.ones((1, 2)), thresholds=[0.0, 0.5]
            )

    def test_writes_deployment_checkpoint_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp, "best.pt")
            destination = Path(temp, "deployment.pt")
            torch.save(
                {
                    "schema_version": 1,
                    "model_config": {"max_horizon": 4},
                    "model_state": {},
                },
                source,
            )

            write_deployment_checkpoint(
                source,
                destination,
                threshold=0.25,
                calibration={"selected_threshold": 0.25},
            )

            original = torch.load(source, weights_only=False)
            deployed = torch.load(destination, weights_only=False)
            self.assertNotIn("stop_threshold", original["model_config"])
            self.assertEqual(deployed["model_config"]["stop_threshold"], 0.25)
            self.assertFalse(deployed["resumable"])
            self.assertEqual(deployed["stop_calibration"]["selected_threshold"], 0.25)


if __name__ == "__main__":
    unittest.main()
