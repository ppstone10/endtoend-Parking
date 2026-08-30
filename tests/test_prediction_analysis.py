"""预测误差分组与叠加图入口 smoke。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model import build_model
from metrics import analyze_prediction_errors
from scripts.analyze_predictions import run_prediction_analysis


class TestPredictionAnalysis(unittest.TestCase):
    def test_aggregates_valid_prefix_and_stop_lengths_by_task(self):
        predictions = np.zeros((2, 3, 3), dtype=np.float32)
        predictions[0, 1, 0] = 1.0
        predictions[1, 0, 0] = 3.0
        targets = np.zeros_like(predictions)
        masks = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.float32)
        stop_logits = np.asarray(
            [[-10.0, 10.0, -10.0], [10.0, -10.0, -10.0]], dtype=np.float32
        )
        metadata = [
            {
                "task_id": "a",
                "scene_name": "S1",
                "task_type": "T1",
                "difficulty": {"maneuver": "forward", "noise_level": "clean", "adjacent_occupancy": 0},
            },
            {
                "task_id": "b",
                "scene_name": "S2",
                "task_type": "T2",
                "difficulty": {"maneuver": "reverse", "noise_level": "high", "adjacent_occupancy": 1},
            },
        ]

        report, rows = analyze_prediction_errors(
            predictions, targets, masks, metadata, stop_logits=stop_logits
        )

        self.assertAlmostEqual(report["overall"]["ade_m"], 4.0 / 3.0)
        self.assertAlmostEqual(report["overall"]["fde_m"], 2.0)
        self.assertEqual(report["overall"]["predicted_length_mae_points"], 0.0)
        self.assertEqual(report["overall"]["stop_found_rate"], 1.0)
        self.assertEqual(report["groups"]["task_type"]["T1"]["samples"], 1)
        self.assertEqual(rows[0]["predicted_length"], 2)

    def test_writes_grouped_report_and_overlay_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = [
                {
                    "task_id": f"task-{index}",
                    "scene_name": "S1_parking_lot",
                    "task_type": "T1" if index == 0 else "T2",
                    "difficulty": {
                        "maneuver": "forward" if index == 0 else "reverse",
                        "noise_level": "clean",
                        "adjacent_occupancy": index,
                    },
                }
                for index in range(2)
            ]
            bev_meta = {
                "resolution": 1.0,
                "extent": [4.0, 4.0, 4.0, 4.0],
                "channels": ["occupancy", "height", "density", "target", "vehicle"],
                "shape": [5, 8, 8],
            }
            data_path = root / "val.npz"
            np.savez_compressed(
                data_path,
                schema_version=np.asarray(2, dtype=np.uint16),
                bev_meta=np.asarray(json.dumps(bev_meta), dtype=np.str_),
                task_meta=np.asarray([json.dumps(item) for item in metadata], dtype=np.str_),
                bevs=np.zeros((2, 5, 8, 8), dtype=np.float32),
                goals=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
                states=np.zeros((2, 5), dtype=np.float32),
                trajs=np.asarray(
                    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
                    dtype=np.float32,
                ),
                masks=np.ones((2, 2), dtype=np.float32),
                dt=np.asarray([0.1], dtype=np.float32),
            )
            model_config = {
                "bev_channels": 5,
                "max_horizon": 2,
                "dt": 0.1,
                "hidden_dim": 4,
            }
            model = build_model("net-v1", model_config)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "model_name": "net-v1",
                    "model_config": model_config,
                    "epoch": 0,
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )

            report = run_prediction_analysis(
                data_path,
                checkpoint,
                output_dir=root / "analysis",
                batch_size=1,
                overlay_count=2,
            )

            self.assertEqual(report["overall"]["samples"], 2)
            self.assertEqual(report["groups"]["task_type"]["T1"]["samples"], 1)
            self.assertIn("predicted_length_mae_points", report["overall"])
            for name in (
                "report.json",
                "grouped_metrics.png",
                "grouped_metrics.pdf",
                "worst_overall.png",
                "worst_overall.pdf",
                "worst_by_task.png",
                "worst_by_task.pdf",
            ):
                self.assertTrue((root / "analysis" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
