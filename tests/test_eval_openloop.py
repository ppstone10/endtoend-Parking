"""多 checkpoint 开环评估入口 smoke。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model import build_model
from scripts.eval_openloop import run_evaluation


class TestEvalOpenLoop(unittest.TestCase):
    def test_checkpoint_evaluation_writes_report_and_comparison_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_path = root / "val.npz"
            np.savez_compressed(
                data_path,
                bevs=np.zeros((1, 5, 8, 8), dtype=np.float32),
                goals=np.zeros((1, 3), dtype=np.float32),
                states=np.zeros((1, 5), dtype=np.float32),
                trajs=np.zeros((1, 2, 3), dtype=np.float32),
                masks=np.ones((1, 2), dtype=np.float32),
                dt=np.asarray([0.1], dtype=np.float32),
            )
            model_config = {
                "bev_channels": 5,
                "horizon": 2,
                "dt": 0.1,
                "hidden_dim": 4,
            }
            model = build_model("net-v0", model_config)
            model(
                torch.zeros(1, 5, 8, 8),
                torch.zeros(1, 3),
                torch.zeros(1, 2),
            )
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "model_name": "net-v0",
                    "model_config": model_config,
                    "epoch": 0,
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )

            report = run_evaluation(
                data_path,
                [f"baseline={checkpoint}"],
                output_dir=root / "evaluation",
                batch_size=1,
            )

            self.assertEqual(report["models"]["baseline"]["model_name"], "net-v0")
            self.assertTrue((root / "evaluation/report.json").exists())
            self.assertTrue((root / "evaluation/openloop_comparison.png").exists())
            self.assertTrue((root / "evaluation/openloop_comparison.pdf").exists())


if __name__ == "__main__":
    unittest.main()
