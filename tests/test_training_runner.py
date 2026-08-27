"""配置化训练的秒级端到端 smoke。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.runner import run_training_from_yaml


def _write_dataset(path: Path, samples: int) -> None:
    np.savez_compressed(
        path,
        bevs=np.zeros((samples, 5, 8, 8), dtype=np.float32),
        goals=np.zeros((samples, 3), dtype=np.float32),
        states=np.zeros((samples, 5), dtype=np.float32),
        trajs=np.zeros((samples, 2, 3), dtype=np.float32),
        masks=np.ones((samples, 2), dtype=np.float32),
        dt=np.asarray([0.1], dtype=np.float32),
    )


class TestTrainingRunner(unittest.TestCase):
    def test_one_epoch_produces_checkpoint_curve_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root / "train.npz", 2)
            _write_dataset(root / "val.npz", 1)
            config = root / "run.yaml"
            config.write_text(
                "model:\n"
                "  name: net-v0\n"
                "  config: {bev_channels: 5, horizon: 2, dt: 0.1, hidden_dim: 4}\n"
                "data: {train: train.npz, val: val.npz, batch_size: 1}\n"
                "training: {epochs: 1, patience: 1, seed: 7}\n"
                "output: {directory: run}\n",
                encoding="utf-8",
            )

            report = run_training_from_yaml(config)

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["metrics"]["samples"], 1)
            for name in (
                "best.pt",
                "last.pt",
                "history.json",
                "report.json",
                "training_curve.png",
                "training_curve.pdf",
            ):
                self.assertTrue((root / "run" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
