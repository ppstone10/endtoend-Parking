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
            self.assertEqual(len(report["history"]["train_rollout_ade_m"]), 1)
            self.assertEqual(len(report["history"]["val_rollout_fde_m"]), 1)
            for name in (
                "best.pt",
                "last.pt",
                "history.json",
                "report.json",
                "training_curve.png",
                "training_curve.pdf",
            ):
                self.assertTrue((root / "run" / name).exists(), name)

    def test_variable_model_writes_calibrated_deployment_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_dataset(root / "train.npz", 2)
            _write_dataset(root / "val.npz", 2)
            config = root / "run.yaml"
            config.write_text(
                "model:\n"
                "  name: net-v1\n"
                "  config: {bev_channels: 5, max_horizon: 2, dt: 0.1, hidden_dim: 4}\n"
                "data: {train: train.npz, val: val.npz, batch_size: 1}\n"
                "training: {epochs: 1, patience: 1, seed: 7, balance_stop_loss: false, stop_target_mode: cumulative}\n"
                "output: {directory: run}\n",
                encoding="utf-8",
            )

            report = run_training_from_yaml(config)

            self.assertTrue((root / "run" / "deployment.pt").is_file())
            self.assertTrue((root / "run" / "stop_threshold_calibration.json").is_file())
            self.assertIn("stop_threshold", report["calibration"])
            self.assertTrue(report["checkpoints"]["deployment"].endswith("deployment.pt"))


if __name__ == "__main__":
    unittest.main()
