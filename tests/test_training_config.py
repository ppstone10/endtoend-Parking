"""配置化训练 YAML 的契约测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from training.config import load_training_run_config


class TestTrainingRunConfig(unittest.TestCase):
    def test_loads_safe_yaml_and_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train.npz").touch()
            (root / "val.npz").touch()
            config_path = root / "run.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      name: net-v1
                      config:
                        bev_channels: 5
                        max_horizon: 12
                        dt: 0.1
                    data:
                      train: train.npz
                      val: val.npz
                      batch_size: 4
                    training:
                      epochs: 5
                      learning_rate: 0.001
                      patience: 2
                      shuffle_train: true
                      balance_stop_loss: true
                      teacher_forcing_start: 1.0
                      teacher_forcing_end: 0.2
                      teacher_forcing_decay_epochs: 4
                      early_stopping_start_epoch: 4
                      stop_target_mode: cumulative
                    output:
                      directory: runs/net-v1
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_training_run_config(config_path)

            self.assertEqual(config.model_name, "net-v1")
            self.assertEqual(config.model_config["max_horizon"], 12)
            self.assertEqual(config.batch_size, 4)
            self.assertEqual(config.train_data, (root / "train.npz").resolve())
            self.assertEqual(config.output_dir, (root / "runs/net-v1").resolve())
            self.assertEqual(config.trainer.checkpoint_dir, str(config.output_dir))
            self.assertTrue(config.trainer.shuffle_train)
            self.assertTrue(config.trainer.balance_stop_loss)
            self.assertEqual(config.trainer.teacher_forcing_ratio(4), 0.2)
            self.assertEqual(config.trainer.early_stopping_start_epoch, 4)
            self.assertEqual(config.trainer.stop_target_mode, "cumulative")

    def test_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            path.write_text(
                "model: {name: net-v0}\n"
                "data: {train: train.npz, val: val.npz}\n"
                "training: {epochs: 1, mystery: true}\n"
                "output: {directory: runs/test}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "未知字段.*mystery"):
                load_training_run_config(path)

    def test_safe_loader_rejects_python_object_tags(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unsafe.yaml"
            path.write_text(
                "!!python/object/apply:os.system ['echo unsafe']",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "YAML"):
                load_training_run_config(path)

    def test_rejects_yaml_values_that_are_not_json_serializable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train.npz").touch()
            (root / "val.npz").touch()
            path = root / "dated.yaml"
            path.write_text(
                "model: {name: net-v0, config: {stamp: 2026-08-27}}\n"
                "data: {train: train.npz, val: val.npz}\n"
                "training: {epochs: 1}\n"
                "output: {directory: runs/test}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "可序列化"):
                load_training_run_config(path)


if __name__ == "__main__":
    unittest.main()
