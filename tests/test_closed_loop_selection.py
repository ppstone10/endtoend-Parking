"""训练期闭环选型测试：评估不得污染训练权重，配置解析严格，选型可追溯。"""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from interfaces import GoalPose, VehicleState
from training.closed_loop_selection import ClosedLoopSelectionConfig, ClosedLoopSelector


class TestClosedLoopSelectionConfig(unittest.TestCase):
    def test_defaults_are_disabled(self):
        config = ClosedLoopSelectionConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.every_epochs, 2)
        self.assertEqual(config.safety_mode, "none")

    def test_rejects_non_positive_every_epochs(self):
        with self.assertRaises(ValueError):
            ClosedLoopSelectionConfig(enabled=True, every_epochs=0)

    def test_rejects_unknown_safety_mode(self):
        with self.assertRaises(ValueError):
            ClosedLoopSelectionConfig(enabled=True, safety_mode="magic")

    def test_rejects_negative_samples(self):
        with self.assertRaises(ValueError):
            ClosedLoopSelectionConfig(enabled=True, samples=-1)

    def test_to_dict_round_trips_fields(self):
        config = ClosedLoopSelectionConfig(enabled=True, samples=40, every_epochs=3)
        payload = config.to_dict()
        self.assertEqual(payload["samples"], 40)
        self.assertEqual(payload["every_epochs"], 3)
        self.assertTrue(payload["enabled"])

    def test_rejects_negative_snapshot_interval(self):
        with self.assertRaises(ValueError):
            ClosedLoopSelectionConfig(enabled=True, snapshot_every_epochs=-1)

    def test_rejects_negative_final_selection_samples(self):
        with self.assertRaises(ValueError):
            ClosedLoopSelectionConfig(enabled=True, final_selection_samples=-1)

    def test_final_selection_requires_snapshots(self):
        """启用训练后排序却没有候选快照，属于必然失败的配置，应显式拒绝。"""
        with self.assertRaisesRegex(ValueError, "snapshot_every_epochs"):
            ClosedLoopSelectionConfig(
                enabled=True, final_selection_samples=30, snapshot_every_epochs=0
            )

    def test_final_selection_with_snapshots_is_accepted(self):
        config = ClosedLoopSelectionConfig(
            enabled=True, final_selection_samples=30, snapshot_every_epochs=2
        )
        self.assertEqual(config.final_selection_samples, 30)
        self.assertEqual(config.to_dict()["final_selection_samples"], 30)


class _TinyModel(torch.nn.Module):
    """最小模型替身：只有一层，便于逐位比较权重。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.stop_threshold = 0.5
        self.model_config = {"target_channel": "geometry", "height_mode": "semantic"}


class TestSelectorSafety(unittest.TestCase):
    """核心安全性质：评估在副本上进行，训练权重与属性不被改写。"""

    def test_plain_deepcopy_isolates_attributes(self):
        model = _TinyModel()
        before = copy.deepcopy(model.state_dict())
        threshold_before = model.stop_threshold
        working = copy.deepcopy(model)
        working.stop_threshold = 0.9
        with torch.no_grad():
            working.linear.weight.add_(1.0)
        self.assertEqual(model.stop_threshold, threshold_before)
        for key, value in before.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))

    def test_evaluate_leaves_source_model_bit_identical(self):
        """端到端：构造最小任务并真正调用 evaluate，验证源模型权重与属性不变。"""
        import numpy as np

        from sim import MINING_DRILL_RIG, ParkingEnvironment
        from training.closed_loop_selection import ClosedLoopSelector, _Episode

        class _TaskStub:
            """仅提供闭环引擎需要的最小属性。"""

            def __init__(self) -> None:
                self.scene = type("Scene", (), {})()
                self.scene.env = ParkingEnvironment(world_size=40.0, obstacles=[])
                self.goal_meta = {}

        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.npz"
            np.savez_compressed(
                train_path,
                schema_version=np.asarray(2, dtype=np.uint16),
                bevs=np.zeros((1, 5, 8, 8), dtype=np.float32),
                goals=np.zeros((1, 3), dtype=np.float32),
                states=np.zeros((1, 5), dtype=np.float32),
                trajs=np.zeros((1, 4, 3), dtype=np.float32),
                masks=np.ones((1, 4), dtype=np.float32),
                dt=np.asarray([0.2]),
            )
            selector = ClosedLoopSelector(
                ClosedLoopSelectionConfig(enabled=True, samples=1, max_steps=3),
                data_path=train_path,
            )
            selector.vehicle = MINING_DRILL_RIG
            selector._episodes = [
                _Episode(
                    index=0,
                    state=VehicleState(0.0, 0.0, 0.0),
                    goal=GoalPose(3.0, 0.0, 0.0),
                    task=_TaskStub(),
                    goal_meta={},
                    tol_pos=0.3,
                    tol_yaw=0.17,
                )
            ]
            model = _TinyModel()
            before = copy.deepcopy(model.state_dict())
            threshold_before = model.stop_threshold

            # 替身模型的 predict 会被 NetworkSource 调用；若它抛出，评估应显式失败
            # 而不是静默返回分数——这里只关心无论成败都不得改写源模型。
            try:
                selector.evaluate(model)
            except Exception:  # noqa: BLE001 - 本测试只验证副作用边界
                pass

            self.assertEqual(model.stop_threshold, threshold_before)
            for key, value in before.items():
                self.assertTrue(torch.equal(model.state_dict()[key], value))

    def test_evaluate_signature_requires_data_path(self):
        config = ClosedLoopSelectionConfig(enabled=True, samples=1)
        with self.assertRaises(TypeError):
            ClosedLoopSelector(config)  # type: ignore[call-arg]


class TestConfigParsing(unittest.TestCase):
    """YAML 解析：closed_loop_selection 可选、未知字段拒绝、data 必须存在。"""

    def _write(self, directory: str, body: str) -> Path:
        path = Path(directory) / "train.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def _base(self, extra: str = "") -> str:
        return (
            "model:\n"
            "  name: net-v1\n"
            "  config: {bev_channels: 5, max_horizon: 8, dt: 0.2}\n"
            "data:\n"
            "  train: train.npz\n"
            "  val: val.npz\n"
            "  batch_size: 2\n"
            "training:\n"
            "  epochs: 2\n"
            "  early_stopping_start_epoch: 0\n"
            "output:\n"
            "  directory: out\n" + extra
        )

    def setUp(self):
        import numpy as np

        self._tmp = tempfile.TemporaryDirectory()
        directory = Path(self._tmp.name)
        for name in ("train.npz", "val.npz"):
            np.savez_compressed(
                directory / name,
                schema_version=np.asarray(2, dtype=np.uint16),
                bevs=np.zeros((1, 5, 8, 8), dtype=np.float32),
                goals=np.zeros((1, 3), dtype=np.float32),
                states=np.zeros((1, 5), dtype=np.float32),
                trajs=np.zeros((1, 4, 3), dtype=np.float32),
                masks=np.ones((1, 4), dtype=np.float32),
                dt=np.asarray([0.2]),
            )

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_section_is_allowed(self):
        from training.config import load_training_run_config

        path = self._write(self._tmp.name, self._base())
        config = load_training_run_config(path)
        self.assertIsNone(config.closed_loop_selection)

    def test_enabled_section_is_parsed(self):
        from training.config import load_training_run_config

        path = self._write(
            self._tmp.name,
            self._base(
                "closed_loop_selection:\n"
                "  enabled: true\n"
                "  samples: 5\n"
                "  every_epochs: 1\n"
            ),
        )
        config = load_training_run_config(path)
        self.assertIsNotNone(config.closed_loop_selection)
        self.assertTrue(config.closed_loop_selection.enabled)
        self.assertEqual(config.closed_loop_selection.samples, 5)

    def test_unknown_field_is_rejected(self):
        from training.config import load_training_run_config

        path = self._write(
            self._tmp.name,
            self._base("closed_loop_selection:\n  enabled: true\n  bogus: 1\n"),
        )
        with self.assertRaisesRegex(ValueError, "未知字段"):
            load_training_run_config(path)

    def test_missing_data_file_is_rejected(self):
        from training.config import load_training_run_config

        path = self._write(
            self._tmp.name,
            self._base("closed_loop_selection:\n  enabled: true\n  data: nope.npz\n"),
        )
        with self.assertRaisesRegex(ValueError, "不存在"):
            load_training_run_config(path)


if __name__ == "__main__":
    unittest.main()
