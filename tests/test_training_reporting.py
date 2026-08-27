"""训练报告与曲线产物测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training import TrainingHistory
from training.reporting import save_training_artifacts


class TestTrainingReporting(unittest.TestCase):
    def test_writes_atomic_history_report_and_two_curve_formats(self):
        history = TrainingHistory(
            train_loss=[2.0, 1.0],
            val_loss=[2.5, 1.5],
            best_epoch=1,
            best_val_loss=1.5,
        )
        with tempfile.TemporaryDirectory() as temp:
            artifacts = save_training_artifacts(
                history,
                {"status": "completed", "metrics": {"ade_m": 0.5}},
                temp,
            )

            self.assertEqual(json.loads(Path(artifacts.history_json).read_text())["best_epoch"], 1)
            self.assertEqual(json.loads(Path(artifacts.report_json).read_text())["status"], "completed")
            self.assertTrue(Path(artifacts.curve_png).exists())
            self.assertTrue(Path(artifacts.curve_pdf).exists())
            self.assertFalse(any(Path(temp).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
