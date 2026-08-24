"""数据集统计与 BEV/专家轨迹叠加图测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.inspection import render_sample_overlay, summarize_dataset


def _data():
    trajs = np.zeros((2, 3, 3), dtype=np.float32)
    trajs[0] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    trajs[1, :2] = [[0, 0, 0], [-1, 0, 0]]
    return {
        "bevs": np.zeros((2, 5, 4, 4), dtype=np.float32),
        "states": np.zeros((2, 5), dtype=np.float32),
        "trajs": trajs,
        "masks": np.array([[1, 1, 1], [1, 1, 0]], dtype=np.float32),
        "bev_meta": {
            "resolution": 0.5,
            "extent": [1.0, 1.0, 1.0, 1.0],
            "channels": ["occupancy", "height", "density", "target", "vehicle"],
            "shape": [5, 4, 4],
        },
        "task_meta": [
            {"scene_name": "S1", "task_type": "T1", "difficulty": {"noise_level": "clean"}},
            {"scene_name": "S2", "task_type": "T2", "difficulty": {"noise_level": "low"}},
        ],
    }


class TestDatasetInspection(unittest.TestCase):
    def test_summary_reports_length_reverse_ratio_and_strata(self):
        summary = summarize_dataset(_data())
        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["trajectory_length_m"]["mean"], 1.5)
        self.assertAlmostEqual(summary["reverse_distance_ratio"], 1.0 / 3.0)
        self.assertEqual(summary["scene_counts"], {"S1": 1, "S2": 1})
        self.assertEqual(summary["task_type_counts"], {"T1": 1, "T2": 1})

    def test_overlay_is_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            render_sample_overlay(_data(), 0, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
