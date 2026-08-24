"""数据集统计与 BEV/专家轨迹叠加图测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.inspection import (
    _to_local,
    _vehicle_polygon,
    render_sample_overlay,
    select_representative_indices,
    summarize_dataset,
)
from dataset.maneuver import require_maneuver_consistency


def _data():
    trajs = np.zeros((2, 3, 3), dtype=np.float32)
    trajs[0] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    trajs[1, :2] = [[0, 0, 0], [-1, 0, 0]]
    return {
        "bevs": np.zeros((2, 5, 4, 4), dtype=np.float32),
        "states": np.zeros((2, 5), dtype=np.float32),
        "goals": np.array([[2, 0, 0], [-1, 0, 0]], dtype=np.float32),
        "trajs": trajs,
        "masks": np.array([[1, 1, 1], [1, 1, 0]], dtype=np.float32),
        "bev_meta": {
            "resolution": 0.5,
            "extent": [1.0, 1.0, 1.0, 1.0],
            "channels": ["occupancy", "height", "density", "target", "vehicle"],
            "shape": [5, 4, 4],
        },
        "task_meta": [
            {
                "scene_name": "S1",
                "task_type": "T1",
                "difficulty": {"noise_level": "clean", "maneuver": "forward"},
            },
            {
                "scene_name": "S2",
                "task_type": "T2",
                "difficulty": {"noise_level": "low", "maneuver": "reverse"},
            },
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
        self.assertEqual(summary["maneuver_consistency"]["consistent_count"], 2)
        self.assertEqual(summary["maneuver_consistency"]["inconsistent_count"], 0)

    def test_overlay_is_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            render_sample_overlay(_data(), 0, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_representative_selection_prioritizes_task_type_coverage(self):
        data = {
            "trajs": np.zeros((6, 1, 3), dtype=np.float32),
            "task_meta": [
                {"task_type": "T1"},
                {"task_type": "T1"},
                {"task_type": "T2"},
                {"task_type": "T3"},
                {"task_type": "T4"},
                {"task_type": "T5"},
            ],
        }
        indices = select_representative_indices(data, 5)
        selected_types = {data["task_meta"][index]["task_type"] for index in indices}
        self.assertEqual(selected_types, {"T1", "T2", "T3", "T4", "T5"})

    def test_representative_selection_rejects_misaligned_metadata(self):
        data = {
            "trajs": np.zeros((2, 1, 3), dtype=np.float32),
            "task_meta": [{"task_type": "T1"}],
        }
        with self.assertRaisesRegex(ValueError, "task_meta"):
            select_representative_indices(data, 1)

    def test_vehicle_footprint_and_goal_heading_use_local_coordinates(self):
        footprint = _vehicle_polygon(np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.ptp(footprint[:, 0])), 3.0)
        self.assertAlmostEqual(float(np.ptp(footprint[:, 1])), 6.0)

        goal_local = _to_local(
            np.array([[10.0, 12.0, np.pi]]),
            x=10.0,
            y=10.0,
            yaw=np.pi / 2.0,
        )[0]
        np.testing.assert_allclose(goal_local, [2.0, 0.0, np.pi / 2.0], atol=1e-7)

    def test_strict_maneuver_gate_rejects_inconsistent_archive_summary(self):
        data = _data()
        data["task_meta"][0]["difficulty"]["maneuver"] = "reverse"
        summary = summarize_dataset(data)
        self.assertEqual(summary["maneuver_consistency"]["inconsistent_count"], 1)
        with self.assertRaisesRegex(ValueError, "机动一致性门禁失败"):
            require_maneuver_consistency(summary)

    def test_strict_maneuver_gate_rejects_missing_task_declaration(self):
        data = _data()
        data["task_meta"] = None
        summary = summarize_dataset(data)
        self.assertEqual(summary["maneuver_consistency"]["missing_maneuver_count"], 2)
        with self.assertRaisesRegex(ValueError, "缺失声明 2"):
            require_maneuver_consistency(summary)


if __name__ == "__main__":
    unittest.main()
