"""旧专家数据可视化参考归档测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import DatasetGenerator, TrainingSample
from interfaces import BEVConfig, BEVTensor, GoalPose, Trajectory, VehicleState
from scripts.archive_visual_references import create_visual_reference_archive


class TestVisualReferenceArchive(unittest.TestCase):
    def test_archive_is_self_describing_readable_and_rendered(self):
        with tempfile.TemporaryDirectory() as temporary:
            cleanup_root = Path(temporary) / "task_dataset"
            source_dir = cleanup_root / "old"
            source_dir.mkdir(parents=True)
            config = BEVConfig(resolution=0.5, extent=(2.0, 2.0, 2.0, 2.0))
            sample = TrainingSample(
                bev=BEVTensor(
                    data=np.zeros((5, *config.shape), dtype=np.float32),
                    resolution=config.resolution,
                    extent=config.extent,
                    channels=[
                        "occupancy",
                        "height",
                        "density",
                        "target",
                        "vehicle",
                    ],
                ),
                goal=GoalPose(1.0, 0.0, np.pi / 2.0),
                state=VehicleState(0.0, 0.0, 0.0),
                expert_trajectory=Trajectory(
                    points=np.array(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, np.pi / 2.0],
                        ]
                    ),
                    dt=0.1,
                ),
                task_meta={
                    "task_id": "archive-test",
                    "scene_name": "S1",
                    "task_type": "T1",
                    "difficulty": {"maneuver": "forward"},
                    "dataset": {
                        "vehicle_model": {
                            "name": "tracked_drill_rig",
                            "model_version": "tracked_pivot_v3",
                        }
                    },
                },
            )
            source = source_dir / "part.npz"
            DatasetGenerator().save([sample], source)
            selection = cleanup_root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "cleanup_root": str(cleanup_root),
                        "samples": [
                            {
                                "source": "old/part.npz",
                                "index": 0,
                                "reasons": ["pivot"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output = cleanup_root / "visual_reference_archive_v3"
            manifest = create_visual_reference_archive(selection, output)
            loaded = DatasetGenerator.load(output / "reference_samples.npz")

            self.assertEqual(manifest["sample_count"], 1)
            self.assertFalse(manifest["full_dataset_recoverable"])
            self.assertEqual(loaded["bevs"].shape[0], 1)
            self.assertTrue((output / "inspection/sample_00.png").is_file())
            self.assertTrue((output / "checksums.sha256").is_file())
            provenance = loaded["task_meta"][0]["dataset"][
                "visual_archive_source"
            ]
            self.assertEqual(provenance["source"], "old/part.npz")
            self.assertEqual(provenance["selection_reasons"], ["pivot"])

    def test_archive_rejects_source_outside_cleanup_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cleanup_root = root / "task_dataset"
            cleanup_root.mkdir()
            selection = cleanup_root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "cleanup_root": str(cleanup_root),
                        "samples": [
                            {"source": "../outside.npz", "index": 0, "reasons": []}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "必须位于"):
                create_visual_reference_archive(
                    selection, cleanup_root / "visual_reference_archive_v3"
                )


if __name__ == "__main__":
    unittest.main()
