"""L4 顶层效率判据（time_dist_ratio / winding）聚合测试。"""

from __future__ import annotations

import unittest

from metrics import EpisodeResult, summarize


def _result(parking_time=10.0, path_length=5.0, pos_err=0.1, yaw_err=0.1,
            success=True, collision=False, failure=None, **meta):
    return EpisodeResult(
        success=success,
        failure=failure,
        steps=int(parking_time / 0.1),
        final_pos_err=pos_err,
        final_yaw_err=yaw_err,
        path_length=path_length,
        parking_time=parking_time,
        tracking_rms=0.05,
        inference_ms=10.0,
        collision=collision,
        meta=meta,
    )


class TestSummarizeTopLevelCriteria(unittest.TestCase):
    def test_time_dist_ratio_computed(self):
        results = [
            _result(parking_time=10.0, path_length=5.0),
            _result(parking_time=20.0, path_length=5.0),
        ]
        summary = summarize(results)
        self.assertAlmostEqual(summary["time_dist_ratio_mean"], 3.0)
        self.assertAlmostEqual(summary["time_dist_ratio_std"], 1.0)

    def test_winding_computed_with_meta(self):
        results = [
            _result(path_length=10.0, start_x=0.0, start_y=0.0, goal_x=5.0, goal_y=0.0),
        ]
        summary = summarize(results)
        self.assertAlmostEqual(summary["winding_mean"], 2.0)

    def test_winding_skipped_without_meta(self):
        results = [_result(path_length=10.0)]
        summary = summarize(results)
        self.assertNotIn("winding_mean", summary)

    def test_time_dist_ratio_skipped_when_path_length_zero(self):
        summary = summarize([_result(path_length=0.0)])
        self.assertNotIn("time_dist_ratio_mean", summary)

    def test_existing_fields_preserved(self):
        results = [_result(), _result(success=False, collision=True, failure="collision")]
        summary = summarize(results)
        self.assertEqual(summary["episodes"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["collision_rate"], 0.5)
        self.assertEqual(summary["failures"], {"collision": 1})
        self.assertIn("final_pos_err_mean", summary)
        self.assertIn("tracking_rms_mean", summary)


if __name__ == "__main__":
    unittest.main()