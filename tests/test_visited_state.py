"""L2 闭环访问状态轨迹质量指标与评测脚本测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from metrics.visited_state import (
    VisitedStateRecord,
    analyze_visited_state_records,
)


def _record(
    step: int,
    *,
    state=(0.0, 0.0, 0.0),
    goal=(5.0, 0.0, 0.0),
    net_length: int = 10,
    expert_length: int = 10,
    net_end_yaw=0.0,
    net_flips: int = 0,
    scene="S1_parking_lot",
    task_type="T1",
    maneuver="forward",
    noise_level="clean",
    task_id="S1-T1-0",
) -> VisitedStateRecord:
    def _straight(n: int, length_m: float = 0.5):
        t = np.linspace(0.0, length_m * (n - 1), n)
        return np.stack([t, np.zeros(n), np.zeros(n)], axis=1)

    net = _straight(net_length)
    if net_end_yaw != 0.0 or net_flips:
        net[:, 2] = np.linspace(0.0, net_end_yaw, net_length)
        if net_flips:
            net[net_length // 2 :, 0] *= -1.0
    expert = _straight(expert_length)
    return VisitedStateRecord(
        step=step,
        state=np.asarray(state, dtype=np.float64),
        network_points=net,
        expert_points=expert,
        goal=np.asarray(goal, dtype=np.float64),
        meta={
            "task_id": task_id,
            "scene_name": scene,
            "task_type": task_type,
            "maneuver": maneuver,
            "noise_level": noise_level,
        },
    )


class TestVisitedStateMetrics(unittest.TestCase):
    def test_pair_error_identical_is_zero(self):
        records = [
            _record(0, net_length=10, expert_length=10),
            _record(1, net_length=10, expert_length=10),
        ]
        report, rows = analyze_visited_state_records(records)
        self.assertAlmostEqual(report["overall"]["vs_ade_m"], 0.0, places=2)
        self.assertAlmostEqual(report["overall"]["vs_fde_m"], 0.0, places=2)
        self.assertEqual(len(rows), 2)

    def test_near_rows_aggregated_separately(self):
        # 一个距目标近（d=2）、一个远（d=4）。
        near = _record(0, state=(3.0, 0.0, 0.0), goal=(5.0, 0.0, 0.0),
                       net_length=4, expert_length=4)
        far = _record(1, state=(1.0, 0.0, 0.0), goal=(5.0, 0.0, 0.0),
                      net_length=10, expert_length=10)
        report, _ = analyze_visited_state_records([near, far])
        self.assertIn("near_len_m", report["overall"])
        self.assertEqual(report["overall"]["near_replans"], 1)

    def test_consistency_counts_yaw_jump(self):
        # 相邻两次重规划终点航向从 0° 跳到 90°。
        r0 = _record(0, net_end_yaw=0.0, task_id="S1-T1-0")
        r1 = _record(1, net_end_yaw=np.deg2rad(60.0), task_id="S1-T1-0")
        report, _ = analyze_visited_state_records([r0, r1])
        self.assertGreater(
            report["overall"]["consistency_yaw_jumps_per_sample"], 0.0
        )

    def test_groups_by_scene(self):
        records = [
            _record(0, scene="S1_parking_lot", task_id="a"),
            _record(1, scene="S2_diagonal_lot", task_id="b"),
        ]
        report, _ = analyze_visited_state_records(records)
        self.assertIn("S1_parking_lot", report["groups"]["scene"])
        self.assertIn("S2_diagonal_lot", report["groups"]["scene"])

    def test_duplicate_task_ids_counted_once_in_samples(self):
        records = [
            _record(0, task_id="a"),
            _record(1, task_id="a"),
            _record(2, task_id="b"),
        ]
        report, _ = analyze_visited_state_records(records)
        self.assertEqual(report["overall"]["samples"], 2)


class TestEvalScriptSmoke(unittest.TestCase):
    def test_script_imports_and_metrics_module(self):
        # 脚本 import 本身会触发全链路加载；这里验证指标模块可直接使用。
        from scripts.evaluate_visited_state_trajectory import main
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()