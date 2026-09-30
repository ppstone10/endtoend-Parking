"""闭环验证记分卡测试：判据评估、配对差值与决策树归因。"""

from __future__ import annotations

import unittest

from experiments.comparison import (
    Gate,
    build_scorecard,
    format_markdown,
)


def _report(
    name: str,
    *,
    bev: str = "sensor",
    traj: str = "network",
    executor: str = "mpc_vehicle",
    success: float,
    collision: float,
    pos_err: float = 0.2,
    yaw_err_deg: float = 5.0,
    tracking_rms: float = 0.03,
    yaw_series: tuple[float, ...] = (),
    model_label: str | None = None,
) -> dict:
    episodes = [
        {
            "cycles": {"yaw_err_end_deg": value},
            "success": True,
            "failure": None,
        }
        for value in yaw_series
    ]
    return {
        "experiment": {
            "name": name,
            "bev_source": bev,
            "trajectory_source": traj,
            "executor": executor,
            "safety_mode": "none",
            "description": name,
        },
        "model_label": model_label,
        "overall": {
            "evaluated_samples": max(len(episodes), 100),
            "success_rate": success,
            "collision_rate": collision,
            "final_pos_err_mean": pos_err,
            "final_yaw_err_mean": yaw_err_deg * 3.141592653589793 / 180.0,
            "tracking_rms_mean": tracking_rms,
            "failures": {},
            "reconstruct_failures": 0,
        },
        "episodes": episodes,
    }


class TestGate(unittest.TestCase):
    def test_upper_bound_gate(self):
        gate = Gate("collision_rate", "<=", 0.1, "碰撞率")
        self.assertTrue(gate.evaluate(0.05))
        self.assertFalse(gate.evaluate(0.2))
        self.assertIsNone(gate.evaluate(None))

    def test_lower_bound_gate(self):
        gate = Gate("success_rate", ">=", 0.7, "成功率")
        self.assertTrue(gate.evaluate(0.7))
        self.assertFalse(gate.evaluate(0.69))
        self.assertIsNone(gate.evaluate(None))


class TestScorecard(unittest.TestCase):
    def test_control_layer_failure_short_circuits_decision(self):
        reports = {
            "E1": _report("E1", traj="expert", success=0.5, collision=0.1),
        }
        scorecard = build_scorecard(reports)
        self.assertEqual(scorecard["decision"]["branch"], "control_layer")

    def test_planning_layer_when_control_ok_but_network_fails(self):
        reports = {
            "E1": _report("E1", traj="expert", success=1.0, collision=0.0),
            "E3": _report("E3", bev="gt", success=0.35, collision=0.09),
            "E5": _report("E5", success=0.30, collision=0.12),
        }
        scorecard = build_scorecard(reports)
        self.assertEqual(scorecard["decision"]["branch"], "planning_layer")
        self.assertFalse(scorecard["gates"]["E1"][0]["passed"] is False)

    def test_perception_layer_when_sensor_bev_loses_against_gt(self):
        reports = {
            "E1": _report("E1", traj="expert", success=1.0, collision=0.0),
            "E3": _report("E3", bev="gt", success=0.80, collision=0.02),
            "E5": _report("E5", success=0.30, collision=0.02),
        }
        scorecard = build_scorecard(reports)
        self.assertEqual(scorecard["decision"]["branch"], "perception_layer")

    def test_pass_branch_when_planning_gates_hold(self):
        reports = {
            "E1": _report("E1", traj="expert", success=1.0, collision=0.0),
            "E3": _report("E3", bev="gt", success=0.75, collision=0.05),
            "E5": _report("E5", success=0.72, collision=0.06),
        }
        scorecard = build_scorecard(reports)
        self.assertEqual(scorecard["decision"]["branch"], "pass")

    def test_perception_gates_use_real_deltas(self):
        reports = {
            "E1": _report("E1", traj="expert", success=1.0, collision=0.0),
            "E4": _report("E4", traj="expert", success=0.9, collision=0.04),
        }
        scorecard = build_scorecard(reports)
        success_gate, collision_gate = scorecard["gates"]["E4_vs_E1"]
        self.assertAlmostEqual(success_gate["value"], -0.1, places=9)
        self.assertFalse(success_gate["passed"])
        self.assertAlmostEqual(collision_gate["value"], 0.04, places=9)
        self.assertFalse(collision_gate["passed"])

    def test_e2_report_is_surfaced_as_reference_not_gate(self):
        reports = {
            "E1": _report("E1", traj="expert", success=1.0, collision=0.0),
            "E3": _report("E3", bev="gt", success=0.3, collision=0.05),
            "E2@v9d-v7data": _report(
                "E2", bev="gt", executor="ideal_path", success=0.35, collision=0.06,
                model_label="v9d-v7data",
            ),
        }
        scorecard = build_scorecard(reports)
        self.assertEqual(scorecard["decision"]["e2"]["key"], "E2@v9d-v7data")
        self.assertNotIn("E2", scorecard["gates"])

    def test_yaw_series_slope_is_reported(self):
        reports = {
            "E1": _report("E1", traj="expert", success=1.0, collision=0.0),
            "E5": _report(
                "E5", success=0.2, collision=0.1, yaw_series=(10.0, 20.0, 30.0)
            ),
        }
        scorecard = build_scorecard(reports)
        row = next(r for r in scorecard["rows"] if r["experiment"] == "E5")
        self.assertAlmostEqual(row["yaw_error_slope_deg_per_sample"], 10.0)

    def test_markdown_contains_decision_and_rows(self):
        reports = {
            "E1": _report("E1", traj="expert", success=1.0, collision=0.0),
            "E5": _report("E5", success=0.2, collision=0.1),
        }
        markdown = format_markdown(build_scorecard(reports))
        self.assertIn("# 闭环五组对照实验记分卡", markdown)
        self.assertIn("决策树落点", markdown)
        self.assertIn("planning_layer", markdown)

    def test_limitations_note_small_sample_size(self):
        reports = {"E1": _report("E1", traj="expert", success=1.0, collision=0.0)}
        reports["E1"]["overall"]["evaluated_samples"] = 34
        reports["E1"]["episodes"] = reports["E1"]["episodes"][:34]
        scorecard = build_scorecard(reports)
        self.assertTrue(any("样本量偏小" in note for note in scorecard["limitations"]))

    def test_limitations_note_reconstruct_failures(self):
        reports = {"E1": _report("E1", traj="expert", success=1.0, collision=0.0)}
        reports["E1"]["overall"]["reconstruct_failures"] = 117
        scorecard = build_scorecard(reports)
        self.assertTrue(any("S3/S5/S7/S8" in note for note in scorecard["limitations"]))


if __name__ == "__main__":
    unittest.main()
