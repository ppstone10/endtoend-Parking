"""全能力单元校准与中断恢复测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dataset.calibration import (
    CalibrationResult,
    CalibrationSettings,
    build_calibration_cases,
    run_calibration,
    run_case_with_budget,
)
from sim import MINING_DRILL_RIG, TaskSampler


def _success(case, _settings, _vehicle):
    return CalibrationResult(
        case_id=case.case_id,
        scene_name=case.scene_name,
        task_type=case.task_type,
        status="success",
        duration_s=0.1,
        attempts=1,
        failure_count=0,
        failure_reasons={},
        task_id=f"task-{case.case_id}",
    )


class TestCalibrationPlan(unittest.TestCase):
    def test_plan_covers_all_expert_cells_independent_of_split_count(self):
        sampler = TaskSampler(
            seed=7,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        cases = build_calibration_cases(sampler, samples_per_cell=3)
        cells = {(case.scene_name, case.task_type) for case in cases}
        regular = {cell for cell in cells if cell[0] != "S9_mine_complex"}
        heldout = {cell for cell in cells if cell[0] == "S9_mine_complex"}
        self.assertEqual(len(cases), 105)
        self.assertEqual(len(regular), 30)
        self.assertEqual(len(heldout), 5)
        for scene_name, task_type in cells:
            self.assertEqual(
                sum(
                    case.cell_id == f"{scene_name}/{task_type}" for case in cases
                ),
                3,
            )


class TestCalibrationRecovery(unittest.TestCase):
    def setUp(self):
        self.settings = CalibrationSettings(
            samples_per_cell=1, seed=7, max_retries=0, task_budget_s=1.0
        )
        self.cases = build_calibration_cases(TaskSampler(seed=7), 1)[:3]

    def test_interruption_preserves_completed_cases_and_resume_skips_them(self):
        calls: list[str] = []

        def interrupt_on_second(case, settings, vehicle):
            calls.append(case.case_id)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return _success(case, settings, vehicle)

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(KeyboardInterrupt):
                run_calibration(
                    temp,
                    self.settings,
                    MINING_DRILL_RIG,
                    cases=self.cases,
                    case_runner=interrupt_on_second,
                )
            state = json.loads(Path(temp, "run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["completed_cases"], 1)
            resumed_calls: list[str] = []

            def resumed(case, settings, vehicle):
                resumed_calls.append(case.case_id)
                return _success(case, settings, vehicle)

            report = run_calibration(
                temp,
                self.settings,
                MINING_DRILL_RIG,
                cases=self.cases,
                case_runner=resumed,
            )
            self.assertEqual(
                resumed_calls, [self.cases[1].case_id, self.cases[2].case_id]
            )
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["is_partial"])

    def test_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run_calibration(
                temp,
                self.settings,
                MINING_DRILL_RIG,
                cases=self.cases[:1],
                case_runner=_success,
            )
            with self.assertRaises(RuntimeError):
                run_calibration(
                    temp,
                    CalibrationSettings(
                        samples_per_cell=1,
                        seed=8,
                        max_retries=0,
                        task_budget_s=1.0,
                    ),
                    MINING_DRILL_RIG,
                    cases=self.cases[:1],
                    case_runner=_success,
                )

    def test_failed_case_does_not_stop_following_cases(self):
        def fail_first(case, settings, vehicle):
            if case == self.cases[0]:
                return CalibrationResult(
                    case_id=case.case_id,
                    scene_name=case.scene_name,
                    task_type=case.task_type,
                    status="failed",
                    duration_s=0.2,
                    attempts=1,
                    failure_count=1,
                    failure_reasons={"planned_failure": 1},
                    error="planned",
                )
            return _success(case, settings, vehicle)

        with tempfile.TemporaryDirectory() as temp:
            report = run_calibration(
                temp,
                self.settings,
                MINING_DRILL_RIG,
                cases=self.cases,
                case_runner=fail_first,
            )
            self.assertEqual(report["completed_cases"], 3)
            self.assertEqual(report["status_counts"], {"failed": 1, "success": 2})
            self.assertTrue(all(cell["completion_rate"] == 1.0 for cell in report["cells"]))
            self.assertTrue(Path(temp, "cells.csv").exists())

    def test_hard_budget_returns_terminal_timeout_record(self):
        result = run_case_with_budget(
            self.cases[0],
            CalibrationSettings(
                samples_per_cell=1,
                seed=7,
                max_retries=0,
                task_budget_s=0.01,
            ),
            MINING_DRILL_RIG,
        )
        self.assertEqual(result.status, "task_budget_exceeded")

    def test_real_easy_case_completes_in_isolated_worker(self):
        result = run_case_with_budget(
            self.cases[0],
            CalibrationSettings(
                samples_per_cell=1,
                seed=7,
                max_retries=0,
                task_budget_s=5.0,
            ),
            MINING_DRILL_RIG,
        )
        self.assertEqual(result.status, "success", result.error)
        self.assertIsNotNone(result.task_id)


if __name__ == "__main__":
    unittest.main()
