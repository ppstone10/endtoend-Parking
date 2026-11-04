"""旧标签必要条件与校准身份/逐控制复核，不运行正式面板。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dataset.wheeled_audit import audit_wheeled_necessary_conditions
from interfaces import GoalPose
from interfaces.wheeled import WheeledControlCmd, WheeledVehicleState
from sim import ParkingEnvironment
from sim.wheeled_config import load_wheeled_vehicle_config
from sim.wheeled_vehicle import KinematicBicycleModel
from verification.wheeled_calibration import inspect_case, precheck, prepare_config, run_case, select_panel
from scripts.calibrate_wheeled_tasks import inspect_formal, case_filename


ROOT = Path(__file__).resolve().parents[1]
VEHICLE = ROOT/"configs/vehicles/wheeled_research_v1.json"


class TestWheeledData(unittest.TestCase):
    def setUp(self):
        self.vehicle = load_wheeled_vehicle_config(VEHICLE)

    def test_model_generated_arc_passes_necessary_conditions_but_is_not_certified(self):
        model = KinematicBicycleModel(self.vehicle)
        state = WheeledVehicleState(0, 0, 0)
        points = [[0, 0, 0]]
        for _ in range(30):
            state = model.step(state, WheeledControlCmd(.5, .3), .1)
            points.append([state.x, state.y, state.yaw])
        result = audit_wheeled_necessary_conditions(points, .1, self.vehicle)
        self.assertEqual(result["classification"], "necessary_conditions_only")
        self.assertFalse(result["direct_reuse_qualified"])

    def test_pivot_and_too_tight_curve_are_rejected(self):
        pivot = audit_wheeled_necessary_conditions([[0, 0, 0], [0, 0, .1]], .2, self.vehicle)
        self.assertIn("body_pivot", pivot["reasons"])
        curve = audit_wheeled_necessary_conditions([[0, 0, 0], [.05, 0, .1]], .2, self.vehicle)
        self.assertIn("curvature_excess", curve["reasons"])

    def test_float32_world_pose_quantization_does_not_invent_steering_failure(self):
        model = KinematicBicycleModel(self.vehicle)
        state = WheeledVehicleState(35, -25, 1.5)
        points = [[state.x, state.y, state.yaw]]
        for _ in range(40):
            state = model.step(state, WheeledControlCmd(.5, .3), .1)
            points.append([state.x, state.y, state.yaw])
        result = audit_wheeled_necessary_conditions(np.asarray(points, dtype=np.float32), .1, self.vehicle)
        self.assertEqual(result["classification"], "necessary_conditions_only", result)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_old_instant_speed_jump_is_separate_from_geometry(self):
        row = audit_wheeled_necessary_conditions([[0, 0, 0], [.1, 0, 0]], .2, self.vehicle)
        self.assertIn("speed_slew_excess", row["reasons"])
        self.assertNotIn("curvature_excess", row["reasons"])

    def test_panel_selects_first_of_each_group_and_rejects_recovery(self):
        items = [dict(scene_name="S1", difficulty=dict(maneuver=m)) for m in ("forward", "forward", "reverse")]
        self.assertEqual(select_panel(items), [0, 2])
        with self.assertRaises(ValueError):
            select_panel([{**items[0], "recovery": {}}])

    def test_prepared_rebuild_rejects_semantic_and_file_drift(self):
        config = prepare_config([ROOT/f"data/task_dataset/tracked_pivot_v8_3000/{s}.npz" for s in ("train", "val")], VEHICLE)
        self.assertEqual(len(config["cases"]), 22)
        altered = deepcopy(config)
        altered["max_steps"] = 601
        with self.assertRaisesRegex(ValueError, "语义"):
            precheck(altered)
        with patch("verification.wheeled_calibration.sha256_file", return_value="drift"):
            with self.assertRaisesRegex(ValueError, "漂移"):
                precheck(config)

    def test_calibration_record_roundtrip_and_tampering_rejected(self):
        config = dict(vehicle_path=str(VEHICLE), model_identity=self.vehicle.identity, dt=.1,
                      max_steps=600, planner_max_seconds=3., case_max_seconds=20.,
                      controller=dict(horizon=8, population=8, iterations=1, seed=7),
                      minimum_requested_ratio=.5, initial_steering_assumption=0.)
        case = dict(index=0, task_id="synthetic", maneuver="forward")
        restored = SimpleNamespace(goal=GoalPose(2, 0, 0), tol_pos=.3, tol_yaw=np.deg2rad(10),
                                   task=SimpleNamespace(scene=SimpleNamespace(env=ParkingEnvironment(world_size=80))))
        start = WheeledVehicleState(0, 0, 0)
        with patch("verification.wheeled_calibration.restore_case", return_value=(restored, start)):
            record = json.loads(json.dumps(run_case(config, case)))
            self.assertTrue(inspect_case(config, case, record))
            altered = deepcopy(record)
            altered["expert_controls"][0][0] += .2
            with self.assertRaises((ValueError, AssertionError)):
                inspect_case(config, case, altered)
            altered = deepcopy(record)
            altered["execution"]["model_identity"] = "0"*64
            with self.assertRaises(ValueError):
                inspect_case(config, case, altered)

    def test_formal_inspection_requires_every_artifact_and_recomputes_gate(self):
        cases = [dict(split="train", index=0), dict(split="val", index=1)]
        config = dict(schema="wheeled_teacher_calibration_v1", cases=cases)
        store = {"prepared.json": json.dumps(config), "config.json": json.dumps(config)}
        digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
        report = dict(schema=config["schema"], formal=True, status="completed", expected_cases=2,
                      prepared_sha256=digest(store["prepared.json"]), gate_passed=False,
                      rows=[dict(case=case, passed=passed) for case, passed in zip(cases, (True, False))])
        store["report.json"] = json.dumps(report)
        for i, case in enumerate(cases):
            store[case_filename(i, case)] = json.dumps(dict(passed=i == 0))
        manifest = {name: digest(value) for name, value in store.items() if name != "prepared.json"}
        store["artifacts.json"] = json.dumps(manifest)
        with patch.object(Path, "read_text", lambda path, **kwargs: store[path.name]), \
                patch("scripts.calibrate_wheeled_tasks.sha256_file", lambda path: digest(store[Path(path).name])), \
                patch("scripts.calibrate_wheeled_tasks.inspect_case", side_effect=lambda cfg, case, row: row["passed"]):
            result = inspect_formal(config, "prepared.json", "output")
            self.assertTrue(result["verified"])
            self.assertFalse(result["gate_passed"])
            bad = dict(manifest)
            del bad[case_filename(1, cases[1])]
            store["artifacts.json"] = json.dumps(bad)
            with self.assertRaisesRegex(ValueError, "分母"):
                inspect_formal(config, "prepared.json", "output")

    def test_wheeled_engine_uses_original_goal_tolerance(self):
        from experiments.wheeled_evaluation import build_wheeled_engine
        engine = build_wheeled_engine(self.vehicle, tol_pos=.2, tol_yaw=.05, max_steps=1,
                                      horizon=2, population=4, iterations=1)
        result = engine.run(WheeledVehicleState(0, 0, 0), GoalPose(.25, 0, 0),
                            lambda s, g: (_ for _ in ()).throw(ValueError("no reference")))
        self.assertFalse(result["success"])
        self.assertEqual(result["failure"], "planning_failure")


if __name__ == "__main__":
    unittest.main()
