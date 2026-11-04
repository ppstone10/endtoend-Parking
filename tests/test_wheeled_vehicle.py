"""轮式基础的物理解析验收；不依赖规划器、MPC或网络。"""

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from interfaces import ControlCmd, VehicleState
from interfaces.wheeled import WheeledControlCmd, WheeledVehicleState
from sim.wheeled_config import WheeledVehicleConfig, load_wheeled_vehicle_config
from sim.wheeled_vehicle import KinematicBicycleModel


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/vehicles/wheeled_research_v1.json"


class TestWheeledConfig(unittest.TestCase):
    def setUp(self):
        self.config = load_wheeled_vehicle_config(CONFIG_PATH)

    def test_research_config_is_explicit_and_geometry_consistent(self):
        c = self.config
        self.assertEqual(c.parameter_source, "research_assumption")
        self.assertTrue(c.parameter_notes)
        self.assertAlmostEqual(c.length, c.wheelbase + c.front_overhang + c.rear_overhang)
        self.assertAlmostEqual(c.min_turning_radius, c.wheelbase / math.tan(c.max_steering_angle))

    def test_metadata_and_json_roundtrip_preserve_identity(self):
        metadata = json.loads(json.dumps(self.config.to_metadata()))
        recovered = WheeledVehicleConfig.from_metadata(metadata)
        self.assertEqual(recovered, self.config)
        self.assertEqual(recovered.identity, self.config.identity)
        with patch.object(Path, "read_text", return_value=json.dumps(metadata)):
            self.assertEqual(load_wheeled_vehicle_config("roundtrip.json"), self.config)

    def test_parameter_change_changes_identity(self):
        changed = replace(self.config, max_steering_angle=0.4)
        self.assertNotEqual(changed.identity, self.config.identity)
        self.assertEqual(replace(self.config, width=int(self.config.width)).identity, self.config.identity)

    def test_invalid_geometry_and_numeric_fields_are_rejected(self):
        for fields in ({"wheelbase": 0}, {"width": math.nan}, {"length": True},
                       {"rear_overhang": -1}, {"max_steering_angle": math.pi / 2},
                       {"max_steering_rate": 0}, {"max_acceleration": -1},
                       {"max_reverse_speed": math.inf}, {"collision_margin": -0.1},
                       {"length": 20}, {"parameter_source": ""}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.config, **fields)

    def test_metadata_requires_exact_model_and_fields(self):
        for field, value in (("motion_model", "differential_drive"),
                             ("pose_reference", "rear_axle"),
                             ("schema_version", True), ("actuation_model", "continuous")):
            data = self.config.to_metadata()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                WheeledVehicleConfig.from_metadata(data)
        for alteration in ("missing", "extra", "extra_parameter"):
            data = self.config.to_metadata()
            if alteration == "missing":
                del data["speed_reference"]
            elif alteration == "extra":
                data["enable_pivot"] = True
            else:
                data["parameters"]["enable_pivot"] = True
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                WheeledVehicleConfig.from_metadata(data)

    def test_old_metadata_is_not_a_wheeled_config(self):
        from sim import MINING_DRILL_RIG
        with self.assertRaises(ValueError):
            WheeledVehicleConfig.from_metadata(MINING_DRILL_RIG.to_metadata())

    def test_bad_json_has_actionable_error(self):
        for contents in ("[]", "{", '{"schema_version": NaN}'):
            with self.subTest(contents=contents), patch.object(Path, "read_text", return_value=contents):
                with self.assertRaises(ValueError):
                    load_wheeled_vehicle_config("bad.json")
        with patch.object(Path, "read_text", side_effect=OSError("cannot read")):
            with self.assertRaisesRegex(ValueError, "无法加载轮式配置"):
                load_wheeled_vehicle_config("missing.json")


class TestWheeledModel(unittest.TestCase):
    def setUp(self):
        self.config = load_wheeled_vehicle_config(CONFIG_PATH)
        self.model = KinematicBicycleModel(self.config)

    def test_state_array_roundtrip_and_no_implicit_legacy_objects(self):
        state = WheeledVehicleState(1, 2, 3, 0.2, 0.3)
        self.assertEqual(WheeledVehicleState.from_array(state.to_array()), state)
        for values in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [[1, 2, 3, 4, 5]],
                       [1, 2, 3, 0, math.nan], [1, 2, 3, True, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                WheeledVehicleState.from_array(values)
        for state_input, command in ((VehicleState(0, 0, 0), WheeledControlCmd(0, 0)),
                                      (state, ControlCmd(0, 0))):
            with self.assertRaises(TypeError):
                self.model.step(state_input, command, 0.1)

    def test_reference_roundtrip_with_asymmetric_body(self):
        config = replace(self.config, front_overhang=0.6, rear_overhang=1.8)
        model = KinematicBicycleModel(config)
        for yaw in (-7.2, -math.pi / 2, 0, 1.7, 9.0):
            center = (3.0, -2.0, yaw)
            rear = model.center_to_rear_pose(center)
            self.assertAlmostEqual(rear[0], 3 - 1.2 * math.cos(yaw))
            self.assertAlmostEqual(rear[1], -2 - 1.2 * math.sin(yaw))
            np.testing.assert_allclose(model.rear_to_center_pose(rear), center, atol=1e-14)
        rear_ahead = KinematicBicycleModel(replace(self.config, wheelbase=1,
                                                  front_overhang=1, rear_overhang=4))
        self.assertEqual(rear_ahead.center_to_rear_pose((0, 0, 0)), (1, 0, 0))
        self.assertEqual(rear_ahead.rear_to_center_pose((1, 0, 0)), (0, 0, 0))

    def test_stopped_steering_cannot_rotate_body(self):
        state = WheeledVehicleState(3, -2, 0.7)
        for _ in range(20):
            state = self.model.step(state, WheeledControlCmd(0, 10), 0.1)
        self.assertEqual((state.x, state.y, state.yaw), (3, -2, 0.7))
        self.assertAlmostEqual(state.steering_angle, self.config.max_steering_angle)
        self.assertEqual(self.model.yaw_rate(state), 0)

    def test_straight_motion_is_exact_and_preserves_input(self):
        state = WheeledVehicleState(1, 2, 0.7, v=1)
        actual = self.model.step(state, WheeledControlCmd(1, 0), 0.4)
        self.assertAlmostEqual(actual.x, 1 + 0.4 * math.cos(0.7))
        self.assertAlmostEqual(actual.y, 2 + 0.4 * math.sin(0.7))
        self.assertEqual(actual.yaw, state.yaw)
        self.assertEqual(state, WheeledVehicleState(1, 2, 0.7, 1, 0))

    def test_forward_reverse_left_right_arcs_match_circle_geometry(self):
        for speed in (-0.7, 0.7):
            for delta in (-0.3, 0.3):
                with self.subTest(speed=speed, delta=delta):
                    state = WheeledVehicleState(2, -1, 0.4, speed, delta)
                    initial_rear = self.model.center_to_rear_pose((state.x, state.y, state.yaw))
                    radius = self.config.wheelbase / math.tan(delta)
                    circle = (initial_rear[0] - radius * math.sin(state.yaw),
                              initial_rear[1] + radius * math.cos(state.yaw))
                    for _ in range(30):
                        state = self.model.step(state, WheeledControlCmd(speed, delta), 0.1)
                    expected_yaw = 0.4 + 3.0 * speed / radius
                    rear = self.model.center_to_rear_pose((state.x, state.y, state.yaw))
                    self.assertAlmostEqual(rear[0], circle[0] + radius * math.sin(expected_yaw), places=11)
                    self.assertAlmostEqual(rear[1], circle[1] - radius * math.cos(expected_yaw), places=11)
                    self.assertAlmostEqual(state.yaw, expected_yaw, places=12)
                    self.assertAlmostEqual(math.hypot(rear[0]-circle[0], rear[1]-circle[1]), abs(radius), places=11)

    def test_rate_and_acceleration_limits_apply_before_motion(self):
        state = self.model.step(WheeledVehicleState(0, 0, 0), WheeledControlCmd(999, 999), 0.1)
        self.assertAlmostEqual(state.v, self.config.max_acceleration * 0.1)
        self.assertAlmostEqual(state.steering_angle, self.config.max_steering_rate * 0.1)
        self.assertLessEqual(abs(self.model.yaw_rate(state)), abs(state.v) / self.config.min_turning_radius)

    def test_amplitude_limits_are_respected(self):
        state = WheeledVehicleState(0, 0, 0)
        for _ in range(100):
            state = self.model.step(state, WheeledControlCmd(999, -999), 0.1)
        self.assertEqual(state.v, self.config.max_forward_speed)
        self.assertEqual(state.steering_angle, -self.config.max_steering_angle)

    def test_reversal_must_visit_zero_before_negative_motion(self):
        state = WheeledVehicleState(0, 0, 0, v=0.05)
        stopped = self.model.step(state, WheeledControlCmd(-999, 0), 0.1)
        self.assertEqual(stopped.v, 0)
        self.assertEqual(stopped.x, state.x)
        reversed_state = self.model.step(stopped, WheeledControlCmd(-999, 0), 0.1)
        self.assertLess(reversed_state.v, 0)
        for _ in range(100):
            reversed_state = self.model.step(reversed_state, WheeledControlCmd(-999, 0), 0.1)
        self.assertEqual(reversed_state.v, -self.config.max_reverse_speed)

    def test_braking_command_does_not_teleport_moving_vehicle_to_rest(self):
        state = WheeledVehicleState(0, 0, 0, v=1)
        actual = self.model.step(state, WheeledControlCmd(0, 0), 0.1)
        self.assertAlmostEqual(actual.v, 1 - self.config.max_deceleration * 0.1)
        self.assertGreater(actual.x, 0)

    def test_near_zero_curvature_is_stable_and_yaw_is_continuous(self):
        state = WheeledVehicleState(0, 0, math.pi-0.01, 1, 1e-14)
        actual = self.model.step(state, WheeledControlCmd(1, 1e-14), 0.1)
        self.assertAlmostEqual(actual.x, 0.1 * math.cos(state.yaw), places=12)
        self.assertAlmostEqual(actual.y, 0.1 * math.sin(state.yaw), places=12)
        turning = WheeledVehicleState(0, 0, math.pi-0.001, 1, 0.3)
        self.assertGreater(self.model.step(turning, WheeledControlCmd(1, 0.3), 0.1).yaw, math.pi)

    def test_center_velocity_includes_reference_offset(self):
        state = WheeledVehicleState(0, 0, 0, 1, 0.3)
        vx, vy = self.model.center_velocity(state)
        omega = math.tan(0.3) / self.config.wheelbase
        self.assertEqual(vx, 1)
        self.assertAlmostEqual(vy, self.config.center_from_rear * omega)
        self.assertGreater(math.hypot(vx, vy), state.v)

    def test_center_velocity_matches_short_arc_derivative(self):
        state = WheeledVehicleState(2, -1, 0.7, -0.8, 0.3)
        dt = 1e-6
        advanced = self.model.step(state, WheeledControlCmd(state.v, state.steering_angle), dt)
        derivative = ((advanced.x-state.x)/dt, (advanced.y-state.y)/dt)
        np.testing.assert_allclose(derivative, self.model.center_velocity(state), atol=1e-7)

    def test_resumed_state_has_no_hidden_model_memory(self):
        state = WheeledVehicleState(0, 0, 0)
        commands = [WheeledControlCmd(1, 0.4)] * 5 + [WheeledControlCmd(-1, -0.4)] * 20
        for command in commands[:10]:
            state = self.model.step(state, command, 0.1)
        clone = WheeledVehicleState.from_array(state.to_array())
        resumed = KinematicBicycleModel(self.config)
        for command in commands[10:]:
            state = self.model.step(state, command, 0.1)
            clone = resumed.step(clone, command, 0.1)
            self.assertEqual(state, clone)

    def test_invalid_inputs_fail_without_mutating_state(self):
        state = WheeledVehicleState(1, 2, 0.3)
        for dt in (0, -1, math.inf, math.nan, True):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.model.step(state, WheeledControlCmd(1, 0), dt)
        for bad_state in (replace(state, v=99), replace(state, steering_angle=1)):
            with self.assertRaises(ValueError):
                self.model.step(bad_state, WheeledControlCmd(0, 0), 0.1)
        for constructor in (lambda: WheeledControlCmd(math.inf, 0),
                            lambda: WheeledControlCmd(True, 0),
                            lambda: WheeledVehicleState(0, 0, math.nan)):
            with self.assertRaises(ValueError):
                constructor()
        self.assertEqual(state, WheeledVehicleState(1, 2, 0.3))

    def test_random_commands_preserve_all_sampled_limits(self):
        rng = np.random.default_rng(11)
        state = WheeledVehicleState(0, 0, 0)
        for _ in range(2000):
            dt = float(rng.uniform(0.01, 0.3))
            command = WheeledControlCmd(float(rng.uniform(-4, 4)), float(rng.uniform(-2, 2)))
            advanced = self.model.step(state, command, dt)
            self.assertGreaterEqual(advanced.v, -self.config.max_reverse_speed)
            self.assertLessEqual(advanced.v, self.config.max_forward_speed)
            self.assertLessEqual(abs(advanced.steering_angle), self.config.max_steering_angle)
            self.assertLessEqual(abs(advanced.steering_angle-state.steering_angle),
                                 self.config.max_steering_rate*dt+1e-14)
            rate = (self.config.max_acceleration if abs(advanced.v) > abs(state.v)
                    else self.config.max_deceleration)
            self.assertLessEqual(abs(advanced.v-state.v), rate*dt+1e-14)
            self.assertGreaterEqual(advanced.v*state.v, 0)
            self.assertTrue(np.isfinite(advanced.to_array()).all())
            state = advanced

    def test_reaching_tiny_target_does_not_cancel_to_zero(self):
        state = WheeledVehicleState(0, 0, 0, 1, 0.3)
        advanced = self.model.step(state, WheeledControlCmd(1e-20, 1e-20), 1)
        self.assertEqual(advanced.v, 1e-20)
        self.assertEqual(advanced.steering_angle, 1e-20)

    def test_extreme_step_is_rejected_without_state_change(self):
        state = WheeledVehicleState(0, 0, 0, 2, 0.3)
        with self.assertRaises(ValueError):
            self.model.step(state, WheeledControlCmd(2, 0.3), 1e308)
        self.assertEqual(state, WheeledVehicleState(0, 0, 0, 2, 0.3))


class TestLegacyIsolation(unittest.TestCase):
    def test_legacy_config_identity_and_differential_pivot_are_unchanged(self):
        from sim import DifferentialDriveModel, MINING_DRILL_RIG, MINING_TRUCK, get_vehicle, load_vehicle_config
        path = CONFIG_PATH.with_name("tracked_drill_rig.json")
        self.assertEqual(load_vehicle_config(path).to_metadata(), json.loads(path.read_text(encoding="utf-8")))
        self.assertIs(MINING_TRUCK, MINING_DRILL_RIG)
        self.assertIs(get_vehicle("mining_truck"), MINING_DRILL_RIG)
        actual = DifferentialDriveModel().step(VehicleState(1, 2, 0), ControlCmd(0, 0.5), 0.1)
        self.assertEqual((actual.x, actual.y, actual.yaw), (1, 2, 0.05))
        with self.assertRaises(ValueError):
            load_vehicle_config(CONFIG_PATH)


if __name__ == "__main__":
    unittest.main()
