"""轮式MPC、预测一致性与短闭环的独立关口。"""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from interfaces.wheeled import WheeledControlCmd, WheeledTrajectory, WheeledVehicleState
from sim.wheeled_config import load_wheeled_vehicle_config
from sim.wheeled_vehicle import KinematicBicycleModel
from controller.wheeled_mpc import WheeledMPCController, WheeledControlError
from runtime.wheeled_execution import WheeledExecutor, WheeledClosedLoopEngine


def config():
    return load_wheeled_vehicle_config(Path(__file__).resolve().parents[1] /
                                      "configs/vehicles/wheeled_research_v1.json")


def reference(model, speed=0.5, delta=0., steps=45):
    state = WheeledVehicleState(0, 0, 0)
    states = [state]
    for command in ([WheeledControlCmd(speed, delta)]*steps + [WheeledControlCmd(0, delta)]*12):
        state = model.step(state, command, .1)
        states.append(state)
    poses = np.array([[s.x, s.y, s.yaw] for s in states])
    return WheeledTrajectory(poses, .1, model.config.identity), states


class TestWheeledExecution(unittest.TestCase):
    def setUp(self):
        self.model = KinematicBicycleModel(config())
        self.mpc = WheeledMPCController(self.model, dt=.1, horizon=8, population=12, iterations=2, seed=7)

    def test_reference_is_readonly_copy_and_rejects_bad_identity(self):
        points = np.zeros((2, 3))
        trajectory = WheeledTrajectory(points, .1, self.model.config.identity)
        points[0, 0] = 99
        self.assertEqual(trajectory.points[0, 0], 0)
        with self.assertRaises(ValueError):
            trajectory.points[0, 0] = 1
        for bad in (np.zeros((0, 3)), [[0, 0, np.nan]]):
            with self.assertRaises(ValueError):
                WheeledTrajectory(bad, .1, self.model.config.identity)
        with self.assertRaises(ValueError):
            WheeledTrajectory([[0, 0, 0]], .1, "tracked")

    def test_rollout_matches_every_scalar_step(self):
        state = WheeledVehicleState(1, 2, .4, .3, -.2)
        controls = np.array([[1, .5], [-1, -.5], [0, 0], [-.4, .3]])
        actual = self.mpc.rollout(state, controls)
        expected = []
        for speed, delta in controls:
            state = self.model.step(state, WheeledControlCmd(speed, delta), .1)
            expected.append(state.to_array())
        np.testing.assert_array_equal(actual, expected)

    def test_wrong_reference_and_model_are_rejected(self):
        state = WheeledVehicleState(0, 0, 0)
        ref, _ = reference(self.model)
        with self.assertRaises(TypeError):
            self.mpc.compute(Trajectory(ref.points, .1), state)
        with self.assertRaises(ValueError):
            self.mpc.compute(replace(ref, model_identity="0"*64), state)
        with self.assertRaises(TypeError):
            self.mpc.compute(ref, VehicleState(0, 0, 0))
        other = KinematicBicycleModel(replace(config(), max_steering_angle=.4))
        with self.assertRaises(ValueError):
            WheeledExecutor(self.mpc, other)

    def test_execution_matches_saved_first_prediction_and_dt(self):
        ref, _ = reference(self.model, delta=.2)
        state = WheeledVehicleState(0, 0, 0)
        executor = WheeledExecutor(self.mpc, self.model)
        with self.assertRaises(ValueError):
            executor.propose(state, ref, .2)
        advanced = executor.propose(state, ref, .1)
        np.testing.assert_array_equal(advanced.to_array(), self.mpc.last_prediction[0])
        self.assertEqual(executor.max_prediction_error, 0)

    def test_reset_restores_rng_and_progress(self):
        ref, _ = reference(self.model, delta=.2)
        state = WheeledVehicleState(0, 0, 0)
        first = self.mpc.compute(ref, state)
        prediction = self.mpc.last_prediction.copy()
        self.mpc.compute(ref, state)
        self.mpc.reset()
        self.assertEqual(self.mpc.compute(ref, state), first)
        np.testing.assert_array_equal(self.mpc.last_prediction, prediction)

    def test_reference_resampling_keeps_dt_and_unwraps_heading(self):
        ref = WheeledTrajectory([[0, 0, 3.1], [.2, 0, -3.1]], .2, self.model.config.identity)
        points = self.mpc.align_reference(ref)
        self.assertEqual(len(points), 3)
        self.assertGreater(points[1, 2], 3.1)
        self.assertAlmostEqual(points[1, 0], .1)

    def test_all_rejected_candidates_raise_specific_error(self):
        ref, _ = reference(self.model)
        mpc = WheeledMPCController(self.model, population=4, iterations=1,
                                   transition_checker=lambda a, b: False)
        with self.assertRaises(WheeledControlError):
            mpc.compute(ref, WheeledVehicleState(0, 0, 0))

    def test_short_forward_reverse_and_arc_tracking(self):
        for speed, delta in ((.5, 0), (-.5, 0), (.5, .3)):
            with self.subTest(speed=speed, delta=delta):
                ref, states = reference(self.model, speed, delta)
                runner = WheeledClosedLoopEngine(WheeledExecutor(self.mpc, self.model), max_steps=100)
                goal = GoalPose(states[-1].x, states[-1].y, states[-1].yaw)
                result = runner.run(WheeledVehicleState(0, 0, 0), goal, lambda s, g: ref)
                self.assertTrue(result["success"], result["summary"])
                self.assertLess(result["summary"]["position_error"], .3)
                self.assertLess(result["summary"]["yaw_error"], np.deg2rad(10))
                self.assertEqual(result["summary"]["max_prediction_error"], 0)
                self.assertFalse(result["summary"]["collision_checked"])

    def test_initial_goal_and_source_failure_are_explicit(self):
        runner = WheeledClosedLoopEngine(WheeledExecutor(self.mpc, self.model), max_steps=3)
        state = WheeledVehicleState(0, 0, 0)
        result = runner.run(state, GoalPose(0, 0, 0), lambda s, g: None)
        self.assertTrue(result["success"])
        self.assertEqual(result["summary"]["steps"], 0)
        def unavailable(state, goal):
            raise ValueError("no reference")
        failed = runner.run(state, GoalPose(2, 0, 0), unavailable)
        self.assertEqual(failed["failure"], "planning_failure")
        def programming_error(state, goal):
            raise RuntimeError("unexpected")
        with self.assertRaises(RuntimeError):
            runner.run(state, GoalPose(2, 0, 0), programming_error)

    def test_guard_prevents_state_commit(self):
        ref, _ = reference(self.model)
        runner = WheeledClosedLoopEngine(WheeledExecutor(self.mpc, self.model), max_steps=3,
                                         transition_checker=lambda a, b: a == b)
        state = WheeledVehicleState(0, 0, 0)
        result = runner.run(state, GoalPose(2, 0, 0), lambda s, g: ref)
        self.assertEqual(result["failure"], "safety_stop")
        np.testing.assert_array_equal(result["states"], [state.to_array()])

    def test_tracking_with_initial_localization_offset(self):
        ref, states = reference(self.model, steps=70)
        runner = WheeledClosedLoopEngine(WheeledExecutor(self.mpc, self.model), max_steps=140)
        goal = GoalPose(states[-1].x, states[-1].y, states[-1].yaw)
        result = runner.run(WheeledVehicleState(0, .35, .03), goal, lambda s, g: ref)
        self.assertTrue(result["success"], result["summary"])

    def test_bad_initial_state_or_goal_cannot_be_accepted_as_success(self):
        runner = WheeledClosedLoopEngine(WheeledExecutor(self.mpc, self.model), max_steps=3)
        with self.assertRaises(ValueError):
            runner.run(WheeledVehicleState(0, 0, 0, v=99), GoalPose(0, 0, 0), lambda s, g: None)
        with self.assertRaises(ValueError):
            runner.run(WheeledVehicleState(0, 0, 0), GoalPose(float("nan"), 0, 0), lambda s, g: None)

    def test_resource_budget_is_failure_and_does_not_call_source(self):
        runner = WheeledClosedLoopEngine(WheeledExecutor(self.mpc, self.model), max_steps=3, max_seconds=1e-12)
        def must_not_run(state, goal):
            raise AssertionError("预算到限后不应请求参考")
        result = runner.run(WheeledVehicleState(0, 0, 0), GoalPose(2, 0, 0), must_not_run)
        self.assertEqual(result["failure"], "budget_exhausted")
        self.assertEqual(result["summary"]["steps"], 0)


if __name__ == "__main__":
    unittest.main()
