"""轮式专家时序、弧线车体检查与专家闭环关口。"""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from experiments.wheeled_evaluation import build_wheeled_engine
from interfaces import GoalPose
from interfaces.wheeled import WheeledControlCmd, WheeledVehicleState
from planner.wheeled_expert import WheeledReedsSheppPlanner, WheeledPlanningError
from planner.wheeled_safety import WheeledTransitionChecker
from sim import ParkingEnvironment, RectangleObstacle
from sim.wheeled_config import load_wheeled_vehicle_config
from sim.wheeled_vehicle import KinematicBicycleModel


class TestWheeledPlanner(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1]/"configs/vehicles/wheeled_research_v1.json"
        self.config = load_wheeled_vehicle_config(path)
        self.model = KinematicBicycleModel(self.config)
        self.env = ParkingEnvironment(world_size=80)
        self.checker = WheeledTransitionChecker(self.model, self.env, dt=.1)
        self.planner = WheeledReedsSheppPlanner(self.model, self.checker, dt=.1)

    def verify_plan(self, plan, goal):
        self.assertEqual(plan.trajectory.model_identity, self.config.identity)
        self.assertEqual(len(plan.states), len(plan.controls)+1)
        self.assertEqual(plan.states[-1, 3], 0)
        for before, command, after in zip(plan.states[:-1], plan.controls, plan.states[1:]):
            start = WheeledVehicleState.from_array(before)
            advanced = self.model.step(start, WheeledControlCmd(*command), .1)
            np.testing.assert_array_equal(advanced.to_array(), after)
            self.assertTrue(self.checker(start, advanced))
            if advanced.v == 0:
                self.assertEqual(advanced.yaw, start.yaw)
        np.testing.assert_allclose(plan.states[-1, :2], [goal.x, goal.y], atol=1e-7)
        self.assertLess(abs(np.arctan2(np.sin(plan.states[-1, 2]-goal.yaw),
                                      np.cos(plan.states[-1, 2]-goal.yaw))), 1e-7)

    def test_forward_reverse_and_stationary_target(self):
        for goal in (GoalPose(2, 0, 0), GoalPose(-2, 0, 0), GoalPose(0, 0, 0)):
            with self.subTest(goal=goal):
                self.verify_plan(self.planner.plan(WheeledVehicleState(0, 0, 0), goal), goal)

    def test_heading_change_uses_moving_maneuver_and_stationary_wheel_steering(self):
        goal = GoalPose(3, 2, .4)
        plan = self.planner.plan(WheeledVehicleState(0, 0, 0), goal)
        self.verify_plan(plan, goal)
        changed_heading = np.abs(np.diff(plan.states[:, 2])) > 1e-12
        self.assertTrue(changed_heading.any())
        self.assertTrue(np.all(np.abs(plan.states[1:, 3][changed_heading]) > 0))
        self.assertTrue(np.any((plan.states[1:, 3] == 0) &
                               (np.abs(np.diff(plan.states[:, 4])) > 0)))

    def test_nonzero_initial_speed_has_legal_braking_prefix(self):
        goal = GoalPose(3, 0, 0)
        plan = self.planner.plan(WheeledVehicleState(0, 0, 0, .5, 0), goal)
        self.verify_plan(plan, goal)
        self.assertAlmostEqual(plan.states[1, 3], .4)

    def test_collision_at_start_or_goal_and_budget_fail_explicitly(self):
        env = ParkingEnvironment(world_size=80, obstacles=[RectangleObstacle(-1, 1, -1, 1)])
        bad = WheeledReedsSheppPlanner(self.model, WheeledTransitionChecker(self.model, env, dt=.1))
        with self.assertRaisesRegex(WheeledPlanningError, "起点"):
            bad.plan(WheeledVehicleState(0, 0, 0), GoalPose(10, 0, 0))
        with self.assertRaisesRegex(WheeledPlanningError, "目标"):
            bad.plan(WheeledVehicleState(10, 0, 0), GoalPose(0, 0, 0))
        limited = WheeledReedsSheppPlanner(self.model, self.checker, max_steps=2)
        with self.assertRaises(WheeledPlanningError):
            limited.plan(WheeledVehicleState(0, 0, 0), GoalPose(2, 0, 0))

    def test_transition_checker_rejects_non_wheeled_rotation(self):
        state = WheeledVehicleState(0, 0, 0)
        self.assertFalse(self.checker(state, replace(state, yaw=.1)))

    def test_wall_separating_free_endpoints_fails_without_target_replacement(self):
        env = ParkingEnvironment(world_size=40, obstacles=[RectangleObstacle(-.1, .1, -20, 20)])
        checker = WheeledTransitionChecker(self.model, env, dt=.1)
        planner = WheeledReedsSheppPlanner(self.model, checker, max_seconds=1)
        start, goal = WheeledVehicleState(-6, 0, 0), GoalPose(6, 0, 0)
        self.assertTrue(checker.pose_free(start))
        self.assertTrue(checker.pose_free(WheeledVehicleState(6, 0, 0)))
        with self.assertRaises(WheeledPlanningError):
            planner.plan(start, goal)

    def test_runtime_refuses_checker_from_other_dt_or_model(self):
        with self.assertRaises(ValueError):
            build_wheeled_engine(self.config, dt=.2, transition_checker=self.checker)
        other = replace(self.config, collision_margin=.3)
        with self.assertRaises(ValueError):
            build_wheeled_engine(other, transition_checker=self.checker)

    def test_transition_checker_detects_collision_between_free_endpoints(self):
        env = ParkingEnvironment(world_size=80, obstacles=[RectangleObstacle(-.1, .1, -1, 1)])
        checker = WheeledTransitionChecker(self.model, env, dt=10)
        before = WheeledVehicleState(-5, 0, 0, 1, 0)
        after = self.model.step(before, WheeledControlCmd(1, 0), 10)
        self.assertTrue(checker.pose_free(before))
        self.assertTrue(checker.pose_free(after))
        self.assertFalse(checker(before, after))

    def test_expert_reference_runs_through_wheeled_mpc_and_geometry(self):
        for goal in (GoalPose(2, 0, 0), GoalPose(-2, 0, 0), GoalPose(3, 2, .4)):
            with self.subTest(goal=goal):
                start = WheeledVehicleState(0, 0, 0)
                plan = self.planner.plan(start, goal)
                self.assertLessEqual(len(plan.controls), 600, "参考本身必须放入项目原600步预算")
                engine = build_wheeled_engine(self.config, max_steps=600, transition_checker=self.checker,
                                              horizon=8, population=8, iterations=1, seed=7)
                result = engine.run(start, goal, lambda s, g: plan.trajectory)
                self.assertTrue(result["success"], result["summary"])
                self.assertTrue(result["summary"]["collision_checked"])
                self.assertEqual(result["summary"]["max_prediction_error"], 0)


if __name__ == "__main__":
    unittest.main()
