"""运行时轨迹安全门禁测试。"""

import unittest

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from planner.collision import RectangleFootprintCollisionChecker
from controller import MPCController
from runtime import (
    FootprintTrajectorySafetyChecker,
    SafetyShieldSource,
)
from sim import ParkingEnvironment, RectangleObstacle
from sim import DifferentialDriveModel
from runtime import ClosedLoopEngine, TerminalChecker


class _Source:
    def __init__(self, points):
        self.trajectory = Trajectory(np.asarray(points, dtype=np.float64), dt=0.2)
        self.calls = 0

    def begin(self, start, goal):
        pass

    def next_trajectory(self, state):
        self.calls += 1
        return self.trajectory, 1.0


class TestSafetyShieldSource(unittest.TestCase):
    def setUp(self):
        env = ParkingEnvironment(
            world_size=20.0,
            obstacles=[RectangleObstacle(1.8, 2.2, -0.2, 0.2)],
        )
        footprint = RectangleFootprintCollisionChecker(
            env,
            vehicle_length=1.0,
            vehicle_width=0.5,
            collision_margin=0.0,
            resolution=0.05,
        )
        self.checker = FootprintTrajectorySafetyChecker(footprint)
        self.start = VehicleState(0.0, 0.0, 0.0)
        self.goal = GoalPose(3.0, 0.0, 0.0)

    def test_safe_primary_passes_without_fallback(self):
        primary = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        fallback = _Source([[0.0, -2.0, 0.0], [3.0, -2.0, 0.0]])
        shield = SafetyShieldSource(primary, fallback, self.checker)
        shield.begin(self.start, self.goal)
        trajectory, elapsed = shield.next_trajectory(self.start)
        self.assertIs(trajectory, primary.trajectory)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual(shield.safety_stats()["interventions"], 0)
        self.assertEqual(elapsed, 1.0)

    def test_swept_collision_switches_to_safe_fallback_and_records_reason(self):
        primary = _Source([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        fallback = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        shield = SafetyShieldSource(primary, fallback, self.checker)
        shield.begin(self.start, self.goal)
        trajectory, elapsed = shield.next_trajectory(self.start)
        stats = shield.safety_stats()
        self.assertIs(trajectory, fallback.trajectory)
        self.assertEqual(stats["checks"], 1)
        self.assertEqual(stats["interventions"], 1)
        self.assertEqual(stats["reasons"], {"swept_collision": 1})
        self.assertEqual(elapsed, 2.0)

    def test_unsafe_fallback_is_never_executed(self):
        unsafe = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        shield = SafetyShieldSource(_Source(unsafe), _Source(unsafe), self.checker)
        shield.begin(self.start, self.goal)
        with self.assertRaisesRegex(RuntimeError, "回退轨迹仍不安全"):
            shield.next_trajectory(self.start)
        self.assertEqual(shield.safety_stats()["fallback_failures"], 1)

    def test_transition_guard_switches_before_entering_planner_margin(self):
        primary = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        fallback = _Source([[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]])
        shield = SafetyShieldSource(primary, fallback, self.checker)
        shield.begin(self.start, self.goal)
        replacement, _ = shield.guard_transition(
            self.start, VehicleState(1.4, 0.0, 0.0)
        )
        stats = shield.safety_stats()
        self.assertIs(replacement, fallback.trajectory)
        self.assertEqual(stats["transition_checks"], 1)
        self.assertEqual(stats["prevented_transitions"], 1)
        self.assertEqual(stats["interventions"], 0)
        self.assertEqual(stats["transition_prevention_rate"], 1.0)
        self.assertEqual(stats["reasons"], {"next_state_swept_collision": 1})

    def test_unavailable_fallback_becomes_safety_stop_not_engine_exception(self):
        unsafe = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        shield = SafetyShieldSource(_Source(unsafe), _Source(unsafe), self.checker)
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(max_v=1.0, max_omega=1.0),
            mpc=MPCController(dt=0.1, horizon=3, seed=0),
            source=shield,
            terminal=TerminalChecker(),
            max_steps=3,
        )
        result = engine.run(self.start, self.goal)
        self.assertEqual(result.failure, "safety_stop")
        self.assertFalse(result.collision)
        self.assertEqual(result.meta["safety_shield"]["safety_stops"], 1)

    def test_primary_runtime_error_is_not_misclassified_as_safety_stop(self):
        class BrokenPrimary(_Source):
            def next_trajectory(self, state):
                raise RuntimeError("model failure")

        safe = [[0.0, -1.0, 0.0], [3.0, -1.0, 0.0]]
        shield = SafetyShieldSource(
            BrokenPrimary(safe),
            _Source(safe),
            self.checker,
        )
        engine = ClosedLoopEngine(
            vehicle_model=DifferentialDriveModel(max_v=1.0, max_omega=1.0),
            mpc=MPCController(dt=0.1, horizon=3, seed=0),
            source=shield,
            terminal=TerminalChecker(),
            max_steps=3,
        )
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            engine.run(self.start, self.goal)


if __name__ == "__main__":
    unittest.main()
