import copy
import unittest
import numpy as np

from interfaces import VehicleState, GoalPose, Trajectory, ControlCmd
from runtime.termination import TerminalChecker
from sim import DifferentialDriveModel
from verification.terminal_capture import TerminalCaptureMPC
from verification.reference_certificate import certify_reference


class OpenSpace:
    def pose_free(self, x, y, yaw):
        return True
    def swept_segment_free(self, start, end):
        return True


class StraightController:
    dt = .1
    def __init__(self):
        self.vehicle_model = DifferentialDriveModel()
        self.goal = GoalPose(.2, 0, 0)
        self.terminal = TerminalChecker(.05, .1)
        self._rng = np.random.default_rng(42)
    def compute(self, trajectory, state):
        self._rng.random()
        return ControlCmd(1., 0.)


class CertificateTests(unittest.TestCase):
    def test_certificate_isolated_from_live_rng_and_observer(self):
        controller = StraightController(); calls = []
        controller.compute = lambda *args: calls.append(args)
        before = copy.deepcopy(controller._rng.bit_generator.state)
        cert = certify_reference(controller, Trajectory(np.array([[.1, 0, 0],[.2, 0, 0]]), dt=.1),
                                 VehicleState(0, 0, 0), OpenSpace(), OpenSpace(), 3)
        self.assertIsNotNone(cert)
        self.assertEqual(len(cert['controls']), 2)
        self.assertEqual(calls, [])
        self.assertEqual(before, controller._rng.bit_generator.state)

    def test_budget_cannot_be_extended_to_reach_goal(self):
        cert = certify_reference(StraightController(), Trajectory(np.array([[.2, 0, 0]]), dt=.1),
                                 VehicleState(0, 0, 0), OpenSpace(), OpenSpace(), 1)
        self.assertIsNone(cert)

    def test_margin_intrusion_rejects_physical_safe_certificate(self):
        class BlockedMargin(OpenSpace):
            def swept_segment_free(self, start, end):
                return end[0] < .15
        cert = certify_reference(StraightController(), Trajectory(np.array([[.2, 0, 0]]), dt=.1),
                                 VehicleState(0, 0, 0), OpenSpace(), BlockedMargin(), 3)
        self.assertIsNone(cert)

    def test_vehicle_specific_tolerances_are_preserved(self):
        controller = StraightController();controller.terminal = TerminalChecker(.2, .08726646259971647)
        cert = certify_reference(controller, Trajectory(np.array([[.2, 0, 0]]), dt=.1),
                                 VehicleState(0, 0, 0), OpenSpace(), OpenSpace(), 3)
        self.assertEqual(cert['tol_pos'], .2)
        self.assertEqual(cert['tol_yaw'], controller.terminal.tol_yaw)

    def test_real_controller_rng_and_cached_state_unchanged(self):
        controller = TerminalCaptureMPC(contact_checker=OpenSpace(), dt=.1, horizon=10, seed=4)
        controller.bind_terminal(GoalPose(0, 0, 0), TerminalChecker(.2, .08726646259971647))
        before = copy.deepcopy(controller.__dict__)
        ref = Trajectory(np.zeros((4,3)),dt=.2)
        certify_reference(controller,ref,VehicleState(.1,0,.2),OpenSpace(),OpenSpace(),8)
        np.testing.assert_array_equal(controller._last_u, before['_last_u'])
        self.assertEqual(controller._rng.bit_generator.state,before['_rng'].bit_generator.state)
        self.assertIsNone(controller._progress_traj)
        self.assertIsNone(controller._aligned_cache)

    def source_fixture(self):
        from types import SimpleNamespace
        from verification.reference_retention import CertifiedReferenceSource
        from experiments.component_capture import VerificationBudget
        controller=StraightController()
        class Network:
            def begin(self, start, goal):self.calls=0
            def next_trajectory(self, state):
                self.calls+=1
                return Trajectory(np.array([[.1,0,0],[.2,0,0]]),dt=.1),0.
        engine=SimpleNamespace(mpc=controller,terminal=controller.terminal,max_steps=600,
                               collision_checker=OpenSpace(),margin_checker=OpenSpace(),vehicle_model=controller.vehicle_model)
        network=Network();source=CertifiedReferenceSource(network,engine,VerificationBudget(5))
        source.begin(VehicleState(0,0,0),controller.goal)
        # 测试资格半径不是控制器位置验收：当前.2m需允许落在3*tol之内。
        controller.terminal=engine.terminal=TerminalChecker(.08,.1)
        return source,controller,network

    def test_retention_still_queries_network_and_reuses_same_object(self):
        source,controller,network=self.source_fixture();state=VehicleState(0,0,0)
        first,_=source.next_trajectory(state)
        command=source.observe_command(controller.compute,first,state)
        state=controller.vehicle_model.step(state,command,.1)
        second,_=source.next_trajectory(state)
        self.assertIs(first,second)
        self.assertEqual(network.calls,2)
        self.assertTrue(source.queries[-1]['holding_previous'])

    def test_state_drift_is_rejected_before_real_control(self):
        source,controller,_=self.source_fixture();reference,_=source.next_trajectory(VehicleState(0,0,0))
        with self.assertRaisesRegex(RuntimeError,'前态'):
            source.observe_command(controller.compute,reference,VehicleState(.7,0,0))
        self.assertEqual(source.step,0)

    def test_retention_factory_is_restored_on_exception(self):
        from verification.reference_retention import retention_scope
        from verification import candidate_execution
        from experiments import validation_matrix as matrix
        from experiments.component_capture import VerificationBudget
        from sim import MINING_DRILL_RIG
        engine,controller=matrix.ClosedLoopEngine,candidate_execution.FootprintConstrainedMPC
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with retention_scope('unused',MINING_DRILL_RIG,{},VerificationBudget(1)):
                raise RuntimeError('fixture')
        self.assertIs(matrix.ClosedLoopEngine,engine)
        self.assertIs(candidate_execution.FootprintConstrainedMPC,controller)

    def test_no_new_arrival_cannot_open_formal_gate(self):
        from verification.retention_study import compare
        row={'dataset_index':1,'success':False,'collision':False,'zero_hold_steps':600,
             'intervened_steps':600,'planning_margin_violations':0}
        self.assertFalse(compare([row],[row])['retention_benefit'])


if __name__ == '__main__':
    unittest.main()
