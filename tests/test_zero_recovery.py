import unittest
from unittest.mock import patch
import numpy as np

from controller import MPCController
from interfaces import VehicleState,GoalPose
from runtime.termination import TerminalChecker
from verification.terminal_capture import TerminalCaptureMPC
from verification.zero_recovery import ZeroRecoveryMPC


class OpenSpace:
    def pose_free(self,*args):return True
    def swept_segment_free(self,*args):return True


class ZeroRecoveryTests(unittest.TestCase):
    def make(self):
        controller=ZeroRecoveryMPC(contact_checker=OpenSpace(),horizon=2,seed=11)
        controller.bind_terminal(GoalPose(10,0,0),TerminalChecker())
        return controller

    def test_nonzero_original_output_and_rng_are_unchanged(self):
        old=TerminalCaptureMPC(contact_checker=OpenSpace(),horizon=2,seed=11)
        old.bind_terminal(GoalPose(10,0,0),TerminalChecker())
        new=self.make();state=VehicleState(0,0,0);mean=np.zeros(4);ref=np.ones((2,3))
        np.testing.assert_array_equal(old._solve(mean,ref,state),new._solve(mean,ref,state))
        self.assertEqual(old._rng.random(),new._rng.random())

    def test_lower_cost_safe_sample_outside_first16_is_used(self):
        controller=self.make();samples=np.tile([1.,1.,1.,1.],(20,1));samples[18]=[.2,0,.2,0]
        def solve(instance,mean,ref,state):
            instance._recovery_bank.append((samples.copy(),np.arange(20,dtype=float)))
            instance.last_safety_audit={'intervened':True,'selected':'zero_hold','selected_cost':25.}
            return np.zeros(4)
        with patch.object(TerminalCaptureMPC,'_solve',solve),patch.object(controller,'sequence_safe',side_effect=lambda s,state: np.array_equal(s,samples[18])):
            selected=controller._solve(np.zeros(4),np.zeros((2,3)),VehicleState(0,0,0))
        np.testing.assert_array_equal(selected,samples[18])
        self.assertEqual(controller.last_safety_audit['zero_recovery_rank'],19)

    def test_no_cheaper_safe_sample_preserves_original_zero(self):
        controller=self.make()
        def solve(instance,mean,ref,state):
            instance._recovery_bank.append((np.ones((20,4)),np.arange(20,dtype=float)))
            instance.last_safety_audit={'intervened':True,'selected':'zero_hold','selected_cost':1.}
            return np.zeros(4)
        with patch.object(TerminalCaptureMPC,'_solve',solve):
            selected=controller._solve(np.zeros(4),np.zeros((2,3)),VehicleState(0,0,0))
        np.testing.assert_array_equal(selected,np.zeros(4))

    def test_factory_restored_on_exception(self):
        from verification import terminal_capture
        from verification.zero_recovery import zero_recovery_scope
        from experiments.component_capture import VerificationBudget
        from sim import MINING_DRILL_RIG
        original=terminal_capture.TerminalCaptureMPC
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with zero_recovery_scope('unused',MINING_DRILL_RIG,{},VerificationBudget(1)):
                raise RuntimeError('fixture')
        self.assertIs(terminal_capture.TerminalCaptureMPC,original)


if __name__=='__main__':unittest.main()
