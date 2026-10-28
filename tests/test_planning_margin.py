import unittest
from unittest.mock import patch
import numpy as np
from controller import MPCController
from interfaces import VehicleState,GoalPose
from planner import RectangleFootprintCollisionChecker
from runtime.termination import TerminalChecker
from sim import ParkingEnvironment,RectangleObstacle
from verification.terminal_capture import TerminalCaptureMPC
from verification.planning_margin import PlanningMarginMPC


class PlanningMarginTests(unittest.TestCase):
    def checker(self):
        env=ParkingEnvironment(world_size=30,obstacles=[RectangleObstacle(4.1,5.,-5.,5.)])
        return RectangleFootprintCollisionChecker(env,vehicle_length=6.,vehicle_width=3.,collision_margin=0.,resolution=.05)

    def controller(self,cls,goal):
        c=cls(contact_checker=self.checker(),dt=.1,horizon=10,seed=0)
        c.bind_terminal(goal,TerminalChecker(.3,np.pi/18));return c

    def test_legacy_mean_safe_physically_but_margin_unsafe_is_filtered(self):
        state=VehicleState(.5,0,0);old=self.controller(TerminalCaptureMPC,GoalPose(-5,0,0));new=self.controller(PlanningMarginMPC,GoalPose(-5,0,0))
        sequence=np.tile([.5,0.],10)
        self.assertTrue(old.sequence_safe(sequence,state));self.assertFalse(new.sequence_safe(sequence,state))
        with patch.object(MPCController,'_solve',return_value=sequence):
            a=old._solve(np.zeros(20),np.zeros((10,3)),state)
            b=new._solve(np.zeros(20),np.zeros((10,3)),state)
        np.testing.assert_array_equal(a,sequence);np.testing.assert_array_equal(b,np.zeros(20))
        self.assertTrue(new.last_safety_audit['intervened'])

    def test_terminal_full_rotation_uses_same_margin(self):
        state=VehicleState(.5,0,0);goal=GoalPose(.5,0,np.pi/2)
        old=self.controller(TerminalCaptureMPC,goal);new=self.controller(PlanningMarginMPC,goal)
        with patch.object(MPCController,'_solve',return_value=np.zeros(20)):
            old._solve(np.zeros(20),np.zeros((10,3)),state)
            result=new._solve(np.zeros(20),np.zeros((10,3)),state)
        self.assertEqual(old.last_safety_audit['selected'],'terminal_rotation')
        self.assertNotEqual(new.last_safety_audit['selected'],'terminal_rotation')
        np.testing.assert_array_equal(result,np.zeros(20))

    def test_initial_margin_violation_fails_instead_of_claiming_recovery(self):
        c=self.controller(PlanningMarginMPC,GoalPose(-5,0,0));state=VehicleState(1,0,0)
        self.assertTrue(c.physical_checker.pose_free(state.x,state.y,state.yaw))
        with self.assertRaises(ValueError):c._solve(np.zeros(20),np.zeros((10,3)),state)

    def test_factory_restores_after_exception(self):
        from verification import terminal_capture
        from verification.planning_margin import planning_margin_scope
        from experiments.component_capture import VerificationBudget
        from sim import MINING_DRILL_RIG
        original=terminal_capture.TerminalCaptureMPC
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with planning_margin_scope('unused',MINING_DRILL_RIG,{},VerificationBudget(1)):raise RuntimeError('fixture')
        self.assertIs(terminal_capture.TerminalCaptureMPC,original)


if __name__=='__main__':unittest.main()
