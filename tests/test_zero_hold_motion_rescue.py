"""MPC-RESCUE: strict-cost, full-safety rescue without changing parent search."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
from interfaces import VehicleState, GoalPose, Trajectory
from sim import DifferentialDriveModel
from controller.constrained_mpc import ControlSafetyError
from controller.mpc import MPCController
from runtime.termination import TerminalChecker
from scripts.ungated_refinement_mpc import UngatedRefinedCEMController
from scripts.zero_hold_motion_rescue import ZeroHoldMotionRescueController
from scripts.zero_hold_motion_capture import audit_rescue_execution
from tests.test_feasible_cem_mpc import FreeSpace, LeftHalfSpace


class RescueTests(unittest.TestCase):
    def controllers(self, checker=None):
        kwargs=dict(footprint_checker=checker or FreeSpace(), goal=GoalPose(4.,0.,0.),terminal=TerminalChecker(),seed=11)
        return UngatedRefinedCEMController(**kwargs),ZeroHoldMotionRescueController(**kwargs)

    def test_ordinary_moving_output_and_rng_are_unchanged(self):
        old,new=self.controllers();state=VehicleState(0.,0.,0.);ref=np.tile([1.,0.,0.],(10,1))
        a=old._solve(np.zeros(20),ref,state);b=new._solve(np.zeros(20),ref,state)
        self.assertFalse(np.all(a==0.));np.testing.assert_array_equal(a,b)
        self.assertEqual(old._rng.bit_generator.state,new._rng.bit_generator.state)
        self.assertFalse(new.safety_audit()['rescue']['attempted'])

    def force_zero_parent(self, new, ref, state):
        # All stochastic candidates rejected, but the scalar oracle still allows
        # the independently evaluated fixed bank. This isolates missing coverage.
        new.batch_safety.eligible=lambda samples,state,oracle:np.zeros(len(samples),dtype=bool)
        return new._solve(np.zeros(20),ref,state)

    def test_zero_hold_is_replaced_only_by_safe_lower_original_cost(self):
        _,new=self.controllers();state=VehicleState(0.,0.,0.)
        selected=self.force_zero_parent(new,np.tile([2.,0.,0.],(10,1)),state)
        self.assertFalse(np.all(selected==0.));self.assertTrue(new.sequence_safe(selected,state))
        audit=new.safety_audit();self.assertTrue(audit['rescue']['intervened'])
        self.assertLess(audit['selected_cost'],audit['rescue']['base_cost'])
        self.assertEqual(audit['rescue']['candidate_count'],8)
        self.assertEqual(new.safety_stats()['rescue_candidates_checked'],8)
        self.assertEqual(new.safety_stats()['zero_hold'],0)

    def test_zero_at_reference_has_no_forced_motion(self):
        _,new=self.controllers();state=VehicleState(0.,0.,0.)
        selected=self.force_zero_parent(new,np.zeros((10,3)),state)
        np.testing.assert_array_equal(selected,np.zeros(20))
        self.assertFalse(new.safety_audit()['rescue']['intervened'])

    def test_low_cost_but_later_unsafe_controls_are_rejected(self):
        class Box(FreeSpace):
            def swept_segment_free(self,start,end):return abs(end[0])<.03
        _,new=self.controllers(Box());state=VehicleState(0.,0.,0.)
        self.assertTrue(new.footprint_checker.swept_segment_free([0.,0.,0.],[.02,0.,0.]))
        selected=self.force_zero_parent(new,np.tile([2.,0.,0.],(10,1)),state)
        np.testing.assert_array_equal(selected,np.zeros(20))
        self.assertFalse(any(new.safety_audit()['rescue']['safe_mask']))

    def test_terminal_rotation_is_preserved(self):
        _,new=self.controllers();new.goal=GoalPose(0.,0.,-.8)
        selected=new._solve(np.zeros(20),np.zeros((10,3)),VehicleState(.1,0.,0.))
        self.assertTrue(np.any(selected[1::2]!=0.));self.assertEqual(new.safety_audit()['selected'],'terminal_rotation')
        self.assertFalse(new.safety_audit()['rescue']['attempted'])

    def test_tie_keeps_zero_and_first_zero_does_not_intercept_a_future_plan(self):
        _,new=self.controllers();state=VehicleState(0.,0.,0.)
        with patch.object(MPCController,'_batch_cost',side_effect=lambda self,samples,ref,state:np.zeros(len(samples)),autospec=True):
            selected=self.force_zero_parent(new,np.zeros((10,3)),state)
        np.testing.assert_array_equal(selected,np.zeros(20));self.assertFalse(new.safety_audit()['rescue']['intervened'])
        warm=np.tile([.2,0.],10);warm[:2]=0.
        selected=new._solve(warm,np.tile([1.,0.,0.],(10,1)),state)
        np.testing.assert_array_equal(selected,warm);self.assertFalse(new.safety_audit()['rescue']['attempted'])

    def test_unsafe_initial_pose_still_stops_before_rescue(self):
        _,new=self.controllers(LeftHalfSpace())
        with self.assertRaises(ControlSafetyError):new._solve(np.zeros(20),np.zeros((10,3)),VehicleState(.1,0.,0.))

    def test_parent_replay_uses_actual_rescued_warm_start_and_rejects_forgery(self):
        class Tube(FreeSpace):
            def swept_segment_free(self,start,end):return abs(end[2])<=1e-12
        checker=Tube();model=DifferentialDriveModel();goal=GoalPose(4.,0.,0.);terminal=TerminalChecker()
        c=ZeroHoldMotionRescueController(footprint_checker=checker,vehicle_model=model,goal=goal,terminal=terminal,seed=0)
        trajectory=Trajectory(np.array([[0.,0.,0.],[2.,0.,0.]]),.1)
        state=VehicleState(0.,0.,0.);states=[];controls=[];windows=[];searches=[];frames=[]
        for _ in range(2):
            states.append(state.to_array());command=c.compute(trajectory,state)
            controls.append(c._last_u.reshape(10,2).copy());windows.append(c._reference_points(c._aligned_cache,c._progress_idx))
            searches.append(c.safety_audit());frames.append(dict(plan_ordinal=0,progress_index=c._progress_idx))
            state=model.step(state,command,.1)
        self.assertTrue(searches[0]['rescue']['intervened']);self.assertFalse(searches[1]['rescue']['attempted'])
        case=dict(index=0,goal=goal,planner=SimpleNamespace(_collision_checker=checker),
            vehicle=SimpleNamespace(mpc_kwargs=lambda:dict(max_v=2.,max_omega=1.)),engine_kwargs=dict(vehicle_model=model,terminal=terminal))
        detail=dict(profile='zero_rescue',searches=searches,frames=frames,controller_stats=c.safety_stats())
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'episode'
            np.savez_compressed(output.with_suffix('.npz'),states_before=np.array(states),control_windows=np.array(windows),
                selected_control_sequences=np.array(controls),global_plan_0000=trajectory.points,plan_dt_0000=np.array(.1))
            result=audit_rescue_execution(output,detail,case);self.assertEqual(result['rescue_interventions'],1)
            forged=deepcopy(detail);forged['searches'][0]['rescue']['safe_mask'][0]=False
            with self.assertRaisesRegex(ValueError,'fixed-bank'):audit_rescue_execution(output,forged,case)


if __name__=='__main__':unittest.main()
