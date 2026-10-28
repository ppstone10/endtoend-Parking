import unittest

import numpy as np

from experiments.reference_compensation import compensate_reference_yaw,reference_internal_safe
from interfaces import Trajectory


class ReferenceCompensationTests(unittest.TestCase):
    def test_negative_feedback_bounds_and_endpoint_invariance(self):
        original=Trajectory(np.c_[np.arange(12),np.zeros((12,2))],.2)
        before=original.points.copy()
        states=np.zeros((10,5));states[:,2]=.3
        trace=np.c_[np.arange(1,11),np.full(10,12)]
        current=original
        for _ in range(2):current=compensate_reference_yaw(original,current,states,trace)
        np.testing.assert_array_equal(original.points,before)
        np.testing.assert_array_equal(current.points[:,:2],before[:,:2])
        np.testing.assert_array_equal(current.points[[0,-1]],before[[0,-1]])
        np.testing.assert_allclose(current.points[1:-1,2],-.1,atol=1e-7)
        self.assertEqual(current.dt,.2)

    def test_wraparound_and_unvisited_prefix(self):
        original=Trajectory(np.c_[np.arange(12),np.zeros(12),np.full(12,np.pi-.01)],.2)
        states=np.zeros((3,5));states[:,2]=-np.pi+.01
        trace=np.array([[6,12],[7,12],[8,12]])
        out=compensate_reference_yaw(original,original,states,trace)
        np.testing.assert_array_equal(out.points[:6],original.points[:6])
        delta=np.arctan2(np.sin(out.points[6:9,2]-original.points[6:9,2]),np.cos(out.points[6:9,2]-original.points[6:9,2]))
        np.testing.assert_allclose(delta,-.01,atol=1e-6)

    def test_invalid_or_misaligned_trace_rejected(self):
        ref=Trajectory(np.zeros((3,3)),.2)
        for trace in ([],[[3,3]],[[0,0]],[[.5,3]],[[np.nan,3]]):
            with self.assertRaises(ValueError):compensate_reference_yaw(ref,ref,np.zeros((1,5)),trace)

    def test_internal_sweep_not_only_discrete_poses(self):
        class Checker:
            def pose_free(self,*pose):return True
            def swept_segment_free(self,a,b):return False
        self.assertFalse(reference_internal_safe(Trajectory(np.zeros((2,3)),.2),Checker()))

    def test_rolling_teacher_records_each_applied_reference_without_changing_it(self):
        from types import SimpleNamespace
        from experiments.executable_behavior import RecordedReplanningExpert,RecordingMPC,PreflightBudget
        from interfaces import VehicleState,GoalPose
        class Pipeline:
            def set_target_goals(self,goals):pass
            def capture_bev(self,*pose):return SimpleNamespace(data=np.zeros((5,2,2),dtype=np.float32))
        class Planner:
            def plan(self,state,goal):
                return Trajectory(np.array([[state.x,0.,0.],[state.x+.2,0.,0.],[state.x+.4,0.,0.]]),.2)
        budget=PreflightBudget();mpc=RecordingMPC(seed=2,budget=budget)
        source=RecordedReplanningExpert(Pipeline(),Planner(),mpc,budget)
        start=VehicleState(0.,0.,0.);goal=GoalPose(2.,0.,0.)
        source.begin(start,goal)
        first,_=source.next_trajectory(start);mpc.compute(first,start)
        second,_=source.next_trajectory(start);mpc.compute(second,start)
        self.assertIsNot(first,second)
        self.assertEqual(len(source.frames),2)
        np.testing.assert_array_equal(first.points,second.points)
        for frame in source.frames:
            self.assertTrue(frame['mask'].any())
            self.assertEqual(frame['length'],3)

    def test_combination_flag_cannot_silently_enable_refinement(self):
        from scripts.preflight_behavior_repair import run
        with self.assertRaises(ValueError):run('not-created',rolling_s4=True)


if __name__=='__main__':unittest.main()
