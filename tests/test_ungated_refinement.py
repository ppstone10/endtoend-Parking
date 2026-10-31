"""REF-UNGATE: restore original progress without changing the frozen solve."""
import unittest
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from controller.mpc import MPCController
from interfaces import VehicleState, Trajectory, GoalPose
from runtime.termination import TerminalChecker
from scripts.runtime_refinement_mpc import RefinedCEMController
from scripts.ungated_refinement_mpc import UngatedRefinedCEMController
from tests.test_feasible_cem_mpc import FreeSpace
from scripts.ungated_refinement_capture import audit_ungated_execution
from sim import DifferentialDriveModel


class UngatedRefinementTests(unittest.TestCase):
    def controllers(self):
        kwargs = dict(footprint_checker=FreeSpace(), goal=GoalPose(3., 0., 0.),
                      terminal=TerminalChecker(), horizon=3, population=8, iterations=2, seed=3)
        return RefinedCEMController(profile='combined', **kwargs), UngatedRefinedCEMController(**kwargs)

    def test_options_differ_only_in_progress(self):
        old, new = self.controllers()
        self.assertEqual(new.options, {**old.options, 'progress': False})

    def test_behind_pose_advances_exactly_like_original_clock(self):
        gated, new = self.controllers();original = MPCController()
        points = np.column_stack((np.arange(20)*.1, np.zeros((20, 2))))
        state = VehicleState(0., 0., 0.)
        gated._progress_idx = new._progress_idx = original._progress_idx = 10
        self.assertEqual(gated._advance_progress(points, state), 10)
        for _ in range(5):
            expected = original._advance_progress(points, state)
            self.assertEqual(new._advance_progress(points, state), expected)
            np.testing.assert_array_equal(new._reference_points(points, expected), original._reference_points(points, expected)[:3])

    def test_same_solve_input_preserves_controls_generations_and_rng(self):
        old, new = self.controllers()
        state = VehicleState(0., 0., 0.);ref = np.array([[.1,0.,0.],[.2,0.,0.],[.3,0.,0.]])
        np.testing.assert_array_equal(old._solve(np.zeros(6),ref,state), new._solve(np.zeros(6),ref,state))
        self.assertEqual(old._rng.bit_generator.state,new._rng.bit_generator.state)
        for key in ['generations','candidate_costs','terminal_rotation','selected_sequence']:
            self.assertEqual(old.safety_audit()[key],new.safety_audit()[key])

    def test_new_prediction_reinitializes_progress_without_resetting_search(self):
        _, new = self.controllers()
        state = VehicleState(0.,0.,0.);trajectory = Trajectory(np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.]]),.1)
        new.compute(trajectory,state)
        new.compute(trajectory,state)
        self.assertGreater(new._progress_idx,0)
        checks=new.safety_stats()['checks']
        new.compute(Trajectory(trajectory.points.copy(),.1),state)
        self.assertEqual(new._progress_idx,0)
        self.assertEqual(new.safety_stats()['checks'],checks+1)

    def test_independent_progress_and_parent_replay_rejects_forged_evidence(self):
        checker=FreeSpace();model=DifferentialDriveModel();goal=GoalPose(3.,0.,0.);terminal=TerminalChecker()
        c=UngatedRefinedCEMController(footprint_checker=checker,vehicle_model=model,goal=goal,terminal=terminal,seed=0)
        trajectory=Trajectory(np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.]]),.1)
        state=VehicleState(0.,0.,0.);c.compute(trajectory,state)
        case=dict(index=0,goal=goal,planner=SimpleNamespace(_collision_checker=checker),
            vehicle=SimpleNamespace(mpc_kwargs=lambda:dict(max_v=2.,max_omega=1.)),
            engine_kwargs=dict(vehicle_model=model,terminal=terminal))
        detail=dict(profile='ungated',searches=[c.safety_audit()],controller_stats=c.safety_stats(),
                    frames=[dict(plan_ordinal=0,progress_index=0)])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'episode'
            np.savez_compressed(output.with_suffix('.npz'),states_before=state.to_array()[None],
                control_windows=c._reference_points(c._aligned_cache,0)[None],
                selected_control_sequences=c._last_u.reshape(1,10,2),
                global_plan_0000=trajectory.points,plan_dt_0000=np.array(.1))
            self.assertEqual(audit_ungated_execution(output,detail,case)['frozen_solve_replays_exact'],1)
            forged=deepcopy(detail);forged['frames'][0]['progress_index']=1
            with self.assertRaisesRegex(ValueError,'progress changed'):audit_ungated_execution(output,forged,case)
            forged=deepcopy(detail);forged['searches'][0]['selected_cost']+=1.
            with self.assertRaisesRegex(ValueError,'parent solve changed'):audit_ungated_execution(output,forged,case)


if __name__=='__main__':unittest.main()
