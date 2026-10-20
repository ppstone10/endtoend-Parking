"""离线求解器保留候选的局部性质，不替代闭环验收。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller import MPCController
from experiments.mpc_candidate_solver import CandidatePreservingMPC
from interfaces import Trajectory, VehicleState
from scripts import compare_mpc_solvers as comparison


class ExperimentalMPCTests(unittest.TestCase):
    def test_static_reference_has_zero_cost_and_command(self):
        mpc = CandidatePreservingMPC(seed=13)
        state = VehicleState(2., -.1, .08)
        ref = np.tile(state.to_array()[:3], (mpc.horizon, 1))
        u = mpc._solve(np.full(2*mpc.horizon, .4), ref, state)
        np.testing.assert_array_equal(u, np.zeros_like(u))
        self.assertEqual(mpc.last_candidate_audit['selected_cost'], 0.)

    def test_same_search_rng_cost_dominance_and_bounds(self):
        for seed in range(8):
            original = MPCController(seed=seed)
            trial = CandidatePreservingMPC(seed=seed)
            state = VehicleState(0., .1, .06)
            ref = np.column_stack((np.linspace(0., .9, 10), np.zeros(10), np.linspace(0., .2, 10)))
            warm = np.tile([.3, -.1], 10)
            before = warm.copy()
            baseline = original._solve(warm, ref, state)
            selected = trial._solve(warm, ref, state)
            costs = original._batch_cost(np.stack([selected, baseline, warm, np.zeros(20)]), ref, state)
            self.assertTrue(np.all(costs[0] <= costs[1:]+1e-12))
            self.assertEqual(original._rng.bit_generator.state, trial._rng.bit_generator.state)
            np.testing.assert_array_equal(warm, before)
            self.assertTrue(np.all(abs(selected[::2]) <= original.max_v))
            self.assertTrue(np.all(abs(selected[1::2]) <= original.max_omega))

    def test_straight_compute_moves_forward_and_reset(self):
        mpc = CandidatePreservingMPC(seed=0)
        traj = Trajectory(np.column_stack((np.linspace(0., 5., 100), np.zeros((100, 2)))), .1)
        state = VehicleState(0., 0., 0.)
        cmd = mpc.compute(traj, state)
        self.assertGreater(cmd.v, 0.)
        self.assertLess(abs(cmd.omega), .15)
        clone = deepcopy(mpc)
        mpc.reset()
        self.assertIsNone(mpc._progress_idx)
        self.assertIsNotNone(clone._progress_idx)

    def test_invalid_input_rejected(self):
        mpc = CandidatePreservingMPC()
        with self.assertRaises(ValueError):
            mpc._solve(np.full(20, np.nan), np.zeros((10, 3)), VehicleState(0., 0., 0.))

    def test_ties_prefer_legacy(self):
        mpc = CandidatePreservingMPC(seed=5, w_xy=0., w_yaw=0., w_terminal=0., w_v=0., w_omega=0.)
        original = MPCController(seed=5, w_xy=0., w_yaw=0., w_terminal=0., w_v=0., w_omega=0.)
        state = VehicleState(0., 0., 0.)
        expected = original._solve(np.zeros(20), np.zeros((10, 3)), state)
        actual = mpc._solve(np.zeros(20), np.zeros((10, 3)), state)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(mpc.last_candidate_audit['selected'], 'legacy_mean')

    def test_nonfinite_cost_rejected(self):
        mpc = CandidatePreservingMPC()
        with patch.object(mpc, '_batch_cost', return_value=np.full(mpc.population, np.nan)):
            with self.assertRaises(ValueError):
                mpc._solve(np.zeros(20), np.zeros((10,3)), VehicleState(0.,0.,0.))


class SolverComparisonTests(unittest.TestCase):
    @staticmethod
    def report(successes=(True,True,False), collisions=(False,False,True)):
        return {'status':'completed','experiment':{'name':'E5','safety_mode':'none'},'reconstruct_failures':[],
                'episodes':[{'success':s,'collision':c,'failure':None if s else ('collision' if c else 'timeout'),
                             'steps':30,'final_pos_err':.1,'final_yaw_err':.01,
                             'meta':{'dataset_index':i,'task_id':f'task-{i}','scene_name':'S4','tol_pos':.2,'tol_yaw':.1}}
                            for i,(s,c) in enumerate(zip(successes,collisions))]}

    def test_scope_restores_on_exception_and_default_is_original(self):
        previous = comparison.matrix.MPCController
        self.assertIs(previous,MPCController)
        with self.assertRaises(RuntimeError):
            with comparison.solver_scope('candidate'):
                self.assertIs(comparison.matrix.MPCController,CandidatePreservingMPC)
                raise RuntimeError('test')
        self.assertIs(comparison.matrix.MPCController,previous)
        with self.assertRaises(ValueError):
            with comparison.solver_scope('typo'):
                pass

    def test_pair_gate_rejects_success_loss_and_identity_mismatch(self):
        original = self.report()
        candidate = self.report((False,True,True),(False,False,False))
        gates = {'min_successes':2,'max_collisions':0,'min_retained_successes':2,'max_scene_success_losses':0}
        summary = comparison.paired_summary(original,candidate,[0,1,2],gates)
        self.assertFalse(summary['gate_passed'])
        self.assertEqual(summary['retained_original_successes'],1)
        candidate['episodes'][0]['meta']['task_id']='wrong'
        with self.assertRaises(ValueError):
            comparison.paired_summary(original,candidate,[0,1,2],gates)

    def test_baseline_mismatch_and_partial_report_rejected(self):
        baseline = self.report()
        comparison.verify_baseline(baseline,deepcopy(baseline),[0,1,2])
        changed = deepcopy(baseline)
        changed['episodes'][1]['steps']+=1
        with self.assertRaises(ValueError):
            comparison.verify_baseline(changed,baseline,[0,1,2])
        changed = deepcopy(baseline)
        changed['episodes'].pop()
        with self.assertRaises(ValueError):
            comparison.episode_map(changed,[0,1,2])

    def test_frozen_config_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input.txt'
            source.write_text('input',encoding='utf-8')
            config = {'kind':'offline_mpc_candidate_comparison','runtime':comparison.runtime_identity(),
                      'code':comparison.execution_identity(),'inputs':{'one':{'path':str(source),'sha256':comparison.file_digest(source)}}}
            path = root/'config.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            digest = comparison.file_digest(path)
            comparison.load_config(path,digest)
            (root/'results').mkdir()
            with self.assertRaises(FileExistsError):
                comparison.run(path,digest)
            source.write_text('tampered',encoding='utf-8')
            with self.assertRaises(ValueError):
                comparison.load_config(path,digest)

    def test_smoke_never_passes_formal_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            config={'smoke_indices':[0,1,2],'arms':['original','candidate'],'inputs':{'data':{'path':'unused'},'model':{'path':'unused'}},
                    'smoke_max_steps':60,'replan_every':10,'seed':0,'gates':{'min_successes':0,'max_collisions':3,'min_retained_successes':0,'max_scene_success_losses':3}}
            with patch.object(comparison,'load_config',return_value=config),patch.object(comparison.matrix,'run_validation_experiment',return_value=self.report()):
                comparison.run(path,'fake',smoke=True)
            report=json.loads((path.parent/'preflight/report.json').read_text(encoding='utf-8'))
            self.assertFalse(report['summary']['gate_passed'])
            self.assertEqual(report['summary']['decision'],'bounded_preflight_not_formal_evidence')


if __name__ == '__main__':
    unittest.main()
