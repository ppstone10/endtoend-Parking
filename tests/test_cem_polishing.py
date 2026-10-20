from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from controller import MPCController
from experiments.cem_polishing import CEMPolishedMPC,torch_original_cost
from interfaces import VehicleState,Trajectory
from experiments.polishing_execution import run_polished_branch,summarize


class PolishingTests(unittest.TestCase):
    def test_original_cost_and_gradient(self):
        mpc=MPCController(horizon=3);state=VehicleState(.2,-.1,3.13)
        ref=np.array([[.1,.2,-3.1],[.2,.2,-3.0],[.3,.2,-2.9]])
        controls=np.array([.1,.2,.2,-.1,.1,.3]);u=torch.tensor(controls,dtype=torch.float64,requires_grad=True)
        loss=torch_original_cost(mpc,u,ref,state);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),float(mpc._batch_cost(controls[None,:],ref,state)[0]),places=10)
        for j in range(6):
            left,right=controls.copy(),controls.copy();left[j]-=1e-5;right[j]+=1e-5
            derivative=(mpc._batch_cost(right[None,:],ref,state)[0]-mpc._batch_cost(left[None,:],ref,state)[0])/2e-5
            self.assertAlmostEqual(float(u.grad[j]),float(derivative),places=6)

    def test_disabled_matches_original_actions_and_rng(self):
        base=MPCController(horizon=3,population=8,iterations=2,seed=7)
        trial=CEMPolishedMPC.from_controller(base,max_evaluations=0)
        trajectory=Trajectory(np.array([[0,0,0],[.1,0,.1],[.2,.1,.2]]),.1)
        for _ in range(3):self.assertEqual(base.compute(trajectory,VehicleState(0,0,0)),trial.compute(trajectory,VehicleState(0,0,0)))
        self.assertEqual(base._rng.bit_generator.state,trial._rng.bit_generator.state)

    def test_hard_budget_bounds_nonworsening_and_copy(self):
        base=MPCController(horizon=3,population=8,iterations=2,seed=7)
        trial=CEMPolishedMPC.from_controller(base,max_evaluations=3)
        rng=deepcopy(base._rng.bit_generator.state)
        trajectory=Trajectory(np.array([[0,0,0],[.1,0,.1],[.2,.1,.2]]),.1)
        trial.compute(trajectory,VehicleState(0,0,0));row=trial.polish_audits[-1]
        self.assertLessEqual(row['evaluations'],3);self.assertLessEqual(row['selected_cost'],row['original_cost'])
        self.assertTrue(np.isfinite(trial._last_u).all());self.assertLessEqual(abs(trial._last_u[0::2]).max(),base.max_v)
        self.assertLessEqual(abs(trial._last_u[1::2]).max(),base.max_omega)
        self.assertEqual(base._rng.bit_generator.state,rng)
        base.compute(trajectory,VehicleState(0,0,0));self.assertEqual(base._rng.bit_generator.state,trial._rng.bit_generator.state)

    def test_zero_cost_tie_keeps_original_and_nonfinite_falls_back(self):
        mpc=CEMPolishedMPC(horizon=3,population=8,iterations=1,max_evaluations=5)
        original=np.zeros(6);ref=np.zeros((3,3));state=VehicleState(0,0,0)
        with patch.object(MPCController,'_solve',return_value=original):
            actual=mpc._solve(original,ref,state);np.testing.assert_array_equal(actual,original)
            self.assertFalse(mpc.polish_audits[-1]['accepted'])
            with patch('experiments.cem_polishing.torch_original_cost',return_value=torch.tensor(float('nan'),requires_grad=True)):
                actual=mpc._solve(original,ref,state)
                np.testing.assert_array_equal(actual,original);self.assertEqual(mpc.polish_audits[-1]['stop_reason'],'nonfinite')

    def test_unexpected_failure_propagates(self):
        mpc=CEMPolishedMPC(horizon=3)
        with patch('experiments.cem_polishing.torch_original_cost',side_effect=RuntimeError('unexpected')):
            with self.assertRaisesRegex(RuntimeError,'unexpected'):
                mpc.polish_controls(np.zeros(6),np.zeros((3,3)),VehicleState(0,0,0))

    def test_runner_keeps_context_and_audits_actual_first_control(self):
        from tests.test_recovery_execution_probe import RecoveryExecutionTests,TinyModel,TinyPlanner
        from interfaces import GoalPose
        engine,_,kwargs=RecoveryExecutionTests().collect();context=engine.contexts[1]
        saved=deepcopy(context.mpc.__dict__);rng=deepcopy(context.pipeline.rng.bit_generator.state)
        for mode in ('network','expert_continuous'):
            result,margins,audit,details=run_polished_branch(context,mode,TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=2,replan_every=10)
            self.assertEqual(result.steps,2);self.assertEqual(len(audit),2)
            self.assertEqual(margins,[False,False]);self.assertLessEqual(details['polish_evaluations'],160)
            self.assertTrue(all(a['polish']['selected_cost']<=a['polish']['original_cost'] for a in audit))
        np.testing.assert_array_equal(context.mpc._last_u,saved['_last_u'])
        self.assertEqual(context.mpc._rng.bit_generator.state,saved['_rng'].bit_generator.state)
        self.assertEqual(context.pipeline.rng.bit_generator.state,rng)

    def test_gate_requires_six_strict_teacher_and_both_network_controls(self):
        rows=[]
        for i in range(6):
            branches={name:{'success':i>=4,'collision':i<4,'reached_with_margin':False}
                      for name in ('original_network','original_expert_continuous','polished_network','polished_expert_continuous')}
            branches['polished_expert_continuous']={'success':True,'collision':False,'reached_with_margin':True}
            rows.append({'role':'success_control' if i>=4 else 'before_first_margin_loss','branches':branches})
        self.assertTrue(summarize(rows)['gate_passed']);self.assertFalse(summarize(rows,smoke=True)['gate_passed'])
        bad=deepcopy(rows);bad[0]['branches']['polished_expert_continuous']['reached_with_margin']=False
        self.assertFalse(summarize(bad)['gate_passed'])
        bad=deepcopy(rows);bad[4]['branches']['polished_network']['success']=False
        self.assertFalse(summarize(bad)['gate_passed'])

    def test_cli_rejects_identity_drift_and_output_reuse(self):
        import json,tempfile
        from pathlib import Path
        from scripts.compare_cem_polishing import load_config,run
        from scripts.prepare_recovery_balance_ab import file_digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.write_bytes(b'frozen');path=root/'config.json'
            config={'kind':'cem_polishing_six_source_ab','code':{},'runtime':{},'inputs':{'x':{'path':str(source),'sha256':file_digest(source)}},
                    'max_polish_iterations':60,'max_evaluations':80,'output':str(root/'results')}
            path.write_text(json.dumps(config),encoding='utf-8');sha=file_digest(path)
            with patch('scripts.compare_cem_polishing.execution_identity',return_value={}),patch('scripts.compare_cem_polishing.runtime_identity',return_value={}):
                self.assertEqual(load_config(path,sha),config)
                with self.assertRaises(ValueError):load_config(path,'bad')
                (root/'results').mkdir()
                with self.assertRaises(FileExistsError):run(path,sha)
                source.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError,'输入身份'):load_config(path,sha)


if __name__=='__main__':unittest.main()
