from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.behavior_preservation import behavior_loss
from experiments.s6_terminal_objective import objective_loss, objective_slots, objective_qualification
from model.network import terminal_pose_loss, variable_loss_fn
from tests.test_s6_terminal_consumption import strict_schedule
from experiments.s6_terminal_consumption import terminal_schedule


def batch():
    schedule,_=terminal_schedule(strict_schedule(),(torch.arange(320)<19).float())
    return schedule[16]


class S6TerminalObjectiveTests(unittest.TestCase):
    def test_existing_stop_and_terminal_terms_keep_original_role_weight(self):
        samples=batch(); points=torch.ones(4,320,3,requires_grad=True); logits=torch.zeros(4,320,requires_grad=True)
        stack=lambda key:torch.stack([s[key] for s in samples])
        targets,stop,mask=stack('points'),stack('stop'),stack('mask')
        base=behavior_loss(points,logits,targets,stop,mask,[s['role'] for s in samples])
        old=.2*F.binary_cross_entropy_with_logits(logits[2],stop[2])
        new=variable_loss_fn(targets[2:3],logits[2:3],targets[2:3],mask[2:3],stop_weight=.2,
                            balance_stop=True,stop_target_mode='terminal',yaw_error_mode='wrapped')
        endpoint=terminal_pose_loss(points[2:3],targets[2:3],mask[2:3],yaw_weight=10,yaw_error_mode='wrapped')
        torch.testing.assert_close(objective_loss(points,logits,samples,'boundary_only'),base+(new-old)/4)
        torch.testing.assert_close(objective_loss(points,logits,samples,'boundary_terminal'),base+(new-old+endpoint)/4)

    def test_after_terminal_logits_have_no_new_gradient_and_others_unchanged(self):
        samples=batch(); points=torch.zeros(4,320,3,requires_grad=True); logits=torch.zeros(4,320,requires_grad=True)
        loss=objective_loss(points,logits,samples,'boundary_terminal')
        loss.backward(); self.assertTrue(torch.equal(logits.grad[2,19:],torch.zeros(301)))
        self.assertLess(float(logits.grad[2,18]),0); self.assertGreater(float(logits.grad[2,0]),0)
        other_points=points.detach().clone().requires_grad_(); other_logits=logits.detach().clone().requires_grad_()
        targets=torch.stack([s['points'] for s in samples]); stop=torch.stack([s['stop'] for s in samples]); mask=torch.stack([s['mask'] for s in samples])
        original=behavior_loss(other_points,other_logits,targets,stop,mask,[s['role'] for s in samples]); original.backward()
        for index in (0,1,3):
            torch.testing.assert_close(points.grad[index],other_points.grad[index])
            torch.testing.assert_close(logits.grad[index],other_logits.grad[index])

    def test_nonterminal_batches_are_exactly_original_and_sparse_query_rejected(self):
        samples=batch(); samples[2]=deepcopy(samples[3])
        points=torch.randn(4,320,3); logits=torch.randn(4,320)
        expected=behavior_loss(points,logits,torch.stack([s['points'] for s in samples]),
            torch.stack([s['stop'] for s in samples]),torch.stack([s['mask'] for s in samples]),[s['role'] for s in samples])
        self.assertTrue(torch.equal(objective_loss(points,logits,samples,'boundary_only'),expected))
        samples=batch(); samples[2]['mask'][18]=0
        with self.assertRaises(ValueError): objective_loss(points,logits,samples,'boundary_terminal')
        with self.assertRaises(ValueError): objective_loss(points,logits,batch(),'unknown')

    def test_objective_applies_only_at_predeclared_actual_query(self):
        plan,_=terminal_schedule(strict_schedule(),(torch.arange(320)<19).float())
        self.assertEqual(objective_slots(plan),[[16,2]])
        plan[0][2]=plan[16][2]
        with self.assertRaises(ValueError): objective_slots(plan)

    def test_precheck_does_not_train_and_existing_output_is_rejected(self):
        from scripts.preflight_s6_terminal_objective import precheck,run
        with patch('scripts.preflight_s6_terminal_objective.prepare_context',return_value={}), \
             patch('scripts.preflight_s6_terminal_objective.fit_point') as fit:
            precheck(); fit.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_terminal_objective.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_gate_requires_original_gate_both_orders_and_preserves_old_success(self):
        def row(index,success,collision=False):
            return dict(dataset_index=index,task_id=str(index),scene_name='S6_scene',success=success,collision=collision)
        old={s:[row(1,True),row(2,False,True),row(3,False)] for s in (20261020,20261021)}
        new=deepcopy(old); new[20261020][1]=row(2,True)
        self.assertTrue(objective_qualification({'passed':True},old,new)['passed'])
        self.assertFalse(objective_qualification({'passed':False},old,new)['passed'])
        self.assertFalse(objective_qualification({'passed':True},old,old)['passed'])
        self.assertFalse(objective_qualification({'passed':True},old,{20261020:new[20261020]})['passed'])
        lost=deepcopy(new); lost[20261021][0]=row(1,False); lost[20261021][2]=row(3,True)
        self.assertFalse(objective_qualification({'passed':True},old,lost)['passed'])
        collision=deepcopy(new); collision[20261021][2]['collision']=True
        self.assertFalse(objective_qualification({'passed':True},old,collision)['passed'])

    def test_budget_failure_is_saved_without_fit_or_retry(self):
        import json
        from scripts.preflight_s6_terminal_objective import run
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_terminal_objective.prepare_context',side_effect=TimeoutError('limit')) as prepare, \
                 patch('scripts.preflight_s6_terminal_objective.fit_point') as fit:
                with self.assertRaises(TimeoutError): run(Path(folder)/'new')
                prepare.assert_called_once(); fit.assert_not_called()
            report=json.loads((Path(folder)/'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertFalse(report['allowed_to_prepare_training_plan'])

    def test_label_drift_is_rejected_before_any_forward_or_gradient(self):
        from experiments.executable_behavior import PreflightBudget
        from experiments.s6_recovery_learning import batch_identity
        from scripts.preflight_s6_terminal_objective import fit_point
        plan,_=terminal_schedule(strict_schedule(),(torch.arange(320)<19).float())
        identities=[batch_identity(samples) for samples in plan]
        plan[0][0]=deepcopy(plan[0][0]); plan[0][0]['points'][0,0]+=1
        model=torch.nn.Linear(1,1); optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
        model(torch.ones(1,1)).sum().backward(); optimizer.step()
        for state in optimizer.state.values(): state['step']=torch.tensor(240.)
        context=dict(config=dict(objective=dict(slots=[[16,2]]),parent_checkpoints={'20261020':'unused'},
                     planned_batch_identities={'boundary_only':{20261020:identities}}),
                     plans={'boundary_only':{20261020:plan}},parent=None)
        with patch('scripts.preflight_s6_terminal_objective.load_candidate',return_value=(model,optimizer.state_dict())):
            with self.assertRaisesRegex(ValueError,'身份漂移'):
                fit_point(context,'boundary_only',20261020,PreflightBudget(1),Path('.'))


if __name__=='__main__': unittest.main()
