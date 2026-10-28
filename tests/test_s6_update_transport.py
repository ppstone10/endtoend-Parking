from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.s6_update_transport import gradient_list, vector_dot, adam_counterfactual, state_equal


class UpdateTransportTests(unittest.TestCase):
    def test_unused_gradients_and_jacobian_projection(self):
        model=torch.nn.Linear(2,1)
        loss=model.weight.square().sum()
        grads=gradient_list(loss,list(model.parameters()))
        self.assertTrue(torch.equal(grads[1],torch.zeros_like(model.bias)))
        self.assertAlmostEqual(vector_dot(grads,grads),sum(float(g.double().square().sum()) for g in grads))

    def test_counterfactuals_do_not_change_original_and_actual_matches_adam(self):
        model=torch.nn.Linear(1,1); opt=torch.optim.Adam(model.parameters(),lr=1e-5)
        model(torch.ones(1,1)).sum().backward(); opt.step()
        initial=deepcopy(model.state_dict()); state=deepcopy(opt.state_dict())
        grads=[torch.full_like(p,.1) for p in model.parameters()]
        actual,after=adam_counterfactual(model,state,grads,'actual')
        memory,_=adam_counterfactual(model,state,grads,'zero_current')
        reset,_=adam_counterfactual(model,state,grads,'zero_first_moment')
        self.assertTrue(state_equal(initial,model.state_dict()))
        self.assertTrue(state_equal(state,opt.state_dict()))
        for p,g in zip(model.parameters(),grads,strict=True): p.grad=g.clone()
        opt.step()
        self.assertTrue(state_equal(actual.state_dict(),model.state_dict()))
        self.assertTrue(state_equal(after,opt.state_dict()))
        self.assertFalse(state_equal(actual.state_dict(),memory.state_dict()))
        self.assertFalse(state_equal(actual.state_dict(),reset.state_dict()))
        with self.assertRaises(ValueError): adam_counterfactual(model,state,grads,'unknown')

    def test_active_step_decomposition_and_finite_update_are_verified(self):
        from experiments.executable_behavior import PreflightBudget
        from scripts.audit_s6_update_transport import active_step
        from tests.test_s6_terminal_objective import batch
        class Tiny(torch.nn.Module):
            stop_threshold=.26
            def __init__(self):
                super().__init__(); self.linear=torch.nn.Linear(1,4)
            def forward_with_stop(self,bev,goal,state):
                value=self.linear(torch.ones(len(bev),320,1))
                return SimpleNamespace(points=value[...,:3],stop_logits=value[...,3])
        model=Tiny(); opt=torch.optim.Adam(model.parameters(),lr=1e-5)
        model.linear(torch.ones(1,1)).sum().backward(); opt.step()
        report=active_step(model,opt,batch(),'boundary_terminal',PreflightBudget(10))
        self.assertLess(report['query_logit18_loss_gradient'],0)
        self.assertLess(report['decomposition_error_norm'],1e-4)
        self.assertEqual(set(report['finite_adam_counterfactuals']),{'actual','zero_current','zero_first_moment'})

    def test_precheck_has_no_steps_and_existing_output_is_not_reused(self):
        from scripts.audit_s6_update_transport import precheck,run
        with patch('scripts.audit_s6_update_transport.prepare_context',return_value={}), \
             patch('scripts.audit_s6_update_transport.step') as update:
            precheck(); update.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.audit_s6_update_transport.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_timeout_is_saved_without_updates_or_retry(self):
        import json
        from scripts.audit_s6_update_transport import run
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.audit_s6_update_transport.prepare_context',side_effect=TimeoutError('limit')) as prepare, \
                 patch('scripts.audit_s6_update_transport.step') as update:
                with self.assertRaises(TimeoutError): run(Path(folder)/'new')
                prepare.assert_called_once(); update.assert_not_called()
            report=json.loads((Path(folder)/'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertFalse(report['allowed_to_prepare_training_plan'])


if __name__=='__main__': unittest.main()
