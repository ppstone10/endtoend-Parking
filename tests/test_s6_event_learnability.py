from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.executable_behavior import PreflightBudget
from experiments.s6_recovery_learning import batch_identity
from experiments.s6_update_transport import state_equal
from scripts.preflight_s6_event_learnability import fit_model,precheck,run
from tests.test_s6_terminal_objective import batch


class Tiny(torch.nn.Module):
    stop_threshold=.26
    def __init__(self):
        super().__init__(); self.linear=torch.nn.Linear(1,4)
    def forward_with_stop(self,bev,goal,state):
        value=self.linear(torch.ones(len(bev),320,1))
        return SimpleNamespace(points=value[...,:3],stop_logits=value[...,3])


class EventLearnabilityTests(unittest.TestCase):
    def parent(self):
        model=Tiny(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
        model.linear(torch.ones(1,1)).sum().backward(); optimizer.step()
        for value in optimizer.state.values(): value['step']=torch.tensor(240.)
        return model,optimizer.state_dict()

    def test_fixed120_updates_and_original_untouched(self):
        model,state=self.parent(); original=deepcopy(model.state_dict()); frozen=deepcopy(state)
        sample=batch()[2]; identity=batch_identity([sample])[0]
        fitted,optimizer,history,norms=fit_model(model,state,sample,'boundary_terminal',PreflightBudget(15),identity)
        self.assertEqual(len(history),120); self.assertEqual(len(norms),120)
        self.assertEqual({int(s['step']) for s in optimizer.state.values()},{360})
        self.assertTrue(state_equal(original,model.state_dict())); self.assertTrue(state_equal(frozen,state))
        self.assertFalse(state_equal(original,fitted.state_dict()))

    def test_identity_and_start_step_rejected_before_updates(self):
        model,state=self.parent(); sample=batch()[2]; identity=batch_identity([sample])[0]
        sample['points'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'身份漂移'):
            fit_model(model,state,sample,'boundary_only',PreflightBudget(1),identity)
        for value in state['state'].values(): value['step']=torch.tensor(241.)
        with self.assertRaisesRegex(ValueError,'Adam240'):
            fit_model(model,state,sample,'boundary_only',PreflightBudget(1),identity)

    def test_precheck_and_output_protection(self):
        with patch('scripts.preflight_s6_event_learnability.prepare_context',return_value={}), \
             patch('scripts.preflight_s6_event_learnability.fit_model') as fit:
            precheck(); fit.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_event_learnability.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_timeout_is_saved_without_retry(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_event_learnability.prepare_context',side_effect=TimeoutError('limit')) as prepare, \
                 patch('scripts.preflight_s6_event_learnability.fit_model') as fit:
                with self.assertRaises(TimeoutError): run(Path(folder)/'new')
                prepare.assert_called_once(); fit.assert_not_called()
            result=json.loads((Path(folder)/'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'budget_exhausted')
            self.assertFalse(result['allowed_to_prepare_training_plan'])


if __name__=='__main__': unittest.main()
