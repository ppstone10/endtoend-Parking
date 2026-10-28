from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.executable_behavior import PreflightBudget
from experiments.s6_recovery_learning import batch_identity
from experiments.s6_replanning_chain import pair_schedule
from experiments.s6_update_transport import state_equal
from scripts.preflight_s6_chain_learning import fit_pair,precheck,run,local_evaluation
from tests.test_s6_event_learnability import Tiny
from tests.test_s6_terminal_objective import batch


class ChainLearningTests(unittest.TestCase):
    def test_local_summary_passes_reference_points_not_trajectory(self):
        import numpy as np
        result=SimpleNamespace(record=SimpleNamespace(refs=[np.zeros((19,3),dtype=np.float32)]))
        contexts={400:SimpleNamespace(state=object()),410:SimpleNamespace(state=object())}
        def audit(points,*args):
            self.assertIsInstance(points,np.ndarray); self.assertEqual(points.shape,(19,3)); return {}
        with patch('scripts.preflight_s6_chain_learning.run_case',return_value=(result,None,None)), \
             patch('scripts.preflight_s6_chain_learning.record_arrays',return_value=(np.zeros((20,5)),np.zeros((20,2)))), \
             patch('scripts.preflight_s6_chain_learning.result_row',return_value={}), \
             patch('scripts.preflight_s6_chain_learning.save_record'), \
             patch('scripts.preflight_s6_chain_learning.reference_audit',side_effect=audit):
            local_evaluation({},SimpleNamespace(dt=.2),contexts,PreflightBudget(1),Path('.'),'test')

    def fixture(self):
        model=Tiny(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
        model.linear(torch.ones(1,1)).sum().backward(); optimizer.step()
        for value in optimizer.state.values(): value['step']=torch.tensor(240.)
        first=batch()[2]; second=deepcopy(first); second['query_step']=410; second['frame_index']=41
        samples=[first,second]
        return model,optimizer.state_dict(),samples,[batch_identity([s]) for s in pair_schedule(*samples)]

    def test_fixed60_each_and_adam360_without_changing_parent(self):
        model,state,samples,planned=self.fixture(); original=deepcopy(model.state_dict()); frozen=deepcopy(state)
        _,optimizer,history,presented=fit_pair(model,state,samples,planned,PreflightBudget(15))
        self.assertEqual(len(history),120); self.assertEqual(presented,planned)
        self.assertEqual(sum(p==[batch_identity([samples[0]])[0]] for p in presented),60)
        self.assertEqual({int(s['step']) for s in optimizer.state.values()},{360})
        self.assertTrue(state_equal(model.state_dict(),original)); self.assertTrue(state_equal(state,frozen))

    def test_identity_drift_and_wrong_start_are_rejected(self):
        model,state,samples,planned=self.fixture(); samples[0]['points'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'身份漂移'):
            fit_pair(model,state,samples,planned,PreflightBudget(1))
        for value in state['state'].values(): value['step']=torch.tensor(241.)
        with self.assertRaisesRegex(ValueError,'Adam240'):
            fit_pair(model,state,samples,planned,PreflightBudget(1))

    def test_precheck_and_output_protection(self):
        with patch('scripts.preflight_s6_chain_learning.prepare_context',return_value={}), \
             patch('scripts.preflight_s6_chain_learning.fit_pair') as fit:
            precheck(); fit.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_chain_learning.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_budget_failure_is_saved_without_retry(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_chain_learning.prepare_context',side_effect=TimeoutError('limit')) as prepare, \
                 patch('scripts.preflight_s6_chain_learning.fit_pair') as fit:
                with self.assertRaises(TimeoutError): run(Path(folder)/'new')
                prepare.assert_called_once(); fit.assert_not_called()
            report=json.loads((Path(folder)/'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertFalse(report['allowed_to_prepare_training_plan'])


if __name__=='__main__': unittest.main()
