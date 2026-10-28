from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.s6_replanning_chain import complete_consumption_mask,full_reference_joint_loss,pair_schedule,reference_at_query
from experiments.s6_terminal_objective import objective_loss
from tests.test_s6_terminal_objective import batch


class ReplanningChainTests(unittest.TestCase):
    def test_reference_record_is_per_control_step_not_per_prediction(self):
        record=SimpleNamespace(refs=[np.full((2,3),i) for i in range(30)])
        self.assertTrue(np.array_equal(reference_at_query(record,400,410),np.full((2,3),10)))
        self.assertTrue(np.array_equal(reference_at_query(record,400,420),np.full((2,3),20)))
        with self.assertRaises(ValueError): reference_at_query(record,400,405)
        with self.assertRaises(ValueError): reference_at_query(record,400,430)

    def test_complete_remaining_labels_require_actual_union_and_offset(self):
        # raw0..4首窗，后续只消费4..9，映射offset4后的六点。
        mask=complete_consumption_mask(10,4,6,[(8,19),(10,19),(14,19)],horizon=10)
        self.assertTrue(torch.equal(mask,(torch.arange(320)<6).float()))
        with self.assertRaises(ValueError): complete_consumption_mask(10,4,6,[(8,19)],horizon=2)
        with self.assertRaises(ValueError): complete_consumption_mask(10,4,5,[(8,19)],horizon=10)

    def test_generic_completed_joint_loss_matches_old_400_loss_and_gradients(self):
        sample=batch()[2]
        points=torch.randn(1,320,3,requires_grad=True); logits=torch.randn(1,320,requires_grad=True)
        old=objective_loss(points,logits,[sample],'boundary_terminal')
        new=full_reference_joint_loss(points,logits,sample)
        torch.testing.assert_close(old,new)
        for a,b in zip(torch.autograd.grad(old,(points,logits),retain_graph=True),torch.autograd.grad(new,(points,logits)),strict=True):
            torch.testing.assert_close(a,b)
        invalid=deepcopy(sample); invalid['mask'][18]=0
        with self.assertRaises(ValueError): full_reference_joint_loss(points,logits,invalid)

    def test_schedule_is_fixed_alternating120_without_mutating_queries(self):
        first=batch()[2]; second=deepcopy(first); second['query_step']=410
        schedule=pair_schedule(first,second)
        self.assertEqual(len(schedule),120)
        self.assertTrue(all(s is first if i%2==0 else s is second for i,s in enumerate(schedule)))
        self.assertEqual(first['query_step'],400)

    def test_precheck_does_not_collect_and_existing_output_is_rejected(self):
        from scripts.preflight_s6_replanning_chain import precheck,run
        with patch('scripts.preflight_s6_replanning_chain.prepare_context',return_value={}), \
             patch('scripts.preflight_s6_replanning_chain.collect') as collect:
            precheck(); collect.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_replanning_chain.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_timeout_is_saved_without_collect_or_retry(self):
        import json
        from scripts.preflight_s6_replanning_chain import run
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_replanning_chain.prepare_context',side_effect=TimeoutError('limit')) as prepare, \
                 patch('scripts.preflight_s6_replanning_chain.collect') as collect:
                with self.assertRaises(TimeoutError): run(Path(folder)/'new')
                prepare.assert_called_once(); collect.assert_not_called()
            report=json.loads((Path(folder)/'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertFalse(report['training_performed'])


if __name__=='__main__': unittest.main()
