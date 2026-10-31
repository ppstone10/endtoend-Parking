"""EMIT-JOINT-002/003/005: fixed provenance, pairing and fail-closed gates."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.emitted_output_data import INDICES, RECIPE, teacher_gate, load_authorized_samples, regression_cohort
from experiments.emitted_output_study import batch_plan, fitting_decision, update
from model.variants import MineParkingNetV1


class EmittedOutputStudyTests(unittest.TestCase):
    def test_teacher_failure_is_not_hidden_or_replaced(self):
        rows = [dict(dataset_index=i,passed=True) for i in INDICES]
        self.assertTrue(teacher_gate(rows)['passed'])
        rows[3]['passed'] = False
        gate = teacher_gate(rows)
        self.assertEqual(gate['failed_indices'], [INDICES[3]])
        self.assertFalse(gate['passed'])
        self.assertIn('no_training',gate['decision'])
        for changed in [rows[:-1], rows[:-1]+[rows[0]], list(reversed(rows))]:
            with self.assertRaises(ValueError): teacher_gate(changed)

    def test_missing_teacher_prevents_training_before_data_load(self):
        with tempfile.TemporaryDirectory() as directory, patch('experiments.emitted_output_data.precheck'):
            with self.assertRaises(FileNotFoundError):
                load_authorized_samples(directory, Path(directory)/'absent')

    def test_plan_is_identical_and_balanced_across_arms(self):
        samples = [dict(source_index=i,frame_index=f) for i in INDICES for f in range(4)]
        a = batch_plan(samples,RECIPE['seeds'][0],200)
        self.assertEqual(a,batch_plan(samples,RECIPE['seeds'][0],200))
        self.assertNotEqual(a,batch_plan(samples,RECIPE['seeds'][1],200))
        self.assertEqual(len(a),200)
        for batch in a:
            self.assertEqual({samples[j]['source_index'] for j in batch},set(INDICES))
        with self.assertRaises(ValueError): batch_plan(samples[4:],1,200)

    def test_final_two_seeds_must_pass_no_best_seed_selection(self):
        arms = [dict(seed=seed,arm=arm,final=dict(passed=True))
                for seed in RECIPE['seeds'] for arm in ('control','joint')]
        self.assertTrue(fitting_decision(arms)['passed'])
        arms[-1]['final']['passed'] = False
        gate = fitting_decision(arms)
        self.assertFalse(gate['passed'])
        self.assertIn('stop_recipe',gate['decision'])
        self.assertFalse(gate['candidate_qualified'])
        with self.assertRaises(ValueError): fitting_decision(arms[:-1])

    def test_regression_is_task_deduplicated_and_validation_only(self):
        lt33 = dict(status='completed',lost_lt32_successes=list(range(22)),gained_successes=list(range(20,28)))
        lt32 = dict(rows=dict(native=[dict(dataset_index=i,success=True) for i in range(93)]))
        metadata = [dict(task_id=str(i)) for i in range(300)]
        result = regression_cohort(lt33,lt32,metadata)
        self.assertEqual(result['unique_tasks'],98)
        self.assertEqual(len(result['indices']),len(set(result['indices'])))
        self.assertTrue(result['evaluation_only'])
        metadata[21] = metadata[20]
        with self.assertRaises(ValueError): regression_cohort(lt33,lt32,metadata)

    def test_actual_model_update_has_both_head_gradients_and_same_original_loss(self):
        torch.manual_seed(3)
        parent = MineParkingNetV1(max_horizon=4,hidden_dim=8)
        sample = dict(bev=torch.randn(5,16,16),goal=torch.tensor([2.,1.,.5]),state=torch.zeros(2),
                      points=torch.randn(4,3),mask=torch.ones(4),tolerances=torch.tensor([.3,.1]),
                      source_index=INDICES[0],frame_index=0)
        histories=[]
        for weight in (0.,.1):
            model=deepcopy(parent); opt=torch.optim.Adam(model.parameters(),lr=1e-5)
            histories.append(update(model,opt,[sample],seed=1,step=1,joint_weight=weight,inspect_gradients=bool(weight)))
        self.assertTrue(np.isclose(histories[0]['total_loss'],histories[1]['base_loss'],rtol=2e-6))
        self.assertTrue(all(np.isfinite(x) and x>0 for x in histories[1]['joint_head_gradient_norms']))
        self.assertEqual(histories[0]['samples'],histories[1]['samples'])


if __name__ == '__main__':
    unittest.main()
