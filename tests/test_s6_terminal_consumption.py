from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.effective_recovery_data import input_id, fingerprint
from experiments.s6_recovery_learning import batch_identity, build_recovery_schedules
from experiments.s6_terminal_consumption import terminal_schedule, terminal_qualification
from tests.test_s6_recovery_learning import fixture, SelfTeacher


def strict_schedule():
    base, queries = fixture()
    for q in queries:
        q['stop'] = (torch.arange(320) >= 18).float()
        q['mask'] = (torch.arange(320) < 11).float()
    return build_recovery_schedules(base, queries, SelfTeacher())[0]['strict_targets']


class S6TerminalConsumptionTests(unittest.TestCase):
    def test_one_mask_changes_other_labels_and_input_remain_exact(self):
        base = strict_schedule(); original = deepcopy(base); union = (torch.arange(320) < 19).float()
        changed, audit = terminal_schedule(base, union)
        self.assertEqual(audit['changed_slot'], [16, 2]); self.assertEqual(audit['added_indices'], list(range(11, 19)))
        for j, batch in enumerate(original):
            for k, sample in enumerate(batch):
                self.assertEqual(input_id(sample), input_id(changed[j][k]))
                self.assertEqual(fingerprint(sample, ('points','stop')), fingerprint(changed[j][k], ('points','stop')))
                if (j,k) != (16,2): self.assertEqual(batch_identity([sample]), batch_identity([changed[j][k]]))
        self.assertEqual(batch_identity(base[16]), batch_identity(original[16]))

    def test_rejects_wrong_query_old_mask_unproved_or_past_stop_range(self):
        mask = (torch.arange(320) < 19).float()
        for field, value in [('query_step', 200), ('source_index', 2256), ('mask', torch.ones(320))]:
            base = strict_schedule(); base[16][2][field] = value
            with self.assertRaises(ValueError): terminal_schedule(base, mask)
        for union in (torch.ones(320), torch.zeros(320), (torch.arange(320) < 18).float()):
            with self.assertRaises(ValueError): terminal_schedule(strict_schedule(), union)

    def test_increment_requires_both_orders_no_success_loss_and_original_gate(self):
        def row(index, success):
            return dict(dataset_index=index, task_id=str(index), scene_name='S6_scene',success=success,collision=not success)
        old = {s:[row(1, True),row(2,False)] for s in (20261020,20261021)}
        new = {s:[row(1, True),row(2,True)] for s in (20261020,20261021)}
        self.assertTrue(terminal_qualification({'passed':True}, old, new)['passed'])
        self.assertFalse(terminal_qualification({'passed':False}, old, new)['passed'])
        self.assertFalse(terminal_qualification({'passed':True}, old, old)['passed'])
        bad = deepcopy(new); bad[20261021][0]['success'] = False
        self.assertFalse(terminal_qualification({'passed':True}, old, bad)['passed'])
        self.assertFalse(terminal_qualification({'passed':True}, old, {20261020:new[20261020]})['passed'])

    def test_precheck_no_fit_and_output_cannot_be_reused(self):
        from scripts.preflight_s6_terminal_consumption import precheck, run
        with patch('scripts.preflight_s6_terminal_consumption.prepare_context', return_value={}), \
             patch('scripts.preflight_s6_terminal_consumption.fit_recovery_point') as fit:
            precheck(); fit.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_terminal_consumption.prepare_context') as prepare:
                with self.assertRaises(FileExistsError): run(Path(folder))
                prepare.assert_not_called()

    def test_budget_exhaustion_keeps_incomplete_report_without_retry(self):
        import json
        from scripts.preflight_s6_terminal_consumption import run
        with tempfile.TemporaryDirectory() as folder:
            with patch('scripts.preflight_s6_terminal_consumption.prepare_context',side_effect=TimeoutError('limit')) as prepare, \
                 patch('scripts.preflight_s6_terminal_consumption.fit_recovery_point') as fit:
                with self.assertRaises(TimeoutError): run(Path(folder)/'new')
                prepare.assert_called_once(); fit.assert_not_called()
            report=json.loads((Path(folder)/'new/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertFalse(report['allowed_to_prepare_training_plan'])


if __name__=='__main__': unittest.main()
