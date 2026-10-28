import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

import numpy as np

from experiments.s8_entry_recovery import select_entry_contexts, validated_entry_query


class EntryTests(unittest.TestCase):
    def test_initial_and_latest_before_first_actual_margin(self):
        contexts=[SimpleNamespace(step=i) for i in (0,10,20,30,40,410)]
        selected,audit=select_entry_contexts(contexts,[False]*25+[True])
        self.assertEqual([c.step for c in selected],[0,20]);self.assertFalse(audit['nonzero_missing'])
        selected,audit=select_entry_contexts(contexts,[False]*7+[True])
        self.assertEqual([c.step for c in selected],[0]);self.assertTrue(audit['nonzero_missing'])
        selected,_=select_entry_contexts(contexts,[False]*600)
        self.assertEqual([c.step for c in selected],[0,40])

    def test_missing_initial_or_invalid_phase_fails_closed(self):
        with self.assertRaises(ValueError):select_entry_contexts([SimpleNamespace(step=10)],[False]*20)
        with self.assertRaises(ValueError):select_entry_contexts([SimpleNamespace(step=0),SimpleNamespace(step=11)],[])

    def test_zero_and_nonzero_queries_share_exact_window_alignment(self):
        budget=SimpleNamespace(check=lambda:None)
        for step in (0,20):
            prefix=np.full((step,5),1.);suffix=np.full((12,5),2.)
            sample=dict(frame_index=step//10,role='recovery')
            with patch('experiments.s8_entry_recovery.query_sample',return_value=sample), \
                    patch('experiments.s8_entry_recovery.audit_source_windows',return_value=[{'clean':True}]) as audit:
                actual,window=validated_entry_query({},[{}],{'index':2351},step,'policy',None,budget,prefix,suffix)
                self.assertIs(actual,sample);self.assertTrue(window['clean'])
                self.assertTrue(np.array_equal(audit.call_args.args[1],np.concatenate((prefix,suffix))))
                self.assertEqual(audit.call_args.args[0][0]['role'],'anchor')
        with self.assertRaises(ValueError):
            validated_entry_query({},[],{'index':2351},20,'policy',None,budget,np.zeros((19,5)),np.zeros((1,5)))

    def test_cli_precheck_and_budget_failure_never_query_teacher(self):
        from scripts import preflight_s8_entry_recovery as cli
        with patch.object(cli,'prepare_config',return_value={}),patch.object(cli,'teacher_query') as teacher, \
                patch.object(cli,'run_case') as run:
            cli.precheck();teacher.assert_not_called();run.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'new'
            with patch.object(cli,'prepare_config',side_effect=TimeoutError('budget')),patch.object(cli,'run_case') as run:
                with self.assertRaises(TimeoutError):cli.run(target)
                run.assert_not_called()
            report=json.loads((target/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted');self.assertEqual(report['training_updates'],0)
            self.assertFalse(report['allowed_to_prepare_training_plan'])
            with self.assertRaises(FileExistsError):cli.run(target)

    def test_unsafe_connected_reference_is_rejected_before_execution(self):
        from scripts import preflight_s8_entry_recovery as cli
        case=dict(index=2351,goal=None,planner=SimpleNamespace(plan=lambda *args:None,_collision_checker=None))
        context=SimpleNamespace(state=None,step=0);row=dict(attempts=[],accepted=False)
        with patch.object(cli,'safe_connected_reference',return_value=False),patch.object(cli,'run_case') as run:
            result=cli.teacher_query(case,context,{},np.zeros((1,5)),None,'policy',SimpleNamespace(check=lambda:None),
                Path('.'),row,lambda:None)
        self.assertIsNone(result);run.assert_not_called();self.assertEqual(len(row['attempts']),1)
        self.assertFalse(row['accepted'])


if __name__=='__main__':unittest.main()
