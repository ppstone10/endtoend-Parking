"""RUN-REFINE-006: fixed denominators, no overwrite/drift and honest decisions."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import compare_runtime_refinement as study
from experiments.emitted_output_data import read_json


def row(index, branch, profile='combined', success=True, **kwargs):
    return dict(dataset_index=index, task_id=str(index), branch=branch, profile=profile,
        success=success, collision=False, margin_violation_steps=0, steps=20,
        final_pos_err=.1, final_yaw_err=.05, zero_hold_steps=0, safety_stop=False,
        control_timing=dict(over_100ms=0), **kwargs)


class RuntimeStudyTests(unittest.TestCase):
    def test_fixed_30_episode_eight_unique_task_protocol(self):
        jobs = study.jobs()
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({j['index'] for j in jobs}), 8)
        self.assertEqual(len({(j['index'], j['branch'], j['profile']) for j in jobs}), 30)
        self.assertEqual(sum(j['profile'] == 'combined' for j in jobs), 24)
        self.assertEqual(len(study.required_files(False)), 80)
        self.assertEqual(len(study.required_files(True)), 10)

    def test_prepare_never_overwrites_and_precheck_rejects_drift(self):
        config = dict(schema='test', code={'x':'a'})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'prepared'
            with patch.object(study, 'frozen_config', return_value=config):
                study.prepare(destination)
                self.assertEqual(study.precheck(destination), config)
                with self.assertRaises(FileExistsError): study.prepare(destination)
            with patch.object(study, 'frozen_config', return_value=dict(schema='changed')):
                with self.assertRaises(ValueError): study.precheck(destination)

    def test_smoke_never_grants_completed_episode_qualification(self):
        rows = [row(0, b) for b in study.legacy.BRANCHES]
        result = study.summaries(rows, formal=False)
        self.assertFalse(result['decision']['ready_for_next_validation'])
        self.assertFalse(result['lt34_training_unlocked'])

    def formal(self):
        rows = [row(j['index'], j['branch'], j['profile']) for j in study.jobs()]
        baseline = [dict(dataset_index=i, **{b: row(i, b, success=i != 860)
                     for b in study.legacy.BRANCHES}) for i in study.INDICES]
        return rows, dict(rows=baseline)

    def test_gains_cannot_offset_lost_success_and_any_slow_call_fails(self):
        rows, baseline = self.formal()
        with patch.object(study, 'read_json', return_value=baseline):
            result = study.summaries(rows, True)
            self.assertTrue(result['decision']['ready_for_next_validation'])
            rows[0]['success'] = False
            result = study.summaries(rows, True)
            self.assertEqual(result['decision']['status'], 'reject_candidate_regression_or_risk')
            rows[0]['success'] = True
            rows[0]['control_timing']['over_100ms'] = 1
            self.assertFalse(study.summaries(rows, True)['decision']['ready_for_next_validation'])

    def test_failed_run_is_incomplete_and_does_not_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / 'prepared'
            prepared.mkdir();(prepared/'config.json').write_text('{}')
            output = Path(directory) / 'output'
            with patch.object(study, 'precheck', return_value={}), patch.object(study, 'load_cohort', side_effect=RuntimeError('fault')):
                with self.assertRaisesRegex(RuntimeError, 'fault'): study.run(prepared, output, smoke=True)
                self.assertEqual(read_json(output/'report.json')['status'], 'incomplete')
                with self.assertRaises(FileExistsError): study.run(prepared, output, smoke=True)


if __name__ == '__main__':
    unittest.main()
