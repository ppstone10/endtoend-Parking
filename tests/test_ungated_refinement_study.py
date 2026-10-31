"""REF-UNGATE-004: paired preservation and separate repair/full-gate decisions."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts import compare_ungated_refinement as study
from tests.test_runtime_refinement_study import row
from experiments.emitted_output_data import read_json


class UngatedStudyTests(unittest.TestCase):
    def fixtures(self):
        old36 = [dict(dataset_index=i, **{b:row(i,b,success=(i != 860 and (b != 'network_rolling' or i != 1290)))
                                         for b in study.BRANCHES}) for i in study.INDICES]
        old37 = [row(j['index'],j['branch'],success=(j['index'] != 860 or j['branch']=='teacher_fixed')
                     and not (j['branch']=='network_fixed' and j['index'] in [1290,2315])) for j in study.jobs()]
        new = [row(j['index'],j['branch'],profile='ungated',success=(j['index']!=860 or j['branch']=='teacher_fixed')) for j in study.jobs()]
        return new, [dict(rows=old36),dict(rows=old37)]

    def test_fixed_24_episodes_66_artifacts_and_single_switch(self):
        self.assertEqual(len(study.jobs()),24)
        self.assertEqual(len({j['index'] for j in study.jobs()}),8)
        self.assertEqual(len(study.required_files(study.INDICES)),66)
        self.assertFalse(study.PROTOCOL['options']['progress'])
        self.assertTrue(study.PROTOCOL['options']['variance'])
        self.assertTrue(study.PROTOCOL['options']['selection'])
        self.assertTrue(study.PROTOCOL['options']['terminal_rotation'])

    def test_repair_does_not_hide_remaining_860_failure(self):
        rows, baseline = self.fixtures()
        with patch.object(study,'read_json',side_effect=baseline):r=study.summaries(rows,True)
        self.assertTrue(r['decision']['gate_repair_passed'])
        self.assertFalse(r['decision']['original_lt37_gates_passed'])
        self.assertFalse(r['decision']['ready_for_next_validation'])
        next(r for r in rows if r['dataset_index']==860 and r['branch']=='network_fixed')['success']=True
        with patch.object(study,'read_json',side_effect=baseline):r=study.summaries(rows,True)
        self.assertTrue(r['decision']['ready_for_next_validation'])
        self.assertFalse(r['lt34_training_unlocked'])

    def test_gain_cannot_offset_loss_in_either_baseline(self):
        rows, baseline=self.fixtures()
        rows[0]['success']=False
        next(r for r in rows if r['dataset_index']==860 and r['branch']=='network_fixed')['success']=True
        with patch.object(study,'read_json',side_effect=baseline):r=study.summaries(rows,True)
        self.assertFalse(r['decision']['gate_repair_passed'])
        self.assertEqual(r['decision']['status'],'reject_regression_or_risk')

    def test_any_slow_control_or_risk_rejects(self):
        rows,baseline=self.fixtures();rows[0]['control_timing']['over_100ms']=1
        with patch.object(study,'read_json',side_effect=baseline):r=study.summaries(rows,True)
        self.assertFalse(r['decision']['gate_repair_passed'])
        rows[0]['control_timing']['over_100ms']=0;rows[0]['margin_violation_steps']=1
        with patch.object(study,'read_json',side_effect=baseline):r=study.summaries(rows,True)
        self.assertFalse(r['decision']['gate_repair_passed'])

    def test_missing_reordered_or_duplicate_episode_rejected(self):
        rows,_=self.fixtures()
        for bad in [rows[:-1],rows[::-1],[rows[0]]*24]:
            with self.assertRaises(ValueError):study.summaries(bad,True)

    def test_smoke_never_grants_full_episode_qualification(self):
        rows=[row(0,b,profile='ungated') for b in study.BRANCHES]
        self.assertFalse(study.summaries(rows,False)['decision']['ready_for_next_validation'])

    def test_prepare_no_overwrite_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prepared'
            with patch.object(study,'frozen_config',return_value={'code':'a'}):
                study.prepare(path);self.assertEqual(study.precheck(path),{'code':'a'})
                with self.assertRaises(FileExistsError):study.prepare(path)
            with patch.object(study,'frozen_config',return_value={'code':'b'}):
                with self.assertRaises(ValueError):study.precheck(path)

    def test_fault_writes_incomplete_without_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)/'prepared';prep.mkdir();(prep/'config.json').write_text('{}')
            output=Path(directory)/'output'
            with patch.object(study,'precheck',return_value={}),patch.object(study,'load_cohort',side_effect=RuntimeError('fault')):
                with self.assertRaisesRegex(RuntimeError,'fault'):study.run(prep,output,True)
                self.assertEqual(read_json(output/'report.json')['status'],'incomplete')
                with self.assertRaises(FileExistsError):study.run(prep,output,True)


if __name__=='__main__':unittest.main()
