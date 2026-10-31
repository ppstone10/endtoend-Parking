"""MPC-RESCUE-005: three-baseline success union and honest failure decisions."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts import compare_zero_hold_motion as study
from tests.test_runtime_refinement_study import row
from experiments.emitted_output_data import read_json


class RescueStudyTests(unittest.TestCase):
    def fixtures(self):
        b36=[dict(dataset_index=i,**{b:row(i,b,success=(i!=860 and (b!='network_rolling' or i!=1290))) for b in study.BRANCHES}) for i in study.INDICES]
        b37=[row(j['index'],j['branch'],success=(j['index']!=860 or j['branch']=='teacher_fixed')
                 and not(j['branch']=='network_fixed' and j['index'] in [1290,2315])) for j in study.jobs()]
        b38=[row(j['index'],j['branch'],profile='ungated',success=(j['index']!=2315 or j['branch']=='network_rolling')
                 and not(j['index']==860 and j['branch']!='network_fixed')) for j in study.jobs()]
        new=[row(j['index'],j['branch'],profile='zero_rescue',success=(j['index']!=860 or j['branch']!='network_rolling')) for j in study.jobs()]
        return new,[dict(rows=b36),dict(rows=b37),dict(rows=b38)]

    def test_fixed_denominator_budget_and_smoke(self):
        self.assertEqual(len(study.jobs()),24);self.assertEqual(len({j['index'] for j in study.jobs()}),8)
        self.assertEqual(len(study.required_files(study.INDICES)),66)
        self.assertEqual(study.PROTOCOL['population']*study.PROTOCOL['iterations'],400)
        self.assertEqual(study.PROTOCOL['supplemental_candidate_limit'],8)
        r=study.summaries([row(0,b) for b in study.BRANCHES],False)
        self.assertFalse(r['decision']['ready_for_next_validation'])

    def test_only_success_union_preservation_can_pass(self):
        rows,baselines=self.fixtures()
        with patch.object(study,'read_json',side_effect=baselines):result=study.summaries(rows,True)
        self.assertTrue(result['decision']['ready_for_next_validation']);self.assertFalse(result['lt34_training_unlocked'])
        # Preserve LT38's number of fixed successes but lose LT36's 2315 again.
        next(r for r in rows if r['dataset_index']==2315 and r['branch']=='network_fixed')['success']=False
        with patch.object(study,'read_json',side_effect=baselines):result=study.summaries(rows,True)
        self.assertEqual(result['decision']['status'],'reject_regression_or_risk')

    def test_teacher_loss_and_any_slow_call_cannot_be_hidden(self):
        rows,baselines=self.fixtures();rows[0]['control_timing']['over_100ms']=1
        with patch.object(study,'read_json',side_effect=baselines):result=study.summaries(rows,True)
        self.assertFalse(result['decision']['ready_for_next_validation'])
        rows[0]['control_timing']['over_100ms']=0
        next(r for r in rows if r['dataset_index']==860 and r['branch']=='teacher_fixed')['success']=False
        with patch.object(study,'read_json',side_effect=baselines):result=study.summaries(rows,True)
        self.assertFalse(result['decision']['ready_for_next_validation'])

    def test_duplicate_missing_or_reordered_episodes_fail(self):
        rows,_=self.fixtures()
        for bad in [rows[:-1],rows[::-1],[rows[0]]*24]:
            with self.assertRaises(ValueError):study.summaries(bad,True)

    def test_freeze_refuses_overwrite_and_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'prepared'
            with patch.object(study,'frozen_config',return_value={'code':'a'}):
                study.prepare(p);self.assertEqual(study.precheck(p),{'code':'a'})
                with self.assertRaises(FileExistsError):study.prepare(p)
            with patch.object(study,'frozen_config',return_value={'code':'b'}):
                with self.assertRaises(ValueError):study.precheck(p)

    def test_failure_is_incomplete_without_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'prepared';p.mkdir();(p/'config.json').write_text('{}')
            output=Path(directory)/'output'
            with patch.object(study,'precheck',return_value={}),patch.object(study,'load_cohort',side_effect=RuntimeError('fault')):
                with self.assertRaisesRegex(RuntimeError,'fault'):study.run(p,output,True)
                self.assertEqual(read_json(output/'report.json')['status'],'incomplete')
                with self.assertRaises(FileExistsError):study.run(p,output,True)


if __name__=='__main__':unittest.main()
