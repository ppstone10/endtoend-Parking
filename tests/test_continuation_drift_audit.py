"""诊断边界与单变量准备契约。"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

from experiments import validation_matrix
from scripts.audit_continuation_drift import select_collision, run_audit
from scripts.prepare_yaw_loss_ab import paired_configs


class DriftAuditTests(unittest.TestCase):
    def report(self):
        return {'status':'completed','experiment':{'name':'E5','safety_mode':'none'},
                'episodes':[{'meta':{'dataset_index':164},'collision':True,'success':False,'steps':164}], 'protocol':{}}

    def test_rejects_unbounded_non_collision_duplicate_or_intervened(self):
        baseline=self.report()
        self.assertEqual(select_collision(baseline,164)['steps'],164)
        for mutate in (lambda r:r['experiment'].update(safety_mode='shield'), lambda r:r['episodes'][0].update(steps=601), lambda r:r['episodes'][0].update(success=True),lambda r:r['episodes'].append(deepcopy(r['episodes'][0]))):
            report=deepcopy(baseline);mutate(report)
            with self.assertRaises(ValueError):select_collision(report,164)

    def test_engine_restored_on_replay_error_and_output_not_overwritten(self):
        baseline=self.report();baseline['protocol']={'data':'missing.npz','checkpoint':'missing.pt','replan_every':10,'control_seed':0}
        original=validation_matrix.ClosedLoopEngine
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.json';path.write_text(json.dumps(baseline),encoding='utf-8');out=Path(directory)/'audit'
            with patch.object(validation_matrix,'run_validation_experiment',side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError,'test failure'):run_audit(path,164,{},out)
            self.assertIs(validation_matrix.ClosedLoopEngine,original)
            with self.assertRaises(FileExistsError):run_audit(path,164,{},out)

    def test_paired_configs_change_only_mode_and_identity(self):
        source={'trainer':{'epochs':4,'checkpoint_dir':'old'},'arm':'R2','train':'same.npz','parent':'parent.pt','validation':'val.npz'}
        frozen=deepcopy(source);configs=paired_configs(source,Path('new'))
        self.assertEqual(source,frozen)
        left,right=deepcopy(configs['W0']),deepcopy(configs['W1'])
        for config in (left,right):
            config.pop('arm');config.pop('output');config['trainer'].pop('checkpoint_dir');config['trainer'].pop('yaw_error_mode')
        self.assertEqual(left,right)


if __name__ == '__main__':unittest.main()
