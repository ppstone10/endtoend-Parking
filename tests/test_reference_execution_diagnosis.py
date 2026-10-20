"""把桥接、参考内部、控制执行的风险分开。"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from experiments import validation_matrix
from training.reporting import atomic_write_json
from scripts.diagnose_reference_execution import window_risks, classify_contact, select_indices, run_diagnosis, summarize_frames, code_identity


class Checker:
    def pose_free(self,x,y,yaw): return x < 1
    def swept_segment_free(self,a,b): return self.pose_free(*a[:3]) and self.pose_free(*b[:3])


class ReferenceExecutionTests(unittest.TestCase):
    def test_safe_reference_actual_execution_collision(self):
        risk=window_risks(np.array([0,0,0]),np.array([[.1,0,0],[.2,0,0]]),np.array([1.1,0,0]),np.array([[1.1,0,0]]),Checker())
        self.assertIsNone(risk['reference_internal_blocked_index'])
        self.assertTrue(risk['connection_safe'])
        self.assertFalse(risk['actual_transition_safe'])
        self.assertEqual(classify_contact(risk),'execution_left_contact_safe_reference')

    def test_unsafe_reference_not_execution_only(self):
        risk=window_risks(np.array([0,0,0]),np.array([[.1,0,0],[1.1,0,0]]),np.array([.1,0,0]),np.array([[.1,0,0]]),Checker())
        self.assertEqual(risk['reference_internal_blocked_index'],1)
        self.assertEqual(classify_contact(risk),'reference_internal_unsafe')

    def test_bridge_failure_does_not_mean_internal_reference_unsafe(self):
        class GapChecker(Checker):
            def swept_segment_free(self,a,b):return abs(a[0]-b[0])<.5
        risk=window_risks(np.array([-1,0,0]),np.array([[0,0,0],[.1,0,0]]),np.array([-.9,0,0]),np.array([[-.9,0,0]]),GapChecker())
        self.assertIsNone(risk['reference_internal_blocked_index'])
        self.assertFalse(risk['connection_safe'])
        self.assertEqual(classify_contact(risk),'state_to_reference_connection_unsafe')

    def test_first_window_pose_checked_and_already_bad_state_is_separate(self):
        risk=window_risks(np.array([0,0,0]),np.array([[1.1,0,0]]),np.array([.1,0,0]),np.array([[.1,0,0]]),Checker())
        self.assertEqual(risk['reference_internal_blocked_index'],0)
        risk=window_risks(np.array([1.1,0,0]),np.array([[.1,0,0]]),np.array([.1,0,0]),np.array([[.1,0,0]]),Checker())
        self.assertEqual(classify_contact(risk),'state_already_outside_contact_free_set')

    def test_explicit_indices_and_success_controls(self):
        report={'status':'completed','experiment':{'name':'E5','safety_mode':'none'},'episodes':[{'meta':{'dataset_index':1},'success':True}]}
        self.assertEqual(select_indices(report,[1]),[1])
        for indices in ([],[1,1],[2]):
            with self.assertRaises(ValueError):select_indices(report,indices)

    def test_earlier_reference_risk_is_not_overwritten_by_safe_window_collision(self):
        base={'state_free':True,'reference_internal_blocked_index':None,'connection_safe':True,'actual_transition_safe':True,'mpc_rollout_blocked_index':None}
        rows=[]
        for step in range(3):
            risk=dict(base)
            if step==0:risk['reference_internal_blocked_index']=2
            if step==2:risk['actual_transition_safe']=False
            rows.append({'step':step,'plan_issue_step':0,'plan_ordinal':0,'contact':risk,'planning':dict(risk)})
        events=summarize_frames(rows)
        self.assertEqual(events['contact_reference_internal']['step'],0)
        self.assertEqual(events['contact_actual_transition']['step'],2)
        self.assertIsNone(events['contact_reference_internal']['previous_step'])
        self.assertEqual(classify_contact(rows[-1]['contact']),'execution_left_contact_safe_reference')

    def test_empty_timeline_has_no_invented_events(self):
        self.assertTrue(all(v is None for v in summarize_frames([]).values()))

    def test_identity_covers_sensor_to_bev_and_controller(self):
        paths=[Path(p).parts for p in code_identity()]
        self.assertTrue(any('sensor2bev' in p for p in paths))
        self.assertTrue(any('controller' in p for p in paths))

    def test_factories_restore_on_error_no_overwrite(self):
        engine,builder=validation_matrix.ClosedLoopEngine,validation_matrix._build_source
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data.npz';data.write_bytes(b'data');model=root/'model.pt';model.write_bytes(b'model')
            p=atomic_write_json(root/'input.json',{'status':'completed','experiment':{'name':'E5','safety_mode':'none'},'protocol':{'data':str(data),'checkpoint':str(model),'control_seed':0,'max_steps':600,'replan_every':10},'episodes':[{'meta':{'dataset_index':1},'success':True}]})
            with patch.object(validation_matrix,'run_validation_experiment',side_effect=RuntimeError('fixture')):
                with self.assertRaisesRegex(RuntimeError,'fixture'):run_diagnosis(p,root/'out',[1])
            self.assertIs(validation_matrix.ClosedLoopEngine,engine);self.assertIs(validation_matrix._build_source,builder)
            with self.assertRaises(FileExistsError):run_diagnosis(p,root/'out',[1])


if __name__=='__main__':unittest.main()
