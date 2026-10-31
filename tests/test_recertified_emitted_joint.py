"""EMIT-RECERT: experimental contract, unchanged objective and authorization boundary."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from interfaces import VehicleState
from scripts import emitted_teacher_recertification as teacher
from scripts import train_recertified_emitted_joint as study
from scripts import recertified_emitted_fit as fitting
from experiments.emitted_output_study import fitting_decision,batch_plan


def gate_data():
    episodes=[dict(index=i,passed=True) for i in teacher.INDICES]
    frames=[dict(source_index=i,frame_index=j,passed=True) for i in teacher.INDICES for j in (0,10,20,30)]
    return episodes,frames


class RecertificationTests(unittest.TestCase):
    def test_gate_requires_eight_sources_and_all_32_frames(self):
        episodes,frames=gate_data()
        result=teacher.teacher_gate(episodes,frames)
        self.assertTrue(result['passed']);self.assertFalse(result['original_mpc_gate_replaced'])
        self.assertFalse(result['original_lt34_fit_unlocked']);self.assertFalse(result['candidate_qualified'])
        for a,b in [(episodes[:-1],frames),(episodes,frames[:-1]),(episodes,frames[:31]+[frames[0]])]:
            with self.assertRaises(ValueError):teacher.teacher_gate(a,b)
        frames[5]['passed']=False
        self.assertEqual(teacher.teacher_gate(episodes,frames)['failed_sources'],[433])

    def test_teacher_execution_failure_blocks_all_frames_of_that_source(self):
        episodes,frames=gate_data();episodes[2]['passed']=False
        self.assertFalse(teacher.teacher_gate(episodes,frames)['passed'])
        self.assertEqual(teacher.teacher_gate(episodes,frames)['failed_sources'],[860])

    def test_remaining_target_is_original_continuous_suffix_with_full_prefix_mask(self):
        reference=np.zeros((5,3),dtype=np.float32);reference[:,0]=np.arange(5)
        points,mask,offset=teacher.remaining_label(reference,9,3,VehicleState(1.,0.,0.))
        self.assertEqual(offset,1);self.assertEqual(int(mask.sum()),4)
        torch.testing.assert_close(points[:4,0],torch.arange(4,dtype=torch.float32),rtol=0,atol=0)
        torch.testing.assert_close(mask[:4],torch.ones(4),rtol=0,atol=0)
        self.assertFalse(points[4:].any());self.assertFalse(mask[4:].any())
        with self.assertRaises(ValueError):teacher.remaining_label(np.zeros((321,3),dtype=np.float32),641,0,VehicleState(0.,0.,0.))

    def test_endpoint_uses_actual_strict_position_yaw_and_wrap(self):
        self.assertFalse(teacher.strict_pose([.3,0.,0.],[0.,0.,0.],[.3,.1])['joint_passed'])
        self.assertTrue(teacher.strict_pose([.29,0.,2*np.pi],[0.,0.,0.],[.3,.1])['joint_passed'])
        wrapped=float(abs(np.arctan2(np.sin(.1),np.cos(.1))))
        self.assertFalse(teacher.strict_pose([0.,0.,.1],[0.,0.,0.],[.3,wrapped])['joint_passed'])

    def test_failed_authorization_refuses_fit_before_output_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'fit';prepared=Path(directory)/'prepared';prepared.mkdir()
            (prepared/'config.json').write_text('{}')
            with patch.object(study,'precheck',return_value=dict(context={})),\
                 patch.object(study,'load_authorized',side_effect=ValueError('teacher gate failed')),\
                 patch.object(study,'execute_fit') as execute,patch('torch.optim.Adam') as optimizer:
                with self.assertRaisesRegex(ValueError,'teacher gate'):study.fit(prepared,'teacher',destination)
                execute.assert_not_called();optimizer.assert_not_called();self.assertFalse(destination.exists())

    def test_false_gate_cannot_be_replaced_by_old_samples(self):
        episodes,frames=gate_data();frames[0]['passed']=False
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);torch.save([],root/'samples.pt')
            report=dict(status='completed',context={},training_performed=False,validation_used_for_fitting=False,
                selected_samples=32,episodes=episodes,frames=frames,gate=teacher.teacher_gate(episodes,frames))
            (root/'report.json').write_text(json.dumps(report))
            (root/'artifacts.json').write_text(json.dumps({str((root/n).resolve()):teacher.file_digest(root/n) for n in ('samples.pt','report.json')}))
            with self.assertRaisesRegex(ValueError,'not passed'):teacher.load_authorized(root,{})

    def test_certification_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):teacher.certify(directory,{},None)


class FittingTests(unittest.TestCase):
    def test_predeclared_dose_and_artifact_denominator(self):
        self.assertEqual(study.RECIPE['seeds'],[20261028,20261029])
        self.assertEqual(study.RECIPE['updates'],200);self.assertEqual(study.RECIPE['checkpoints'],[0,50,100,200])
        self.assertEqual(len(fitting.expected_files()),56)
        self.assertEqual(study.RECIPE['joint_weight'],.1);self.assertEqual(study.RECIPE['temperature'],.25)
        samples=[dict(source_index=i,frame_index=j) for i in teacher.INDICES for j in (0,1,2,3)]
        a=batch_plan(samples,20261028,200);b=batch_plan(samples,20261028,200)
        self.assertEqual(a,b);self.assertNotEqual(a,batch_plan(samples,20261029,200))
        for batch in a:self.assertEqual({samples[j]['source_index'] for j in batch},set(teacher.INDICES))

    def test_earlier_good_checkpoint_or_one_seed_cannot_rescue_final_failure(self):
        arms=[dict(seed=s,arm=a,final=dict(passed=True),checkpoints=[dict(step=50,passed=True)])
              for s in study.RECIPE['seeds'] for a in ('control','joint')]
        self.assertTrue(fitting_decision(arms)['passed'])
        arms[-1]['final']['passed']=False
        self.assertFalse(fitting_decision(arms)['passed'])
        with self.assertRaises(ValueError):fitting_decision(arms[:-1])

    def test_native_hard_endpoint_is_not_teacher_length_or_legacy_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'native'
            def old_evaluate(*args):
                points=np.zeros((1,320,3),dtype=np.float64);points[0,1,0]=.3
                np.savez_compressed(output.with_suffix('.npz'),points=points,logits=np.zeros((1,320)))
                return dict(rows=[dict(frame_index=0,native_length=2,teacher_length=3,contact_safe=True,margin_safe=True,passed=True)])
            sample=dict(goal=torch.zeros(3),tolerances=torch.tensor([.3,.1],dtype=torch.float64))
            with patch.object(fitting,'original_evaluate',side_effect=old_evaluate):
                result=fitting.native_evaluate(None,[sample],None,None,output)
            self.assertFalse(result['passed']);self.assertTrue(result['rows'][0]['teacher_index_endpoint']['joint_passed'])
            self.assertFalse(result['teacher_length_used_for_inference'])

    def test_new_prepared_refuses_overwrite_and_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared=Path(directory)/'prepared'
            with patch.object(study,'frozen_config',return_value=dict(code='a')):
                study.prepare(prepared,'teacher');self.assertEqual(study.precheck(prepared,'teacher'),dict(code='a'))
                with self.assertRaises(FileExistsError):study.prepare(prepared,'teacher')
            with patch.object(study,'frozen_config',return_value=dict(code='b')):
                with self.assertRaises(ValueError):study.precheck(prepared,'teacher')

    def test_fit_fault_is_incomplete_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fit'
            config=dict(config_sha256='abc',teacher_artifacts=dict(sha256='teacher'),checkpoint=dict(path='missing'))
            with patch.object(fitting,'load_model_checkpoint',side_effect=RuntimeError('fault')):
                with self.assertRaisesRegex(RuntimeError,'fault'):fitting.execute_fit(config,[],None,None,output,None,lambda:None)
                report=json.loads((output/'report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['status'],'incomplete');self.assertFalse(report['training_performed'])
                with self.assertRaises(FileExistsError):fitting.execute_fit(config,[],None,None,output,None,lambda:None)


if __name__=='__main__':unittest.main()
