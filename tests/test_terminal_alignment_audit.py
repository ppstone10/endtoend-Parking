import unittest
import numpy as np
from experiments.terminal_alignment_audit import analyze_terminal_alignment


class TerminalAlignmentTests(unittest.TestCase):
    def fixture(self):
        points=np.array([[[2,0,0],[0,0,0],[0,0,0]],[[2,0,0],[0,0,0],[2,0,0]],[[2,0,0],[2,0,0],[2,0,0]]],dtype=float)
        targets=np.zeros_like(points);masks=np.ones((3,3));goals=np.zeros((3,3));logits=np.array([[-5,5,5],[5,5,5],[-5,-5,-5]],dtype=float)
        metadata=[{'task_id':str(i),'scene_name':'S4'} for i in range(3)]
        return points,logits,targets,masks,goals,metadata

    def test_three_categories_and_oracle_is_not_deployment(self):
        rows,summary=analyze_terminal_alignment(*self.fixture(),threshold=.5)
        self.assertEqual([r['category'] for r in rows],['deployment_joint_valid','stop_misaligned_with_joint_point','no_joint_point'])
        self.assertEqual(rows[1]['stop_timing'],'early');self.assertFalse(rows[2]['stop_found'])
        self.assertEqual(summary['samples'],3);self.assertEqual(sum(summary['categories'].values()),3)
        self.assertEqual(summary['by_scene']['S4']['categories'],summary['categories'])

    def test_wrap_boundary_threshold_and_target_index(self):
        args=list(self.fixture());args[0][:,:,2]=2*np.pi;args[3][0]=[1,1,0];args[1][0]=[0,5,5]
        rows,_=analyze_terminal_alignment(*args,threshold=.5)
        self.assertEqual(rows[0]['stop_length'],1);self.assertEqual(rows[0]['target_length'],2)
        self.assertTrue(rows[0]['teacher_index_prediction']['joint_valid'])

    def test_finite_prefix_shape_and_threshold_guards(self):
        for index,value in [(0,np.full((3,3,3),np.nan)),(3,np.array([[1,0,1]]*3)),(4,np.zeros((1,3)))]:
            args=list(self.fixture());args[index]=value
            with self.assertRaises(ValueError):analyze_terminal_alignment(*args,threshold=.5)
        with self.assertRaises(ValueError):analyze_terminal_alignment(*self.fixture(),threshold=1.)

    def test_late_and_inside_gap_are_not_merged(self):
        args=list(self.fixture());args[0][1]=[[0,0,0],[2,0,0],[0,0,0]];args[1][1]=[-5,5,5]
        rows,_=analyze_terminal_alignment(*args,threshold=.5);self.assertEqual(rows[1]['stop_timing'],'inside_gap')
        args[0][1]=[[0,0,0],[2,0,0],[2,0,0]]
        rows,_=analyze_terminal_alignment(*args,threshold=.5);self.assertEqual(rows[1]['stop_timing'],'late')

    def test_group_denominators_and_source_indices(self):
        args=list(self.fixture());args[5][2]['scene_name']='S8'
        rows,summary=analyze_terminal_alignment(*args,threshold=.5,indices=[20,30,40])
        self.assertEqual([r['dataset_index'] for r in rows],[20,30,40])
        self.assertEqual(sum(g['samples'] for g in summary['by_scene'].values()),3)
        with self.assertRaises(ValueError):analyze_terminal_alignment(*args,threshold=.5,indices=[20,20,40])

    def test_cli_identity_and_existing_output_guards(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from scripts.audit_terminal_alignment import load_config,run
        from scripts.prepare_recovery_balance_ab import file_digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.write_bytes(b'frozen');path=root/'config.json'
            config={'kind':'train_terminal_alignment_audit','code':{},'runtime':{},'expected_samples':2400,'batch_size':8,
                    'inputs':{'data':{'path':str(source),'sha256':file_digest(source)}},'output':str(root/'results')}
            path.write_text(json.dumps(config),encoding='utf-8');sha=file_digest(path)
            with patch('scripts.audit_terminal_alignment.execution_identity',return_value={}),patch('scripts.audit_terminal_alignment.runtime_identity',return_value={}):
                self.assertEqual(load_config(path,sha),config)
                with self.assertRaises(ValueError):load_config(path,'bad')
                (root/'results').mkdir()
                with self.assertRaises(FileExistsError):run(path,sha)
                source.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError,'输入身份'):load_config(path,sha)
