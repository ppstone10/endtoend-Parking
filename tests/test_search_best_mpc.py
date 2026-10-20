"""当前调用最佳样本实验，不等同于安全优化。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller import MPCController
from experiments.mpc_search_best_solver import SearchBestMPC
from interfaces import VehicleState
from scripts import compare_search_best_mpc as comparison


class SearchBestTests(unittest.TestCase):
    def test_identical_populations_rng_and_cost_dominance(self):
        for seed in range(6):
            base=MPCController(seed=seed);trial=SearchBestMPC(seed=seed)
            state=VehicleState(0.,.1,.05)
            ref=np.column_stack((np.linspace(0.,1.,10),np.zeros((10,2))))
            warm=np.tile([.4,-.1],10);saved=warm.copy()
            populations=[];costs=[];base_cost=base._batch_cost
            def record(samples,reference,pose):
                values=base_cost(samples,reference,pose)
                populations.append(samples.copy());costs.append(values.copy())
                return values
            with patch.object(base,'_batch_cost',side_effect=record):
                legacy=base._solve(warm,ref,state)
            trial_cost=trial._batch_cost;seen=[]
            def observe(samples,reference,pose):
                seen.append(samples.copy())
                return trial_cost(samples,reference,pose)
            with patch.object(trial,'_batch_cost',side_effect=observe):
                selected=trial._solve(warm,ref,state)
            self.assertEqual(len(seen),base.iterations+1)
            for actual,expected in zip(seen[:-1],populations):np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(seen[-1][0],legacy)
            self.assertEqual(base._rng.bit_generator.state,trial._rng.bit_generator.state)
            np.testing.assert_array_equal(warm,saved)
            selected_cost=float(base_cost(selected[None],ref,state)[0])
            self.assertAlmostEqual(selected_cost,min(float(base_cost(legacy[None],ref,state)[0]),min(float(x.min()) for x in costs)))
            self.assertIn(trial.last_search_audit['selected'],('legacy_mean','search_sample'))
            self.assertTrue(np.all(abs(selected[::2])<=base.max_v))
            self.assertTrue(np.all(abs(selected[1::2])<=base.max_omega))

    def test_tie_prefers_original_mean(self):
        kwargs=dict(seed=3,w_xy=0.,w_yaw=0.,w_terminal=0.,w_v=0.,w_omega=0.)
        base=MPCController(**kwargs);trial=SearchBestMPC(**kwargs)
        ref=np.zeros((10,3));warm=np.zeros(20);state=VehicleState(0.,0.,0.)
        np.testing.assert_array_equal(base._solve(warm,ref,state),trial._solve(warm,ref,state))
        self.assertEqual(trial.last_search_audit['selected'],'legacy_mean')

    def test_nonfinite_input_and_cost_rejected(self):
        trial=SearchBestMPC()
        with self.assertRaises(ValueError):trial._solve(np.full(20,np.nan),np.zeros((10,3)),VehicleState(0.,0.,0.))
        with patch.object(trial,'_batch_cost',return_value=np.full(trial.population,np.inf)):
            with self.assertRaises(ValueError):trial._solve(np.zeros(20),np.zeros((10,3)),VehicleState(0.,0.,0.))


class SearchBestEntryTests(unittest.TestCase):
    def test_scope_is_explicit_and_restored(self):
        previous=comparison.matrix.MPCController,comparison.matrix.ClosedLoopEngine
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                with comparison.solver_scope('candidate',directory):
                    self.assertIs(comparison.matrix.MPCController,SearchBestMPC)
                    self.assertIsNot(comparison.matrix.ClosedLoopEngine,previous[1])
                    raise RuntimeError('test')
        self.assertEqual((comparison.matrix.MPCController,comparison.matrix.ClosedLoopEngine),previous)

    def test_config_tamper_and_output_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.txt';source.write_text('input',encoding='utf-8')
            config={'kind':'offline_mpc_search_best_comparison','solver_variant':'current_search_best',
                    'runtime':comparison.runtime_identity(),'code':comparison.execution_identity(),
                    'inputs':{'one':{'path':str(source),'sha256':comparison.file_digest(source)}}}
            path=root/'config.json';path.write_text(json.dumps(config),encoding='utf-8');h=comparison.file_digest(path)
            comparison.load_config(path,h)
            (root/'results').mkdir()
            with self.assertRaises(FileExistsError):comparison.run(path,h)
            source.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError):comparison.load_config(path,h)

    def test_audit_hash_identity_and_cost_validation(self):
        episode={'meta':{'dataset_index':1,'task_id':'id-1'},'steps':1}
        report={'status':'completed','reconstruct_failures':[],'experiment':{'name':'E5','safety_mode':'none'},'episodes':[episode]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'audits').mkdir();path=root/'audits/task-0001.json'
            row={'step':0,'selected':'search_sample','selected_cost':1.,'legacy_mean_cost':2.,'best_sample_cost':1.}
            ledger={'dataset_index':1,'task_id':'id-1','solver_class':'SearchBestMPC','steps':1,'records':[row]}
            path.write_text(json.dumps(ledger),encoding='utf-8')
            result=comparison.audit_outputs(report,root,[1])
            self.assertEqual(result['1']['sha256'],comparison.file_digest(path))
            row['selected_cost']=3.
            path.write_text(json.dumps(ledger),encoding='utf-8')
            with self.assertRaises(ValueError):comparison.audit_outputs(report,root,[1])


if __name__=='__main__':unittest.main()
