"""STOP-ACCEPT: acceptance, not the earliest eligible point, owns success."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from scripts.accepted_native_objectives import acceptance_stop_loss, qualified_points, objective
from experiments.emitted_output_joint import native_lengths


class AcceptedStopTests(unittest.TestCase):
    def test_accepted_later_point_has_zero_loss_and_gradient(self):
        for arm in ('gate','accept'):
            logits=torch.tensor([[-14.,-2.,1.]],requires_grad=True)
            loss,rows=acceptance_stop_loss(logits,torch.tensor([[True,True,True]]),arm)
            loss.backward()
            self.assertEqual(float(loss.detach()),0);self.assertFalse(logits.grad.any())
            self.assertEqual(rows[0]['status'],'accepted')

    def test_early_wrong_stop_is_suppressed(self):
        logits=torch.tensor([[1.,-2.,2.]],requires_grad=True)
        loss,rows=acceptance_stop_loss(logits,torch.tensor([[False,False,True]]),'accept')
        loss.backward();self.assertGreater(float(logits.grad[0,0]),0)
        self.assertEqual(rows[0]['status'],'early');self.assertGreater(float(loss.detach()),0)

    def test_late_stop_promotes_a_valid_earlier_point(self):
        logits=torch.tensor([[-2.,-1.,2.]],requires_grad=True)
        loss,rows=acceptance_stop_loss(logits,torch.tensor([[False,True,False]]),'accept')
        loss.backward();self.assertLess(float(logits.grad[0,1]),0)
        self.assertEqual(rows[0]['status'],'late_or_gap')

    def test_set_target_does_not_force_earliest_valid_point(self):
        logits=torch.tensor([[2.,-4.,-2.,1.]],requires_grad=True)
        mask=torch.tensor([[False,True,False,True]])
        loss,rows=acceptance_stop_loss(logits,mask,'accept')
        loss.backward();self.assertEqual(rows[0]['selected_index'],3)
        self.assertEqual(float(logits.grad[0,1]),0)
        self.assertGreater(float(logits.grad[0,0]),0)

    def test_empty_set_is_explicit_and_keeps_autograd(self):
        logits=torch.zeros(1,4,requires_grad=True)
        loss,rows=acceptance_stop_loss(logits,torch.zeros(1,4,dtype=torch.bool),'accept')
        loss.backward();self.assertEqual(rows[0]['status'],'no_window')
        self.assertEqual(float(loss.detach()),0);self.assertFalse(logits.grad.any())

    def test_no_crossing_fallback_can_be_accepted(self):
        loss,rows=acceptance_stop_loss(torch.full((1,3),-2.),torch.tensor([[False,False,True]]),'accept')
        self.assertEqual(float(loss),0);self.assertEqual(rows[0]['status'],'accepted')

    def test_gate_keeps_old_unaccepted_window_exactly(self):
        from scripts.native_output_objectives import stop_window_loss
        logits=torch.tensor([[2.,-4.,-2.,1.]])
        mask=torch.tensor([[False,True,False,True]])
        loss,rows=acceptance_stop_loss(logits,mask,'gate')
        self.assertEqual(float(loss),float(stop_window_loss(logits,[1])))
        self.assertEqual(rows[0]['selected_index'],1)

    def test_zero_loss_on_covered_inputs_implies_native_acceptance(self):
        rng=torch.Generator().manual_seed(44)
        logits=torch.randn(300,7,generator=rng)*3
        mask=torch.rand(300,7,generator=rng)>.5
        for arm in ('gate','accept'):
            for row,valid in zip(logits,mask):
                loss,_=acceptance_stop_loss(row[None],valid[None],arm)
                if valid.any() and float(loss)==0:
                    self.assertTrue(valid[int(native_lengths(row[None])[0])-1])

    def test_invalid_inputs_are_rejected(self):
        for logits,mask,arm in [(torch.zeros(1,2),torch.zeros(1,2),'accept'),
            (torch.tensor([[float('nan')]]),torch.ones(1,1,dtype=torch.bool),'accept'),
            (torch.zeros(1,2),torch.ones(1,1,dtype=torch.bool),'accept'),
            (torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool),'AB')]:
            with self.assertRaises(ValueError):acceptance_stop_loss(logits,mask,arm)


class AcceptedGeometryTests(unittest.TestCase):
    def test_strict_pose_boundaries_and_wrapped_yaw(self):
        from tests.test_native_output_contract import geometry
        sample=dict(goal=torch.zeros(3,dtype=torch.float64),tolerances=torch.tensor([.3,.2],dtype=torch.float64),pose=np.zeros(3))
        points=torch.tensor([[.3,0.,0.],[0.,0.,.2],[.2,0.,2*np.pi+.1]],dtype=torch.float64)
        self.assertEqual(qualified_points(points,sample,geometry()).tolist(),[False,False,True])

    def test_dangerous_bridge_invalidates_later_goal(self):
        from tests.test_native_output_contract import geometry
        from sim import RectangleObstacle
        sample=dict(goal=torch.zeros(3),tolerances=torch.tensor([.3,.2]),pose=np.zeros(3))
        points=torch.tensor([[4.,0.,0.],[0.,0.,0.]])
        geo=geometry([RectangleObstacle(5,6,-3,3)])
        self.assertEqual(qualified_points(points,sample,geo).tolist(),[False,False])

    def test_empty_set_keeps_all_AB_coordinate_geometry_terms(self):
        from tests.test_native_output_contract import geometry,TinyNetwork,tiny_sample
        from scripts.native_output_objectives import objective as old_objective
        sample=tiny_sample();model=TinyNetwork()
        cases={0:dict(planner=type('Planner',(),{'_collision_checker':geometry().checker})())}
        old_loss,old=old_objective(model,[sample],cases,'AB',1,1)
        for arm in ('gate','accept'):
            model.zero_grad(set_to_none=True)
            loss,record=objective(model,[sample],cases,arm,1,1)
            self.assertEqual(record['stop_deferred'],1)
            for key in ('imitation_loss','native_pose_loss','geometry_loss','geometry_native_lengths'):
                self.assertEqual(record[key],old[key])
            self.assertEqual(float(loss.detach()),float(old_loss.detach()))
            loss.backward();self.assertGreater(float(model.decoder.point_head.weight.grad.abs().sum()),0)
            self.assertFalse(model.decoder.stop_head.weight.grad.any())

    def test_sigmoid_rounding_uses_original_native_selector(self):
        logits=torch.tensor([[-1e-9,-1.]])
        self.assertEqual(int(native_lengths(logits)[0]),1)
        loss,rows=acceptance_stop_loss(logits,torch.tensor([[True,False]]),'accept')
        self.assertEqual(float(loss),0);self.assertEqual(rows[0]['status'],'accepted')


class AcceptedStudyTests(unittest.TestCase):
    def test_only_both_final_accept_seeds_can_pass(self):
        from scripts.accepted_native_fit import decision
        rows=[dict(seed=s,arm=a,final=dict(passed=True)) for s in (20261028,20261029) for a in ('gate','accept')]
        self.assertTrue(decision(rows)['passed'])
        rows[-1]['final']['passed']=False
        self.assertFalse(decision(rows)['passed'])
        with self.assertRaises(ValueError):decision(rows[:-1])
        self.assertFalse(decision(rows)['candidate_qualified'])

    def test_gains_do_not_hide_old_input_losses(self):
        from scripts.accepted_native_fit import comparison
        before=dict(rows=[dict(source_index=1,frame_index=0,passed=True)])
        after=dict(rows=[dict(source_index=1,frame_index=0,passed=False),dict(source_index=2,frame_index=0,passed=True)])
        row=comparison([dict(seed=1,arm='accept',final=after)],[dict(seed=1,arm='LT43_AB',final=before)])[0]
        self.assertEqual(row['lost'],[[1,0]]);self.assertEqual(row['gained'],[[2,0]])

    def test_authorization_before_optimizer_and_output(self):
        from scripts import train_accepted_native_stopping as driver
        with TemporaryDirectory() as directory:
            output=Path(directory)/'fit'
            with (patch.object(driver,'precheck',return_value=dict(source={})),
                  patch.object(driver,'supervision',side_effect=ValueError('blocked')),
                  patch.object(driver.fitting,'execute') as execute,patch('torch.optim.Adam') as optimizer):
                with self.assertRaises(ValueError):driver.fit('prepared',output)
                execute.assert_not_called();optimizer.assert_not_called();self.assertFalse(output.exists())

    def test_prepared_identity_drift_and_overwrite_rejected(self):
        from scripts import train_accepted_native_stopping as driver
        with TemporaryDirectory() as directory:
            target=Path(directory)/'prepared'
            with patch.object(driver,'frozen_config',return_value=dict(source='one')):
                driver.prepare(target)
                with self.assertRaises(FileExistsError):driver.prepare(target)
                self.assertEqual(driver.precheck(target),dict(source='one'))
            with patch.object(driver,'frozen_config',return_value=dict(source='two')):
                with self.assertRaises(ValueError):driver.precheck(target)

    def test_incomplete_fit_kept_and_never_resumed(self):
        from scripts import accepted_native_fit as fitting
        import json
        with TemporaryDirectory() as directory:
            output=Path(directory)/'fit';config=dict(config_sha256='hash',checkpoint=dict(path='missing'))
            with patch.object(fitting,'baseline',return_value=[]),patch.object(fitting,'load_model_checkpoint',side_effect=ValueError('broken')):
                with self.assertRaises(ValueError):fitting.execute(config,[],None,None,output,None,lambda:None)
                self.assertEqual(json.loads((output/'report.json').read_text(encoding='utf-8'))['status'],'incomplete')
                with self.assertRaises(FileExistsError):fitting.execute(config,[],None,None,output,None,lambda:None)

    def test_artifact_count_and_checkpoint_contract(self):
        from scripts.accepted_native_fit import expected_files
        names=expected_files()
        self.assertEqual(len(names),56);self.assertEqual(len(set(names)),56)
        self.assertEqual(sum(n.endswith('.pt') for n in names),16)
        self.assertEqual(sum(n.endswith('.npz') for n in names),16)

    def test_coverage_counts_exposures_and_deferred_separately(self):
        from scripts.accepted_native_fit import coverage
        row=dict(samples=[dict(source_index=i,frame_index=0) for i in range(8)],
            stop_rows=[dict(status='no_window',native_length=3,eligible_count=0) for _ in range(8)],
            stop_deferred=8,already_accepted=0)
        report=coverage([row,row]);self.assertEqual(report['0:0']['no_window'],2)
        self.assertEqual(report['0:0']['exposures'],2)
        row['stop_deferred']=0
        with self.assertRaises(ValueError):coverage([row])


    def test_synthetic_fit_inspect_roundtrip_rejects_rehashed_checkpoint_identity(self):
        # Tiny synthetic tensors exercise serialization/audit, not parking training.
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        import io
        from scripts import accepted_native_fit as fitting
        from scripts.train_emitted_output_joint import StageBudget
        from experiments.emitted_output_data import INDICES,read_json,file_digest
        from training.reporting import atomic_write_json

        samples=[dict(source_index=i,frame_index=t) for i in INDICES for t in range(4)]
        def load(path):
            model=torch.nn.Linear(1,1)
            with torch.no_grad():model.weight.fill_(1);model.bias.fill_(0)
            if Path(path).exists():model.load_state_dict(torch.load(path,weights_only=False)['model_state'])
            return SimpleNamespace(model=model,model_name='synthetic',model_config={})
        def evaluate(model,selected,data,cases,path):
            report=dict(initial_passed=8,all_frames_passed=32,passed=True,
                rows=[dict(**s,passed=True) for s in selected])
            atomic_write_json(Path(path).with_suffix('.json'),report)
            np.savez(Path(path).with_suffix('.npz'),points=np.zeros((32,2,3)),logits=np.ones((32,2)))
            return report
        def update(model,optimizer,batch,cases,arm,seed,step):
            optimizer.zero_grad();loss=model.weight.sum()+model.bias.sum();loss.backward();optimizer.step()
            return dict(step=step,samples=batch,total_loss=float(loss.detach()),
                stop_rows=[dict(status='accepted',native_length=1,eligible_count=1) for _ in batch],
                stop_deferred=0,already_accepted=8,head_gradient_norms=[1.,1.],unclipped_gradient_norm=1.)
        config=dict(config_sha256='synthetic-hash',checkpoint=dict(path='not-a-real-parent'))
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output=Path(directory)/'fit'
            with (patch.object(fitting,'baseline',return_value=[]),patch.object(fitting,'load_model_checkpoint',side_effect=load),
                  patch.object(fitting,'native_evaluate',side_effect=evaluate),patch.object(fitting,'check_origin'),
                  patch.object(fitting,'update',side_effect=update)):
                fitting.execute(config,samples,None,None,output,StageBudget(10),lambda:None)
                verified=fitting.inspect(config,samples,None,None,output,StageBudget(10))
                self.assertEqual(verified['native_evaluations'],512)
                self.assertEqual(verified['checkpoints'],16)
                self.assertFalse(verified['gate']['candidate_qualified'])
                path=output/'20261028-gate/update-0050.pt'
                payload=torch.load(path,weights_only=False);payload['updates']=51;torch.save(payload,path)
                manifest=read_json(output/'artifacts.json');manifest[str(path.resolve())]=file_digest(path)
                atomic_write_json(output/'artifacts.json',manifest)
                with self.assertRaisesRegex(ValueError,'checkpoint identity'):
                    fitting.inspect(config,samples,None,None,output,StageBudget(10))

if __name__=='__main__':unittest.main()
