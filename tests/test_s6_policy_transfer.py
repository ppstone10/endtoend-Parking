import unittest
from copy import deepcopy
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from experiments.s6_policy_transfer import build_transfer_schedules, fit_transfer, validate_policy_samples
from experiments.s6_recovery_learning import batch_identity


def sample(frame=1, role='recovery', source=2149):
    return dict(bev=torch.full((1, 2, 2), float(frame)), goal=torch.zeros(3), state=torch.zeros(2),
                points=torch.ones(320, 3), stop=torch.full((320,), .4), mask=(torch.arange(320)<5).float(),
                role=role, source_index=source, frame_index=frame, query_step=frame*10,
                origin='validated_success_policy', query_policy='chain120-bias-noncandidate-2149',
                teacher_checkpoint_sha256='teacher')


class Teacher:
    def eval(self): return self
    def forward_with_stop(self, bev, goal, state):
        n=len(bev)
        return SimpleNamespace(points=torch.zeros(n,320,3), stop_logits=torch.zeros(n,320))


class Student(torch.nn.Module):
    stop_threshold=.26
    def __init__(self):
        super().__init__(); self.value=torch.nn.Parameter(torch.tensor(.1))
    def forward_with_stop(self,bev,goal,state):
        return SimpleNamespace(points=self.value.expand(len(bev),320,3),stop_logits=self.value.expand(len(bev),320))


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.samples=[sample(i) for i in range(1,54)]
        self.base=[[sample(i+1,'anchor',100),sample(i+1,'anchor',101),
                    sample(i+1,'recovery',102),sample(i+1,'recovery',103)] for i in range(24)]

    def test_pair_keeps_all_legacy_and_covers_all_new(self):
        plans,audit=build_transfer_schedules(self.base,self.samples,Teacher(),20261020)
        self.assertEqual(audit['unique_policy_inputs'],53)
        self.assertEqual(audit['policy_slots'],56)
        for j,(left,right) in enumerate(zip(plans['self_targets'],plans['policy_targets'])):
            self.assertEqual(len(left),6)
            self.assertEqual(batch_identity(left[:4]),batch_identity(self.base[j%24]))
            self.assertEqual(batch_identity(left[:4]),batch_identity(right[:4]))
            for a,b in zip(left[4:],right[4:]):
                self.assertTrue(torch.equal(a['bev'],b['bev']))
                self.assertTrue(torch.equal(a['mask'],b['mask']))
                self.assertFalse(torch.equal(a['points'],b['points']))

    def test_second_order_deterministic_and_different(self):
        a,_=build_transfer_schedules(self.base,self.samples,Teacher(),20261021)
        b,_=build_transfer_schedules(self.base,self.samples,Teacher(),20261021)
        c,_=build_transfer_schedules(self.base,self.samples,Teacher(),20261020)
        ids=lambda p:[batch_identity(x) for x in p['policy_targets']]
        self.assertEqual(ids(a),ids(b)); self.assertNotEqual(ids(a),ids(c))

    def test_identity_source_phase_and_nonfinite_rejected(self):
        for field,value in [('source_index',2280),('frame_index',0),('query_step',11),
                            ('origin','planner'),('teacher_checkpoint_sha256','other')]:
            bad=deepcopy(self.samples); bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_policy_samples(bad,'teacher')
        bad=deepcopy(self.samples); bad[0]['stop'][0]=float('nan')
        with self.assertRaises(ValueError): validate_policy_samples(bad,'teacher')

    def test_missing_duplicate_and_bad_base_rejected(self):
        for values in (self.samples[:-1],self.samples[:-1]+[self.samples[0]]):
            with self.assertRaises(ValueError): validate_policy_samples(values,'teacher')
        with self.assertRaises(ValueError):
            build_transfer_schedules(self.base[:-1],self.samples,Teacher(),20261020)

    def test_real_optimizer_count_and_label_drift_before_update(self):
        model=Student(); opt=torch.optim.Adam(model.parameters(),lr=1e-5)
        opt.state[model.value]=dict(step=torch.tensor(240.),exp_avg=torch.zeros_like(model.value),
                                   exp_avg_sq=torch.zeros_like(model.value))
        plans,_=build_transfer_schedules(self.base,self.samples,Teacher(),20261020)
        schedule=plans['policy_targets']; planned=[batch_identity(b) for b in schedule]; updates=[]
        trained,optimizer,history,presented=fit_transfer(model,opt.state_dict(),schedule,planned,
            20261020,lambda:None,updates.append)
        self.assertEqual(len(history),28); self.assertEqual(presented,planned)
        self.assertEqual(updates,list(range(1,29)))
        self.assertEqual({int(s['step']) for s in optimizer.state.values()},{268})
        self.assertEqual(trained.stop_threshold,.26)
        broken=deepcopy(planned); broken[0][0]['label_sha256']='drift'; updates.clear()
        with self.assertRaises(ValueError):
            fit_transfer(model,opt.state_dict(),schedule,broken,20261020,lambda:None,updates.append)
        self.assertEqual(updates,[])

    def test_budget_failure_retains_report_and_does_not_fit(self):
        from scripts import preflight_s6_policy_transfer as cli
        with tempfile.TemporaryDirectory() as folder:
            cfg=Path(folder)/'prepared.json'; cfg.write_text('{}',encoding='utf-8')
            output=Path(folder)/'new'
            with patch.object(cli,'verify_config'),patch.object(cli,'prepare_context',side_effect=TimeoutError('budget')), \
                    patch.object(cli,'fit_transfer') as fit:
                with self.assertRaises(TimeoutError): cli.run(cfg,output)
                fit.assert_not_called()
            report=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertFalse(report['training_performed']); self.assertFalse(report['allowed_to_prepare_training_plan'])
            with patch.object(cli,'verify_config'):
                with self.assertRaises(FileExistsError): cli.run(cfg,output)

    def test_precheck_has_no_updates_or_rollouts(self):
        from scripts import preflight_s6_policy_transfer as cli
        with patch.object(cli,'prepare_context',return_value={'config':{}}), \
                patch.object(cli,'fit_transfer') as fit,patch.object(cli,'evaluate_cases') as evaluate:
            cli.precheck(); fit.assert_not_called(); evaluate.assert_not_called()

    def test_complete_orchestration_four_models_with_json_seed_keys(self):
        from scripts import preflight_s6_policy_transfer as cli
        seeds=(20261020,20261021); arms=('self_targets','policy_targets')
        cfg=dict(parent_checkpoints={str(s):'parent' for s in seeds},probe_indices=[8],fit_indices=[2149],
            planned_batch_identities={a:{s:[] for s in seeds} for a in arms})
        rows=[{'dataset_index':8}]
        old=dict(r0_probe=rows,parent_probe={str(s):rows for s in seeds},
            completed_arms={'flat_old':{str(s):{'probe':rows} for s in seeds}})
        context=dict(config=cfg,plans={a:{s:[] for s in seeds} for a in arms},parent=Student(),previous=old,
            consumption={'completed_seeds':{str(s):{'probe':rows} for s in seeds}},parent_fit={s:rows for s in seeds})
        optimizer=torch.optim.Adam(context['parent'].parameters())
        def fit(*args):
            args[-1](28)
            return Student(),optimizer,[0.]*28,[]
        with tempfile.TemporaryDirectory() as folder:
            prepared=Path(folder)/'prepared.json'; prepared.write_text(json.dumps(cfg),encoding='utf-8')
            with patch.object(cli,'verify_config'),patch.object(cli,'prepare_context',return_value=context), \
                    patch.object(cli.DatasetGenerator,'load'),patch.object(cli.matrix,'load_dataset_manifest',return_value={'vehicle_model':{}}), \
                    patch.object(cli,'VehicleConfig'),patch.object(cli,'build_case'), \
                    patch.object(cli,'load_candidate',return_value=(Student(),{})), \
                    patch.object(cli,'fit_transfer',side_effect=fit) as fitter, \
                    patch.object(cli,'evaluate_cases',return_value=rows) as evaluator, \
                    patch.object(cli,'recovery_qualification',return_value={'passed':False}):
                report=cli.run(prepared,Path(folder)/'output')
            self.assertEqual(fitter.call_count,4); self.assertEqual(evaluator.call_count,8)
            self.assertEqual(report['status'],'completed'); self.assertEqual(len(report['actual_updates']),4)
            self.assertFalse(report['allowed_to_prepare_training_plan'])


if __name__=='__main__': unittest.main()
