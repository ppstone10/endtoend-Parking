import unittest
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from experiments.s8_preservation_learning import batch_loss,preservation_objective,preservation_schedule,fit_preservation
from experiments.s6_recovery_learning import batch_identity
from tests.test_s6_policy_transfer import Student,sample


class DropoutStudent(Student):
    def __init__(self):
        super().__init__();self.dropout=torch.nn.Dropout(.5)
    def forward_with_stop(self,bev,goal,state):
        out=super().forward_with_stop(bev,goal,state)
        out.points=self.dropout(out.points)
        return out


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.base=[sample(i+1,'anchor' if i<2 else 'recovery',100+i) for i in range(6)]
        self.queries=[sample(i+10,'anchor',2338) for i in range(2)]

    def test_primary_equivalence_and_aux_keeps_rng_mode_and_gradient(self):
        model=DropoutStudent().train();torch.manual_seed(123);before=torch.get_rng_state()
        original=batch_loss(model,self.base);after=torch.get_rng_state()
        torch.set_rng_state(before)
        total,primary,keep=preservation_objective(model,self.base,self.queries)
        self.assertTrue(torch.equal(original,primary))
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertTrue(model.training);self.assertTrue(keep.requires_grad)
        self.assertTrue(torch.equal(total,primary+.25*keep))
        zero,_,_=preservation_objective(model,self.base,self.queries,weight=0)
        self.assertTrue(torch.isfinite(zero))

    def test_schedule_covers_all_queries_balances_sources(self):
        pool={source:[sample(f+source,'anchor',source) for f in range(1,4)] for source in (2338,2351,2375)}
        schedule=preservation_schedule(pool,20261020)
        flat=[s for b in schedule for s in b]
        self.assertEqual(len(flat),56)
        self.assertEqual(len({(s['source_index'],s['frame_index']) for s in flat}),9)
        self.assertLessEqual(max(sum(s['source_index']==i for s in flat) for i in pool)-min(sum(s['source_index']==i for s in flat) for i in pool),1)
        with self.assertRaises(ValueError):preservation_schedule({2338:pool[2338][:2]},20261020)

    def test_real_adam_steps_and_preserved_actual_identities(self):
        model=Student();opt=torch.optim.Adam(model.parameters(),lr=1e-5)
        opt.state[model.value]=dict(step=torch.tensor(240.),exp_avg=torch.zeros_like(model.value),exp_avg_sq=torch.zeros_like(model.value))
        base=[self.base]*28;queries=[self.queries]*28
        bp=[batch_identity(b) for b in base];qp=[batch_identity(q) for q in queries]
        trained,optimizer,history,presented=fit_preservation(model,opt.state_dict(),base,queries,bp,qp,20261020,lambda:None)
        self.assertEqual(len(history),28);self.assertEqual(len(presented),28)
        self.assertEqual({int(s['step']) for s in optimizer.state.values()},{268})
        self.assertEqual(trained.stop_threshold,model.stop_threshold)
        broken=deepcopy(qp);broken[0][0]['input_sha256']='drift';updates=[]
        with self.assertRaises(ValueError):
            fit_preservation(model,opt.state_dict(),base,queries,bp,broken,20261020,lambda:None,updates.append)
        self.assertEqual(updates,[])

    def test_cli_precheck_no_training_and_failure_no_retry(self):
        from scripts import preflight_s8_preservation_learning as cli
        with patch.object(cli,'prepare_context',return_value={}),patch.object(cli,'fit_preservation') as fit, \
                patch.object(cli,'evaluate_cases') as evaluate:
            cli.precheck();fit.assert_not_called();evaluate.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'new'
            with patch.object(cli,'prepare_context',side_effect=TimeoutError('budget')),patch.object(cli,'fit_preservation') as fit:
                with self.assertRaises(TimeoutError):cli.run(target)
                fit.assert_not_called()
            r=json.loads((target/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(r['status'],'budget_exhausted');self.assertFalse(r['allowed_to_prepare_training_plan'])
            self.assertEqual(r['actual_updates'],{})
            with self.assertRaises(FileExistsError):cli.run(target)


if __name__=='__main__':unittest.main()
