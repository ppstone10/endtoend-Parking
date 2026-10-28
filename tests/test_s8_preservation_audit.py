import unittest
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from experiments.s8_preservation_audit import comparison_metrics, gradient_relation, temporal_clean_queries


def sample(frame=1):
    return dict(bev=torch.zeros(1,2,2),goal=torch.tensor([float(frame),0.,0.]),state=torch.zeros(2),
        points=torch.zeros(320,3),stop=torch.full((320,),.5),mask=(torch.arange(320)<3).float(),
        frame_index=frame,source_index=2330,role='anchor')


class AuditTests(unittest.TestCase):
    def test_self_metrics_zero_and_wrapped_yaw(self):
        s=sample();out=SimpleNamespace(points=s['points'][None].clone(),stop_logits=torch.zeros(1,320))
        row=comparison_metrics(out,s,.26)
        self.assertAlmostEqual(row['pose_loss'],0)
        self.assertAlmostEqual(row['stop_kl'],0)
        out.points[...,2]=2*torch.pi
        self.assertLess(comparison_metrics(out,s,.26)['pose_loss'],1e-10)

    def test_gradient_zero_not_reported_conflict(self):
        p=torch.nn.Parameter(torch.tensor(2.))
        row=gradient_relation(p*p,-p*p,[p])
        self.assertAlmostEqual(row['cosine'],-1.)
        zero=gradient_relation(p*0,p*p,[p])
        self.assertIsNone(zero['cosine'])
        self.assertIsNone(p.grad)

    def test_safety_and_nonzero_before_temporal_selection(self):
        samples=[sample(i) for i in range(7)]
        rows=[dict(frame_index=i,clean=i!=2) for i in range(7)]
        selected,audit=temporal_clean_queries(samples,rows,source_success=True)
        self.assertTrue(audit['passed'])
        self.assertEqual([s['frame_index'] for s in selected],[1,4,6])
        selected,audit=temporal_clean_queries(samples,rows,source_success=False)
        self.assertEqual(selected,[]); self.assertFalse(audit['passed'])

    def test_insufficient_or_misaligned_windows_fail_closed(self):
        samples=[sample(1),sample(2)]
        selected,audit=temporal_clean_queries(samples,[dict(frame_index=i,clean=True) for i in (1,2)],source_success=True)
        self.assertEqual(selected,[]);self.assertFalse(audit['passed'])
        with self.assertRaises(ValueError): temporal_clean_queries(samples,[],source_success=True)

    def test_cli_gate_failure_has_no_rollout_or_bank(self):
        from scripts import audit_s8_preservation as cli
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'new'
            with patch.object(cli,'prepare_context',side_effect=ValueError('identity')),patch.object(cli,'run_case') as run:
                with self.assertRaises(ValueError):cli.run(target)
                run.assert_not_called()
            r=json.loads((target/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(r['status'],'failed');self.assertEqual(r['training_updates'],0)
            self.assertFalse((target/'queries.pt').exists())


if __name__=='__main__': unittest.main()
