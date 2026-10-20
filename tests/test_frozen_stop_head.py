from copy import deepcopy
import unittest
import numpy as np
import torch

from model import build_model
from model.network import variable_loss_fn
from experiments.frozen_stop_head import capture_stop_features,cumulative_stop_loss,fit_stop_head,backbone_digest,make_head_checkpoint


class FrozenStopHeadTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        model=build_model('net-v1',{'bev_channels':5,'hidden_dim':8,'max_horizon':6,'dt':.2,'stop_threshold':.26})
        batch=(torch.randn(3,5,16,16),torch.randn(3,3),torch.zeros(3,2),torch.randn(3,6,3),torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,1],[1,1,0,0,0,0]],dtype=torch.float32))
        return model,batch

    def test_capture_matches_direct_and_hook_is_removed(self):
        model,batch=self.fixture();saved=deepcopy(model.state_dict())
        for mode in ('guided','free'):
            cache=capture_stop_features(model,[batch],mode)
            expected=model.forward_with_stop(*batch[:3],**({'teacher_points':batch[3],'teacher_forcing_ratio':1.} if mode=='guided' else {}))
            torch.testing.assert_close(cache['logits'],expected.stop_logits.detach(),atol=1e-6,rtol=0)
            torch.testing.assert_close(model.decoder.stop_head(cache['features']).squeeze(-1),cache['logits'],atol=1e-6,rtol=0)
        self.assertFalse(model.decoder.stop_head._forward_pre_hooks)
        self.assertTrue(all(torch.equal(saved[k],v) for k,v in model.state_dict().items()))

    def test_cumulative_loss_matches_existing_objective(self):
        logits=torch.randn(3,6);lengths=torch.tensor([4,6,2]);mask=(torch.arange(6)[None]<lengths[:,None]).float();points=torch.zeros(3,6,3)
        expected=variable_loss_fn(points,logits,points,mask,stop_weight=.2,stop_target_mode='cumulative')
        torch.testing.assert_close(cumulative_stop_loss(logits,lengths),expected)

    def test_only_head_changes_and_training_is_deterministic(self):
        model,batch=self.fixture();cache=capture_stop_features(model,[batch],'free');before=deepcopy(model.state_dict());digest=backbone_digest(model)
        first,history=fit_stop_head(model.decoder.stop_head,cache,epochs=2,batch_size=2,seed=9)
        second,_=fit_stop_head(model.decoder.stop_head,cache,epochs=2,batch_size=2,seed=9)
        self.assertTrue(all(torch.equal(first.state_dict()[k],v) for k,v in second.state_dict().items()))
        self.assertFalse(torch.equal(first.weight,model.decoder.stop_head.weight));self.assertEqual(len(history),2)
        self.assertEqual(backbone_digest(model),digest)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        candidate=deepcopy(model);candidate.decoder.stop_head.load_state_dict(first.state_dict())
        self.assertTrue(torch.equal(model(*batch[:3]),candidate(*batch[:3])))

    def test_checkpoint_is_nonresumable_and_retains_frozen_tensors(self):
        model,batch=self.fixture();cache=capture_stop_features(model,[batch],'free');head,_=fit_stop_head(model.decoder.stop_head,cache,epochs=1)
        parent={'schema_version':1,'model_name':'net-v1','model_config':{'stop_threshold':.26},'model_state':deepcopy(model.state_dict()),'epoch':6,'optimizer_state':{'old':1}}
        checkpoint=make_head_checkpoint(parent,head,epoch=1,metadata={'arm':'free'})
        self.assertFalse(checkpoint['resumable']);self.assertIsNone(checkpoint['optimizer_state']);self.assertEqual(checkpoint['head_adaptation']['source_epoch'],6)
        self.assertEqual(checkpoint['model_config'],parent['model_config'])
        for k,v in parent['model_state'].items():
            if not k.startswith('decoder.stop_head.'):self.assertTrue(torch.equal(v,checkpoint['model_state'][k]))

    def test_guided_features_differ_and_hook_cleanup_on_failure(self):
        from unittest.mock import patch
        model,batch=self.fixture();guided=capture_stop_features(model,[batch],'guided');free=capture_stop_features(model,[batch],'free')
        self.assertFalse(torch.equal(guided['features'],free['features']))
        with patch.object(model,'forward_with_stop',side_effect=RuntimeError('unexpected')):
            with self.assertRaisesRegex(RuntimeError,'unexpected'):capture_stop_features(model,[batch],'free')
        self.assertFalse(model.decoder.stop_head._forward_pre_hooks);self.assertTrue(model.training)

    def test_invalid_cache_and_budgets_fail(self):
        model,batch=self.fixture();cache=capture_stop_features(model,[batch],'free')
        with self.assertRaises(ValueError):fit_stop_head(model.decoder.stop_head,cache,epochs=0)
        with self.assertRaises(ValueError):cumulative_stop_loss(torch.zeros(3,6),torch.tensor([0,6,2]))
        bad=deepcopy(cache);bad['features'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):fit_stop_head(model.decoder.stop_head,bad)

    def test_export_reload_and_metric_source_indices(self):
        import json,tempfile
        from pathlib import Path
        from scripts.train_frozen_stop_head import save_torch,verify_export,evaluate_head
        from training.checkpoint import load_model_checkpoint
        model,batch=self.fixture();cache=capture_stop_features(model,[batch],'free');head,_=fit_stop_head(model.decoder.stop_head,cache,epochs=1)
        config={'bev_channels':5,'hidden_dim':8,'max_horizon':6,'dt':.2,'stop_threshold':.26}
        parent={'schema_version':1,'model_name':'net-v1','model_config':config,'model_state':deepcopy(model.state_dict()),'epoch':6}
        payload=make_head_checkpoint(parent,head,epoch=1,metadata={})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=save_torch(root/'head.pt',payload);loaded=load_model_checkpoint(path).model
            self.assertTrue(verify_export(parent,payload,loaded,batch,cache['points'])['probe_points_bitwise_equal'])
            with self.assertRaises(FileExistsError):save_torch(path,payload)
            meta=[{'task_id':str(i),'scene_name':'S4'} for i in range(3)]
            evaluate_head(head,cache,meta,root,threshold=.26,count=3,source_indices=[10,20,30])
            self.assertEqual([r['dataset_index'] for r in json.loads((root/'samples.json').read_text(encoding='utf-8'))],[10,20,30])

    def test_config_identity_and_output_reuse_guards(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.train_frozen_stop_head import SETTINGS,load_config,run
        from scripts.prepare_recovery_balance_ab import file_digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.write_bytes(b'frozen');path=root/'config.json'
            config={'kind':'frozen_stop_head_guided_free_ab','code':{},'runtime':{},'settings':SETTINGS,'expected_samples':2918,'original_samples':2400,
                    'inputs':{'x':{'path':str(source),'sha256':file_digest(source)}},'output':str(root/'results')}
            path.write_text(json.dumps(config),encoding='utf-8');sha=file_digest(path)
            with patch('scripts.train_frozen_stop_head.execution_identity',return_value={}),patch('scripts.train_frozen_stop_head.runtime_identity',return_value={}):
                self.assertEqual(load_config(path,sha),config)
                with self.assertRaises(ValueError):load_config(path,'bad')
                (root/'results').mkdir()
                with self.assertRaises(FileExistsError):run(path,sha)
                source.write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError,'输入身份'):load_config(path,sha)
