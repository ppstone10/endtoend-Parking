"""周期航向监督及历史训练语义兼容。"""

import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from model.network import loss_fn, endpoint_alignment_loss, terminal_pose_loss, variable_loss_fn
from model import build_model
from training.trainer import Trainer, TrainerConfig, TrainingHistory


class AngularLossTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.zeros(1, 3, 3)
        self.target[..., 2] = -math.pi + .01
        self.pred = self.target.clone()
        self.pred[..., 2] = math.pi - .01
        self.mask = torch.tensor([[1., 1., 0.]])

    def test_all_pose_terms_use_shortest_angle(self):
        for fn, kwargs, expected in [(loss_fn, {}, .0004), (endpoint_alignment_loss, {'tail_points': 1}, .0004), (terminal_pose_loss, {'yaw_weight': 10}, .004)]:
            self.assertAlmostEqual(float(fn(self.pred, self.target, self.mask, yaw_error_mode='wrapped', **kwargs)), expected, places=5)
            self.assertGreater(float(fn(self.pred, self.target, self.mask, **kwargs)), 30.)

    def test_periodicity_position_mask_and_gradient(self):
        p = self.pred.clone().requires_grad_()
        loss_fn(p, self.target, self.mask, yaw_error_mode='wrapped').backward()
        self.assertLess(float(p.grad[0, 0, 2]), 0.)  # 梯度下降朝 +pi 跨界靠近目标
        self.assertEqual(float(p.grad[0, 2].abs().sum()), 0.)
        shifted = self.target.clone(); shifted[..., 2] += 2 * math.pi
        self.assertLess(float(loss_fn(shifted, self.target, self.mask, yaw_error_mode='wrapped')), 1e-10)
        shifted[..., 0] = 2.
        self.assertAlmostEqual(float(loss_fn(shifted, self.target, self.mask, yaw_error_mode='wrapped')), 4., places=5)
        shifted[..., 0] = 0.; shifted[..., 2] = self.target[..., 2] + math.pi
        self.assertAlmostEqual(float(loss_fn(shifted, self.target, self.mask, yaw_error_mode='wrapped')), math.pi**2, places=5)

    def test_stop_loss_unaffected_and_default_is_raw(self):
        torch.manual_seed(1)
        pred,target=torch.randn(2,7,3),torch.randn(2,7,3);mask=torch.ones(2,7)
        self.assertTrue(torch.equal(loss_fn(pred,target,mask), ((pred-target).square()*mask.unsqueeze(-1)).sum()/mask.sum()))
        logits = torch.zeros(1, 3)
        for mode in ('terminal', 'cumulative'):
            a = variable_loss_fn(self.pred, logits, self.target, self.mask, stop_target_mode=mode)
            b = variable_loss_fn(self.pred, logits, self.target, self.mask, stop_target_mode=mode, yaw_error_mode='wrapped')
            expected = loss_fn(self.pred, self.target, self.mask) - loss_fn(self.pred, self.target, self.mask, yaw_error_mode='wrapped')
            self.assertAlmostEqual(float(a-b), float(expected), places=4)
        with self.assertRaises(ValueError): TrainerConfig(yaw_error_mode='invalid')
        with self.assertRaises(ValueError): loss_fn(self.pred, self.target, self.mask, yaw_error_mode='invalid')

    def test_trainer_routes_all_three_losses_and_resume_rejects_mode_change(self):
        config = {'bev_channels': 5, 'max_horizon': 3, 'dt': .2, 'hidden_dim': 8}
        model = build_model('net-v1', config)
        batch = (torch.zeros(1,5,16,16),torch.zeros(1,3),torch.zeros(1,2),self.target,self.mask)
        with tempfile.TemporaryDirectory() as directory:
            settings = TrainerConfig(epochs=1, checkpoint_dir=directory, endpoint_alignment_weight=.5, terminal_pose_weight=1, yaw_error_mode='wrapped')
            trainer = Trainer(model, settings, model_name='net-v1',model_config=config)
            points = model.forward_with_stop(*batch[:3])
            expected = variable_loss_fn(points.points, points.stop_logits, self.target,self.mask,yaw_error_mode='wrapped') + .5*endpoint_alignment_loss(points.points,self.target,self.mask,yaw_error_mode='wrapped') + terminal_pose_loss(points.points,self.target,self.mask,yaw_error_mode='wrapped')
            actual,_ = trainer._batch_loss(batch,training=False,epoch=0,batch_index=0)
            torch.testing.assert_close(actual,expected)
            path = trainer._save_checkpoint('test.pt',0,TrainingHistory())
            other = Trainer(build_model('net-v1',config),replace(settings,yaw_error_mode='raw'),model_name='net-v1',model_config=config)
            with self.assertRaisesRegex(RuntimeError,'超参数'): other.load_checkpoint(path)
            trainer.load_checkpoint(path)
            payload=torch.load(path,weights_only=False);payload['trainer_config'].pop('yaw_error_mode');torch.save(payload,Path(directory)/'old.pt')
            with self.assertRaises(RuntimeError): trainer.load_checkpoint(Path(directory)/'old.pt')
            other.load_checkpoint(Path(directory)/'old.pt')


if __name__ == '__main__': unittest.main()
