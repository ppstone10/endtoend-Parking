from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.behavior_exploration import (TRAINING_SEEDS,arm_samples,first_stop,
                                               fit_exploration_arm,paired_counts,rank_directions,fixed_frame_metrics,
                                               source_balanced_pools,source_balanced_batch)
from experiments.executable_behavior import prepare_supervision_frames
from experiments.exploration_store import ExplorationBudget,ExplorationStore,exploration_lock,verify_unit


def row(index,success=False,collision=False,scene='S4_dump_area'):
    return {'dataset_index':index,'task_id':f'task-{index}','scene_name':scene,
            'success':success,'collision':collision}


class Tiny(torch.nn.Module):
    stop_threshold=.26
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(2,4)
    def forward_with_stop(self,bev,goal,state):
        value=self.linear(state)
        return SimpleNamespace(points=value[:,:3,None].transpose(1,2),stop_logits=value[:,3:])


def sample(role='recovery'):
    return {'bev':torch.zeros(1),'goal':torch.zeros(3),'state':torch.ones(2),'points':torch.ones(1,3),
            'stop':torch.ones(1),'parent_stop':torch.full((1,),.2),'parent_length':1,
            'mask':torch.ones(1),'role':role}


class LearningExplorationTests(unittest.TestCase):
    def test_arm_changes_only_declared_supervision(self):
        sparse=[sample('anchor'),sample()];dense=sparse+[sample()]
        self.assertIs(arm_samples(sparse,dense,'sparse_joint'),sparse)
        self.assertIs(arm_samples(sparse,dense,'dense_joint'),dense)
        changed=arm_samples(sparse,dense,'dense_parent_stop')
        torch.testing.assert_close(changed[1]['stop'],dense[1]['parent_stop'])
        torch.testing.assert_close(dense[1]['stop'],torch.ones(1))
        self.assertEqual(len(arm_samples(sparse,dense,'dense_recovery_only')),2)
        with self.assertRaises(ValueError):arm_samples(sparse,dense,'unknown')

    def test_dense_time_coverage_does_not_change_source_sampling_weight(self):
        from collections import Counter
        sparse=[dict(sample(),source_index=i,frame_index=j) for i in (1,2) for j in range(4)]
        dense=[dict(sample(),source_index=i,frame_index=j) for i,n in [(1,5),(2,27)] for j in range(n)]
        def selected(samples):
            pools=source_balanced_pools(samples,TRAINING_SEEDS[0])
            return [s for step in range(12) for s in source_balanced_batch(pools,step)]
        a,b=selected(sparse),selected(dense)
        self.assertEqual(Counter(s['source_index'] for s in a),Counter(s['source_index'] for s in b))
        self.assertEqual([s['source_index'] for s in a],[s['source_index'] for s in b])
        self.assertGreater(len({s['frame_index'] for s in b if s['source_index']==2}),4)

    def test_fixed_fit_copies_parent_and_restores_optimizer(self):
        parent=Tiny();before=deepcopy(parent.state_dict());saved=[]
        def capture(step,model,optimizer,history):saved.append((step,deepcopy(optimizer.state_dict()),len(history)))
        model=fit_exploration_arm(parent,[sample(),sample('anchor')],seed=TRAINING_SEEDS[0],steps=2,
                                  snapshots=[2],check_budget=lambda:None,checkpoint=capture)
        self.assertEqual(saved[0][0],2);self.assertEqual(saved[0][2],2)
        for k,v in before.items():torch.testing.assert_close(v,parent.state_dict()[k],atol=0,rtol=0)
        fixed_frame_metrics(model,[sample()])
        fit_exploration_arm(model,[sample()],seed=TRAINING_SEEDS[0],steps=2,snapshots=[2],
                            optimizer_state=saved[0][1],start_step=2,check_budget=lambda:None,checkpoint=capture)
        self.assertEqual(saved[-1][0],4)
        self.assertTrue(all(int(s['step'])==4 for s in saved[-1][1]['state'].values()))
        self.assertEqual(first_stop(torch.tensor([.1,.3]),.26),2)

    def test_formal_snapshots_and_legacy_first_twelve_are_identical(self):
        from experiments.behavior_preservation import fit_bounded_copy
        parent=Tiny();samples=[sample('anchor'),sample()]
        expected,_=fit_bounded_copy(parent,samples);seen=[]
        def capture(step,model,optimizer,history):
            seen.append(step)
            if step==12:
                for k,v in expected.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
        fit_exploration_arm(parent,samples,seed=TRAINING_SEEDS[0],steps=240,snapshots=[12,60,240],
                            check_budget=lambda:None,checkpoint=capture)
        self.assertEqual(seen,[12,60,240])

    def test_pairing_and_confirmation_joint_gate(self):
        baseline=[row(i,success=i<30,collision=i>=30) for i in range(60)]
        candidate=deepcopy(baseline)
        for i in [30,31,32]:candidate[i].update(success=True,collision=False)
        result=paired_counts(baseline,candidate)
        self.assertTrue(result['confirmation_passed'])
        candidate[0].update(success=False,collision=True)
        self.assertFalse(paired_counts(baseline,candidate)['confirmation_passed'])
        with self.assertRaises(ValueError):paired_counts(baseline,candidate[:-1])
        broken=deepcopy(candidate);broken[0]['task_id']='different'
        with self.assertRaises(ValueError):paired_counts(baseline,broken)

    def test_no_gain_does_not_eliminate_confirmation_directions(self):
        base=[row(1,collision=True)]
        finals={name:{s:{'probe':base} for s in TRAINING_SEEDS} for name in ('z','a','b')}
        ranked=rank_directions(finals,base)
        self.assertEqual([x['direction'] for x in ranked],['a','b'])
        self.assertTrue(all(not x['discovery_eligible'] for x in ranked))

    def test_source_group_split_is_deterministic_and_excludes_fit_probe(self):
        from scripts.explore_closed_loop_learning import confirmation_indices
        # 合成同形训练元数据，覆盖固定索引和全部场景，不读真实训练数据。
        scenes=(1,2,3,4,5,6,8)
        metadata=[{'task_id':f'id-{i}','source_task_id':f'group-{i//2}',
                   'scene_name':f'S{scenes[i%7]}_scene'} for i in range(2400)]
        a=confirmation_indices(metadata);b=confirmation_indices(metadata)
        from scripts.preflight_behavior_repair import FIT,PROBE
        self.assertEqual(a,b);self.assertEqual(len(set(a)),60)
        self.assertFalse(set(a)&set(FIT+PROBE))
        self.assertTrue(all(not metadata[i]['scene_name'].startswith('S9_') for i in a))
        forbidden={metadata[i]['source_task_id'] for i in FIT+PROBE}
        self.assertFalse({metadata[i]['source_task_id'] for i in a}&forbidden)
        self.assertEqual(len({metadata[i]['source_task_id'] for i in a}),60)

    def test_all_frames_explicit_and_default_remains_four(self):
        frame={'kind':'expert','bev':np.zeros((5,2,2),dtype=np.float32),'goal':np.zeros(3,dtype=np.float32),
               'state':np.zeros(2,dtype=np.float32),'points':np.zeros((320,3),dtype=np.float32),
               'length':4,'mask':np.ones(320,dtype=np.float32)}
        frames=[deepcopy(frame) for _ in range(9)]
        self.assertEqual(len(prepare_supervision_frames(frames,None,'recovery',1)),4)
        self.assertEqual(len(prepare_supervision_frames(frames,None,'recovery',1,frame_limit=None)),9)
        with self.assertRaises(ValueError):prepare_supervision_frames(frames,None,'recovery',1,frame_limit=0)

    def test_store_skips_verified_and_rejects_drift_and_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);calls=[];store=ExplorationStore(root,lambda:None,ExplorationBudget(60))
            def operation(path):calls.append(1);return {'value':3}
            payload,attempt=store.execute('one',operation)
            self.assertEqual(payload['value'],3)
            self.assertEqual(store.execute('one',operation)[0],payload);self.assertEqual(len(calls),1)
            with exploration_lock(root):
                with self.assertRaises(FileExistsError):
                    with exploration_lock(root):pass
            (attempt/'payload.pt').write_bytes(b'damaged')
            with self.assertRaises(ValueError):verify_unit(root/'units/one')

    def test_failed_attempt_is_preserved_and_not_treated_as_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);store=ExplorationStore(root,lambda:None,ExplorationBudget(60))
            def fail(path):raise ValueError('expected failure')
            with self.assertRaises(ValueError):store.execute('one',fail)
            self.assertFalse((root/'units/one/completed.json').exists())
            store.execute('one',lambda path:{'ok':True})
            self.assertEqual(len(list((root/'units/one').glob('attempt-*'))),2)

    def test_config_drift_and_budget_cannot_silently_resume(self):
        from unittest.mock import patch
        from scripts.explore_closed_loop_learning import prepare,verify_identity,IdentityMismatch
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'new'
            with patch('scripts.explore_closed_loop_learning.make_config',return_value={'smoke':True,'version':1}):
                prepare(output,smoke=True);verify_identity(output)
            with patch('scripts.explore_closed_loop_learning.make_config',return_value={'smoke':True,'version':2}):
                with self.assertRaises(IdentityMismatch):verify_identity(output)
        with self.assertRaises(TimeoutError):ExplorationBudget(1,used=2).check()


if __name__=='__main__':unittest.main()
