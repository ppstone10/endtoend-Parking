"""参考观察不得干预，单次延迟不得跳过感知或重置控制进度。"""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.reference_update_stability import ReferenceUpdateCollector,SingleDelayedUpdateSource,choose_context,compare_windows,continue_update,summarize
from experiments.recovery_execution import ContextCollector,record_arrays,verify_network_suffix
from interfaces import GoalPose,VehicleState,Trajectory
from runtime import NetworkSource
from tests.test_recovery_execution_probe import FreeChecker,NoisePipeline,TinyModel,engine_parts
from scripts import validate_reference_updates as cli


class ReferenceUpdateTests(unittest.TestCase):
    def collect(self,cls=ReferenceUpdateCollector):
        kwargs,mpc=engine_parts()
        engine=cls(mpc=mpc,source=NetworkSource(NoisePipeline(),TinyModel()),planning_checker=FreeChecker(),
                   max_steps=30,replan_every=10,**kwargs)
        result=engine.run(VehicleState(0,0,0),GoalPose(100,0,0))
        return engine,result,kwargs

    def test_observation_is_nonintervening_and_aligned(self):
        engine,result,_=self.collect();baseline,old,_=self.collect(ContextCollector)
        for a,b in zip(record_arrays(result),record_arrays(old)):np.testing.assert_array_equal(a,b)
        self.assertEqual([u['step'] for u in engine.updates],[0,10,20])
        self.assertFalse(engine.updates[0]['has_previous'])
        self.assertTrue(all(u['has_previous'] for u in engine.updates[1:]))
        self.assertTrue(all(not any(u['safety_regression'].values()) for u in engine.updates[1:]))
        self.assertEqual(engine.mpc._rng.bit_generator.state,baseline.mpc._rng.bit_generator.state)

    def test_wrapped_angles_and_shape_guard(self):
        old=np.array([[0,0,np.pi-.01],[1,0,np.pi-.01]])
        new=old.copy();new[:,2]=-np.pi+.01;new[:,1]=.1
        metrics=compare_windows(old,new)
        self.assertAlmostEqual(metrics['xy_rms_m'],.1)
        self.assertAlmostEqual(metrics['yaw_rms_rad'],.02)
        with self.assertRaises(ValueError):compare_windows(old,new[:1])

    def test_delay_consumes_first_prediction_and_is_only_once(self):
        class Primary:
            def begin(self,start,goal):self.calls=0
            def next_trajectory(self,state):
                self.calls+=1
                return Trajectory(np.zeros((2,3))+self.calls,.1),self.calls
        previous=Trajectory(np.zeros((2,3)),.1);primary=Primary();source=SingleDelayedUpdateSource(primary,previous)
        source.begin(None,None)
        first,_=source.next_trajectory(None);second,_=source.next_trajectory(None)
        self.assertIs(first,previous);self.assertIsNot(second,previous)
        self.assertEqual(primary.calls,2);self.assertEqual(source.delays,1)
        with self.assertRaises(ValueError):SingleDelayedUpdateSource(primary,None)

    def test_fresh_full_suffix_and_delay_object_progress(self):
        engine,original,kwargs=self.collect();context=engine.contexts[1]
        state_before=deepcopy(context.pipeline.rng.bit_generator.state);warm=context.mpc._last_u.copy()
        goal=GoalPose(100,0,0)
        fresh,_=continue_update(context,TinyModel(),goal,kwargs,FreeChecker(),delay_once=False,max_steps=20,replan_every=10)
        self.assertTrue(verify_network_suffix(original,fresh,10)['matched'])
        delayed,_=continue_update(context,TinyModel(),goal,kwargs,FreeChecker(),delay_once=True,max_steps=20,replan_every=10)
        clone=deepcopy(context.mpc)
        expected=clone.compute(clone._progress_traj,context.state)
        self.assertEqual(delayed.record.cmds[0],expected)
        self.assertEqual(context.pipeline.rng.bit_generator.state,state_before)
        np.testing.assert_array_equal(context.mpc._last_u,warm)
        with self.assertRaises(ValueError):continue_update(engine.contexts[0],TinyModel(),goal,kwargs,FreeChecker(),delay_once=True,max_steps=20,replan_every=10)

    def test_success_control_is_not_lost_when_source_had_margin_loss(self):
        engine,_,_=self.collect();events=[False]*30;events[12]=True
        context,role=choose_context(engine.contexts,events,True)
        self.assertEqual(role,'success_control');self.assertGreater(context.step,0)
        context,role=choose_context(engine.contexts,events,False)
        self.assertEqual(context.step,10);self.assertEqual(role,'pre_loss_recovery')
        events[1]=True
        self.assertIsNone(choose_context(engine.contexts,events,False)[0])

    def test_summary_reports_missing_controls_and_never_trains(self):
        row={'scene_name':'S4_dump_area','updates':[],'context_role':'pre_loss_recovery',
             'fresh':{'success':False,'reached_with_margin':False},'delay_once':{'success':True,'reached_with_margin':True}}
        result=summarize([row]*8)
        self.assertFalse(result['mechanism_coverage_satisfied'])
        self.assertFalse(result['automatic_training'])
        self.assertEqual(summarize([row],smoke=True)['decision'],'bounded_preflight_only')

    def test_fixed_sampling_is_disjoint_and_scene_balanced(self):
        metadata=[]
        for scene,count in [('S4_dump_area',12),('S8_weigh_station',12),('S1_parking_lot',3),('S2_diagonal_lot',3)]:
            for _ in range(count):
                metadata.append({'task_id':str(len(metadata)),'scene_name':scene,
                                 'difficulty':{'maneuver':'turn','noise_level':'clean'}})
        indices=cli.select_sources(metadata,{'held_out'})
        self.assertEqual(indices,cli.select_sources(metadata,{'held_out'}))
        self.assertEqual(len(indices),len(set(indices)))
        self.assertEqual(Counter(metadata[i]['scene_name'] for i in indices),
                         {'S4_dump_area':12,'S8_weigh_station':6,'S1_parking_lot':3,'S2_diagonal_lot':3})
        with self.assertRaises(ValueError):cli.select_sources(metadata,{'0'})

    def test_identity_tampering_and_output_reuse_are_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.write_text('frozen')
            config={'kind':'training_domain_reference_update_validation','code':{},'runtime':{},
                    'output':str(root/'results'),'inputs':{'input':{'path':str(source),'sha256':cli.file_digest(source)}}}
            path=root/'config.json';path.write_text(json.dumps(config));digest=cli.file_digest(path)
            with patch.object(cli,'execution_identity',return_value={}),patch.object(cli,'runtime_identity',return_value={}):
                self.assertEqual(cli.load_config(path,digest),config)
                (root/'results').mkdir()
                with self.assertRaises(FileExistsError):cli.run(path,digest)
                source.write_text('changed')
                with self.assertRaises(ValueError):cli.load_config(path,digest)
                with self.assertRaises(ValueError):cli.load_config(path,'wrong')

    def test_coverage_is_not_a_training_or_smoke_pass(self):
        rows=[]
        for scene,roles in [('S4_dump_area',['pre_loss_recovery']*4+['success_control']*2),
                            ('S1_parking_lot',['success_control']*2),('S2_diagonal_lot',['success_control']*2)]:
            for role in roles:
                rows.append({'scene_name':scene,'context_role':role,'updates':[],
                    'fresh':{'success':True,'reached_with_margin':True},
                    'delay_once':{'success':True,'reached_with_margin':True}})
        report=summarize(rows)
        self.assertTrue(report['mechanism_coverage_satisfied'])
        self.assertFalse(report['automatic_training']);self.assertFalse(report['automatic_deployment'])
        self.assertFalse(report['s8_success_control_available'])
        self.assertFalse(summarize(rows,smoke=True)['mechanism_coverage_satisfied'])

    def test_prior_source_mismatch_is_rejected(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'report.json'
            path.write_text(json.dumps({'tasks':[{'dataset_index':12,'original':{'success':False,'final_pos_error':.2}}]}))
            config={'inputs':{'prior_report':{'path':str(path)}}}
            self.assertTrue(cli.verify_prior_source(config,12,{'success':False,'final_pos_error':.2})['matched'])
            with self.assertRaises(ValueError):cli.verify_prior_source(config,12,{'success':True,'final_pos_error':.2})
            self.assertFalse(cli.verify_prior_source(config,13,{})['applicable'])


if __name__=='__main__':unittest.main()
