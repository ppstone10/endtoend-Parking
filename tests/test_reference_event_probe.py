"""事件分叉与一次专家替换不得改变原控制行为或感知消耗。"""
from copy import deepcopy
from types import SimpleNamespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.reference_event_probe import OneExpertUpdateSource, run_event_branch, validate_event, summarize_events
from experiments.recovery_execution import ContextCollector, record_arrays, verify_network_suffix
from interfaces import GoalPose,VehicleState
from runtime import NetworkSource
from tests.test_recovery_execution_probe import FreeChecker,NoisePipeline,TinyModel,TinyPlanner,InfeasiblePlanner,engine_parts
from scripts import probe_reference_events as cli


class EventProbeTests(unittest.TestCase):
    def collect(self):
        kwargs,mpc=engine_parts()
        engine=ContextCollector(mpc=mpc,source=NetworkSource(NoisePipeline(),TinyModel()),planning_checker=FreeChecker(),max_steps=30,replan_every=10,**kwargs)
        result=engine.run(VehicleState(0,0,0),GoalPose(100,0,0))
        return engine,result,kwargs

    def test_once_expert_then_network_and_rng_consumption(self):
        primary=NetworkSource(NoisePipeline(),TinyModel())
        expert=SimpleNamespace(begin=lambda *args:None,next_trajectory=lambda state:('expert',1.))
        source=OneExpertUpdateSource(primary,expert)
        source.begin(VehicleState(0,0,0),GoalPose(100,0,0))
        self.assertEqual(source.next_trajectory(VehicleState(0,0,0))[0],'expert')
        self.assertNotEqual(source.next_trajectory(VehicleState(0,0,0))[0],'expert')
        rng=np.random.default_rng(21);rng.normal(size=1);rng.normal(size=1)
        self.assertEqual(primary.sensor_pipeline.rng.bit_generator.state,rng.bit_generator.state)
        self.assertEqual((source.calls,source.attempts,source.applied),(2,1,1))

    def test_private_audit_is_nonintervening_and_fresh_full_suffix_matches(self):
        engine,original,kwargs=self.collect();context=engine.contexts[1]
        before=deepcopy(context.mpc._rng.bit_generator.state)
        result,margins,audit,counts=run_event_branch(context,'fresh',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10)
        self.assertTrue(verify_network_suffix(original,result,10)['matched'])
        self.assertEqual(len(audit),result.steps)
        self.assertEqual(before,context.mpc._rng.bit_generator.state)
        for j,row in enumerate(audit):
            expected=kwargs['vehicle_model'].step(VehicleState.from_array(np.array(row['state'])),result.record.cmds[j],.1).to_array()
            np.testing.assert_allclose(expected[:3],row['rollout'][0],rtol=0,atol=1e-9)
            np.testing.assert_allclose(expected,result.record.states[j].to_array(),rtol=0,atol=1e-9)

    def test_step_zero_has_no_delay_but_expert_failure_is_kept(self):
        engine,_,kwargs=self.collect();context=engine.contexts[0]
        with self.assertRaises(ValueError):run_event_branch(context,'delay_once',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=10,replan_every=10)
        result,margins,audit,counts=run_event_branch(context,'expert_once',TinyModel(),InfeasiblePlanner(),GoalPose(100,0,0),kwargs,max_steps=10,replan_every=10)
        self.assertEqual(result.failure,'planning_failure');self.assertEqual(result.steps,0)
        self.assertEqual(counts['expert_attempts'],1);self.assertEqual(counts['expert_applied'],0)
        self.assertEqual(audit,[])

    def test_branches_do_not_mutate_snapshot(self):
        engine,_,kwargs=self.collect();context=engine.contexts[1]
        warm=context.mpc._last_u.copy();rng=deepcopy(context.pipeline.rng.bit_generator.state)
        for mode in ['delay_once','expert_once']:
            result,_,audit,counts=run_event_branch(context,mode,TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10)
            self.assertEqual(len(audit),result.steps)
            if mode=='expert_once':self.assertEqual(counts['expert_applied'],1)
        np.testing.assert_array_equal(context.mpc._last_u,warm)
        self.assertEqual(context.pipeline.rng.bit_generator.state,rng)

    def test_event_must_match_type_and_be_no_later_than_first_loss(self):
        row={'original':{'success':False,'first_margin_violation_step':21},
             'updates':[{'step':20,'has_previous':True,'safety_regression':{'planning':True,'contact':False}}]}
        validate_event(row,20,'planning')
        with self.assertRaises(ValueError):validate_event(row,20,'contact')
        row['original']['first_margin_violation_step']=19
        with self.assertRaises(ValueError):validate_event(row,20,'planning')

    def test_cases_are_fixed_and_early_has_no_fake_delay(self):
        cases=cli.fixed_cases()
        self.assertEqual(len(cases),11)
        self.assertEqual(len(set(c['index'] for c in cases)),11)
        self.assertEqual([(c['index'],c['step']) for c in cases[:2]],[(1298,210),(1455,180)])
        self.assertEqual(sum(c['horizon_limited'] and c['step']==0 for c in cases),4)

    def test_archive_prefix_and_full_length_are_not_confused(self):
        _,result,_=self.collect();states,commands=record_arrays(result)
        archive={'original_states':states,'original_commands':commands}
        self.assertTrue(cli.verify_archived_source(result,archive,prefix_only=False)['matched'])
        archive['original_states']=np.vstack([states,states[-1]])
        archive['original_commands']=np.vstack([commands,commands[-1]])
        self.assertTrue(cli.verify_archived_source(result,archive,prefix_only=True)['matched'])
        with self.assertRaises(ValueError):cli.verify_archived_source(result,archive,prefix_only=False)

    def test_identity_and_reuse_guards(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.write_text('frozen')
            config={'kind':'reference_event_execution_probe','code':{},'runtime':{},'output':str(root/'results'),
                    'inputs':{'input':{'path':str(source),'sha256':cli.file_digest(source)}}}
            path=root/'config.json';path.write_text(json.dumps(config));digest=cli.file_digest(path)
            with patch.object(cli,'execution_identity',return_value={}),patch.object(cli,'runtime_identity',return_value={}):
                self.assertEqual(cli.load_config(path,digest),config)
                (root/'results').mkdir()
                with self.assertRaises(FileExistsError):cli.run(path,digest)
                source.write_text('changed')
                with self.assertRaises(ValueError):cli.load_config(path,digest)
                with self.assertRaises(ValueError):cli.load_config(path,'wrong')

    def test_early_safety_and_smoke_are_not_full_task_success(self):
        summary={'success':False,'collision':False,'steps':10,'margin_violation_steps':0,
                 'reached_with_margin':False,'planning_failure_details':None}
        row={'horizon_limited':True,'branches':{'fresh':summary,'expert_once':summary}}
        report=summarize_events([row])
        self.assertEqual(report['full_task_evaluated_sources'],0)
        self.assertEqual(report['early_planning_safe']['expert_once'],1)
        self.assertEqual(report['event_cohort_variants']['expert_once']['strict_margin_recoveries'],0)
        self.assertFalse(report['automatic_training'])
        self.assertEqual(summarize_events([row],smoke=True)['decision'],'bounded_preflight_only')


if __name__=='__main__':unittest.main()
