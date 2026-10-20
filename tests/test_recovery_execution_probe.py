"""分叉必须保持传感器/MPC状态，并精确复现原回合后缀。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller import MPCController
from interfaces import GoalPose,Trajectory,VehicleState
from runtime import NetworkSource,TerminalChecker
from sim import DifferentialDriveModel
from experiments.recovery_execution import ContextCollector,MarginAuditEngine,ContinuationContext,execute_branch,select_context,verify_network_suffix,record_arrays
from scripts.probe_recovery_execution import select_training_sources,summarize,load_config,run
from scripts.prepare_recovery_balance_ab import file_digest


class FreeChecker:
    def pose_free(self,*args):return True
    def swept_segment_free(self,*args):return True


class NoisePipeline:
    def __init__(self):self.rng=np.random.default_rng(21)
    def set_target_goals(self,goals):self.goals=goals
    def capture_bev(self,*args):return SimpleNamespace(data=self.rng.normal(size=1))


class TinyModel:
    def predict(self,bev,goal,state):
        x=np.linspace(0,1,6);noise=float(bev.data[0])*.03
        return Trajectory(np.stack([x,np.full(6,noise),np.zeros(6)],axis=1),dt=.2)


class TinyPlanner:
    _collision_checker=FreeChecker()
    def plan(self,state,goal):
        return Trajectory(np.array([[state.x+i*.1,state.y,state.yaw] for i in range(6)]),dt=.2)


class InfeasiblePlanner(TinyPlanner):
    def plan(self,state,goal):raise RuntimeError('搜索预算耗尽')


def engine_parts():
    kwargs={'vehicle_model':DifferentialDriveModel(max_v=1),'terminal':TerminalChecker(),
            'collision_checker':FreeChecker()}
    mpc=MPCController(dt=.1,horizon=3,max_v=1,population=8,iterations=1,seed=8)
    return kwargs,mpc


class RecoveryExecutionTests(unittest.TestCase):
    def collect(self):
        kwargs,mpc=engine_parts()
        engine=ContextCollector(mpc=mpc,source=NetworkSource(NoisePipeline(),TinyModel()),planning_checker=FreeChecker(),max_steps=30,replan_every=10,**kwargs)
        result=engine.run(VehicleState(0,0,0),GoalPose(100,0,0))
        return engine,result,kwargs

    def test_observation_does_not_change_original_rollout(self):
        engine,original,kwargs=self.collect();_,mpc=engine_parts()
        baseline=MarginAuditEngine(mpc=mpc,source=NetworkSource(NoisePipeline(),TinyModel()),planning_checker=FreeChecker(),max_steps=30,replan_every=10,**kwargs).run(VehicleState(0,0,0),GoalPose(100,0,0))
        for a,b in zip(record_arrays(original),record_arrays(baseline)):np.testing.assert_array_equal(a,b)
        self.assertEqual([c.step for c in engine.contexts],[0,10,20])

    def test_nonzero_context_matches_and_branches_are_independent(self):
        engine,original,kwargs=self.collect();context=engine.contexts[1];rng=deepcopy(context.pipeline.rng.bit_generator.state);warm=context.mpc._last_u.copy()
        for _ in range(2):
            continued,_=execute_branch(context,'network',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10)
            self.assertTrue(verify_network_suffix(original,continued,10)['matched'])
        execute_branch(context,'expert',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10)
        self.assertEqual(context.pipeline.rng.bit_generator.state,rng);np.testing.assert_array_equal(context.mpc._last_u,warm)

    def test_wrong_rng_and_wrong_phase_are_rejected(self):
        engine,original,kwargs=self.collect();context=deepcopy(engine.contexts[1]);context.mpc._rng=np.random.default_rng(999)
        continued,_=execute_branch(context,'network',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10)
        with self.assertRaisesRegex(ValueError,'未精确复现'):verify_network_suffix(original,continued,10)
        context.step=11
        with self.assertRaises(ValueError):execute_branch(context,'network',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10)

    def test_infeasible_expert_is_a_recorded_outcome_not_silently_removed(self):
        engine,_,kwargs=self.collect()
        result,violations=execute_branch(engine.contexts[1],'expert',TinyModel(),InfeasiblePlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10)
        self.assertFalse(result.success);self.assertEqual(result.failure,'planning_failure');self.assertEqual(result.steps,0)
        self.assertEqual(record_arrays(result)[0].shape,(0,5));self.assertEqual(violations,[])

    def test_progress_covers_multiple_replanning_cycles(self):
        engine,_,kwargs=self.collect();events=[]
        execute_branch(engine.contexts[1],'expert',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=20,replan_every=10,progress=lambda n,p,t:events.append((n,p)))
        self.assertEqual(events,[(1,'BEGIN'),(1,'END'),(2,'BEGIN'),(2,'END')])

    def test_prefix_not_full_result_and_nonfinite_rejected(self):
        engine,original,kwargs=self.collect();context=engine.contexts[1]
        continued,_=execute_branch(context,'network',TinyModel(),TinyPlanner(),GoalPose(100,0,0),kwargs,max_steps=10,replan_every=10)
        self.assertTrue(verify_network_suffix(original,continued,10,prefix_only=True)['prefix_only'])
        with self.assertRaises(ValueError):verify_network_suffix(original,continued,10)
        continued.record.states[0].x=float('nan')
        with self.assertRaisesRegex(ValueError,'非有限'):record_arrays(continued)

    def test_context_selection_first_loss_and_controls(self):
        contexts=[SimpleNamespace(step=i) for i in (0,10,20)]
        self.assertEqual(select_context(contexts,[False]*14+[True]+[False]*15,False)[0].step,10)
        self.assertEqual(select_context(contexts,[False]*30,True)[1],'success_control')
        self.assertIsNone(select_context([], [True],False)[0])
        self.assertIsNone(select_context(contexts,[False]*30,False)[0])

    def test_training_identity_no_overlap_or_duplicates(self):
        metadata=[{'task_id':f'{scene}-{i}','scene_name':scene,'difficulty':{'maneuver':'forward','noise_level':'clean'}} for scene in ('S4_dump_area','S8_weigh_station') for i in range(6)]
        self.assertEqual(len(select_training_sources(metadata,[])),12)
        self.assertEqual(select_training_sources(metadata,[]),select_training_sources(metadata,[]))
        with self.assertRaises(ValueError):select_training_sources(metadata,[metadata[0]['task_id']])
        with self.assertRaises(ValueError):select_training_sources(metadata+[metadata[0]],[])

    def test_decision_requires_coverage_controls_and_execution_success(self):
        rows=[]
        for scene in ('S4_dump_area','S8_weigh_station'):
            for i in range(5):rows.append({'scene_name':scene,'selection_reason':'success_control' if i==0 else 'before_first_margin_loss','network':{'success':i==0},'expert':{'success':True},'execution_label_passed':True})
        self.assertEqual(summarize(rows)['decision'],'candidate_for_separate_data_training_plan')
        failed=deepcopy(rows);failed[0]['execution_label_passed']=False
        self.assertIn('no_training',summarize(failed)['decision'])
        smoke=deepcopy(rows);smoke[0]['smoke']=True
        self.assertIn('no_training',summarize(smoke)['decision'])
        self.assertIn('no_training',summarize(rows[:5])['decision'])

    def test_input_code_config_tamper_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);input_path=root/'data';input_path.write_bytes(b'input');config_path=root/'config.json'
            config={'kind':'training_domain_recovery_execution_probe','code':{},'runtime':{},'inputs':{'x':{'path':str(input_path),'sha256':file_digest(input_path)}},'output':str(root/'results')}
            config_path.write_text(json.dumps(config),encoding='utf-8');digest=file_digest(config_path)
            with patch('scripts.probe_recovery_execution.execution_identity',return_value={}),patch('scripts.probe_recovery_execution.runtime_identity',return_value={}):
                self.assertEqual(load_config(config_path,digest),config)
                with self.assertRaises(ValueError):load_config(config_path,'wrong')
                (root/'results').mkdir()
                with self.assertRaises(FileExistsError):run(config_path,digest)
                input_path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'输入身份'):load_config(config_path,digest)


if __name__=='__main__':unittest.main()
