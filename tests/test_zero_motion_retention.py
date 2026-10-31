"""RESCUE-KEEP: identities, paired counterfactual attribution and hard rejection."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts import validate_zero_motion_retention as study
from scripts.zero_motion_retention_capture import audit_paired_prefix, audit_pair_arm
from scripts.feasible_cem_capture import latency_summary
from interfaces import VehicleState, GoalPose
from runtime import NetworkSource, TerminalChecker
from tests.test_feasible_cem_mpc import FreeSpace, LeftHalfSpace


def fixtures():
    metadata = [dict(task_id=f'task-{i}',scene_name=f'scene-{i%3}') for i in range(300)]
    groups = dict(regressions=list(range(93,115)),recoveries=[80,81,92,115,116,117,118,119],
                  residual=list(range(120,125)),native_success=list(range(93)))
    regression = dict(groups=groups,indices=list(range(125)),
        task_ids=[m['task_id'] for m in metadata[:125]],unique_tasks=125,
        evaluation_only=True,independent_unseen_test=False)
    config = dict(regression=regression,sources=study.validate_cohort(regression,metadata))
    rows = [dict(dataset_index=i,task_id=f'task-{i}',scene_name=f'scene-{i%3}',arm=arm,
        success=i<120 or (i==120 and arm=='candidate'),collision=False,margin_violation_steps=0,
        initial_contact_safe=True,initial_margin_safe=True,control_timing=dict(over_100ms=0))
        for i in range(125) for arm in study.ARMS]
    return metadata,config,rows


class RetentionTests(unittest.TestCase):
    def test_overlap_is_preserved_but_unique_denominator_is_125(self):
        metadata,config,_ = fixtures()
        sources = study.validate_cohort(config['regression'],metadata)
        self.assertEqual(len(sources),125)
        self.assertEqual(sources[80]['groups'],['recoveries','native_success'])
        self.assertEqual(len(study.jobs(config)),250)
        self.assertEqual(len(study.required_files(config)),752)
        self.assertEqual(len(study.required_files(config,True)),14)
        metadata[3]['task_id'] = metadata[2]['task_id']
        with self.assertRaises(ValueError):study.validate_cohort(config['regression'],metadata)

    def test_missing_group_member_or_task_identity_is_rejected(self):
        metadata,config,rows = fixtures()
        config['regression']['groups']['regressions'].pop()
        with self.assertRaises(ValueError):study.validate_cohort(config['regression'],metadata)
        _,config,rows = fixtures()
        rows[3]['task_id']='wrong-task'
        with self.assertRaises(ValueError):study.summarize(rows,config,formal=True)

    def test_complete_retention_and_real_increment_are_required(self):
        _,config,rows = fixtures()
        result = study.summarize(rows,config,formal=True)
        self.assertTrue(result['decision']['ready_for_next_validation'])
        self.assertEqual(result['decision']['historical_success_union'],120)
        self.assertEqual(result['decision']['gained_parent_failures'],[120])
        self.assertFalse(result['lt34_training_unlocked'])
        self.assertFalse(result['production_qualified'])
        next(r for r in rows if r['dataset_index']==120 and r['arm']=='candidate')['success']=False
        result = study.summarize(rows,config,formal=True)
        self.assertEqual(result['decision']['status'],'retention_passed_no_increment')

    def test_new_success_cannot_compensate_for_any_historical_loss(self):
        _,config,rows = fixtures()
        for i in (0,93,115):
            with self.subTest(index=i):
                changed=deepcopy(rows)
                for r in changed:
                    if r['dataset_index']==i:r['success']=False
                result=study.summarize(changed,config,formal=True)
                self.assertEqual(result['decision']['lost_historical_successes'],[i])
                self.assertFalse(result['decision']['ready_for_next_validation'])

    def test_parent_success_outside_historical_union_must_also_be_kept(self):
        _,config,rows = fixtures()
        for r in rows:
            if r['dataset_index']==124:r['success']=r['arm']=='parent'
        result=study.summarize(rows,config,formal=True)
        self.assertEqual(result['decision']['lost_parent_successes'],[124])
        self.assertFalse(result['decision']['ready_for_next_validation'])

    def test_initial_risk_margin_contact_and_any_slow_call_reject(self):
        _,config,rows = fixtures()
        for key,value in [('initial_margin_safe',False),('initial_contact_safe',False),
                          ('collision',True),('margin_violation_steps',1),('control_timing',dict(over_100ms=1))]:
            with self.subTest(key=key):
                changed=deepcopy(rows);changed[-1][key]=value
                self.assertFalse(study.summarize(changed,config,formal=True)['decision']['ready_for_next_validation'])

    def test_partial_duplicate_and_reordered_results_cannot_be_scored(self):
        _,config,rows = fixtures()
        for changed in (rows[:-1],rows[::-1],[rows[0]]*250):
            with self.assertRaises(ValueError):study.summarize(changed,config,formal=True)
        smoke=[r for r in rows if r['dataset_index']==0]
        smoke.extend([dict(r,dataset_index=183,task_id='task-183') for r in rows[:2]])
        config['sources'].append(dict(index=183,task_id='task-183'))
        self.assertFalse(study.summarize(smoke,config,formal=False)['decision']['ready_for_next_validation'])

    def test_prepare_and_run_never_overwrite_or_resume(self):
        _,config,_ = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):study.prepare(directory)
            prepared=Path(directory)/'prepared';prepared.mkdir();(prepared/'config.json').write_text('{}')
            output=Path(directory)/'result'
            with patch.object(study,'precheck',return_value=config),patch.object(study,'load_cases',side_effect=RuntimeError('fault')):
                with self.assertRaisesRegex(RuntimeError,'fault'):study.run(prepared,output,smoke=True)
                self.assertEqual(study.read_json(output/'report.json')['status'],'incomplete')
                with self.assertRaises(FileExistsError):study.run(prepared,output,smoke=True)

    def test_frozen_config_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'config.json').write_text('{"code":"a"}')
            with patch.object(study,'frozen_config',return_value=dict(code='b')):
                with self.assertRaises(ValueError):study.precheck(directory)

    def test_artifact_missing_extra_and_digest_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=root/'report.json';f.write_text('{}')
            manifest={str(f.resolve()):study.file_digest(f)}
            (root/'artifacts.json').write_text(json.dumps(manifest))
            self.assertEqual(study.verify_artifacts(root,['report.json']),1)
            f.write_text('{"forged":true}')
            with self.assertRaises(ValueError):study.verify_artifacts(root,['report.json'])
            with self.assertRaises(ValueError):study.verify_artifacts(root,['report.json','missing'])


class PrefixTests(unittest.TestCase):
    def write_pair(self,root,intervene):
        for arm in study.ARMS:
            stem=root/arm
            detail=dict(steps=2,success=False,failure='timeout',collision=False,
                margin_violation_steps=0,planning_failure=None,
                searches=[dict(rescue=dict(intervened=False)),dict(rescue=dict(intervened=intervene and arm=='candidate'))])
            stem.with_suffix('.json').write_text(json.dumps(detail))
            values=np.zeros((2,5));after=values.copy()
            if intervene and arm=='candidate':after[1,0]=1.
            np.savez_compressed(stem.with_suffix('.npz'),initial_state=np.zeros(5),goal=np.zeros(3),
                states_before=values,states_after=after,control_windows=np.zeros((2,10,3)),
                selected_control_sequences=np.zeros((2,10,2)))
            np.savez_compressed(root/f'{arm}-predictions.npz',issue_steps=np.array([0]),
                points=np.zeros((1,320,3)),logits=np.zeros((1,320)))

    def test_only_post_intervention_state_divergence_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_pair(root,True)
            self.assertEqual(audit_paired_prefix(root/'parent',root/'candidate')['shared_control_steps'],1)
            file=root/'candidate-predictions.npz'
            np.savez_compressed(file,issue_steps=np.array([0]),points=np.ones((1,320,3)),logits=np.zeros((1,320)))
            with self.assertRaises(AssertionError):audit_paired_prefix(root/'parent',root/'candidate')

    def test_no_intervention_requires_full_execution_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_pair(root,False)
            self.assertEqual(audit_paired_prefix(root/'parent',root/'candidate')['shared_control_steps'],2)
            file=root/'candidate.npz'
            with np.load(file,allow_pickle=False) as data:changed={k:data[k] for k in data.files}
            changed['selected_control_sequences'][1,0,0]=.2
            np.savez_compressed(file,**changed)
            with self.assertRaises(AssertionError):audit_paired_prefix(root/'parent',root/'candidate')


class ZeroStepTests(unittest.TestCase):
    def test_unsafe_start_is_auditable_negative_and_forged_reason_is_rejected(self):
        state=VehicleState(.1,0.,0.);goal=GoalPose(1.,0.,0.);terminal=TerminalChecker()
        case=dict(start=state,goal=goal,planner=SimpleNamespace(_collision_checker=LeftHalfSpace()),
            engine_kwargs=dict(collision_checker=FreeSpace(),terminal=terminal,vehicle_model=None))
        detail=dict(arm='candidate',branch='network_rolling',profile='zero_rescue',
            initial_contact_safe=True,initial_margin_safe=False,steps=0,frames=[],searches=[],
            success=False,collision=False,failure='safety_stop',safety_stop=True,
            margin_violation_steps=0,zero_hold_steps=0,final_pos_err=.9,final_yaw_err=0.,
            control_timing=latency_summary([1.]),controller_stats={},predictions=1,planning_failure=None,
            failed_calls=[dict(before=state.to_array().tolist(),reason='current footprint violates required clearance',compute_ms=1.)])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'zero'
            local=np.zeros((1,3),dtype=np.float32)
            world=NetworkSource._to_global(local,state)
            np.savez_compressed(output.with_suffix('.npz'),states_before=np.empty((0,5)),states_after=np.empty((0,5)),
                selected_control_sequences=np.empty((0,10,2)),selected_control_rollouts=np.empty((0,10,5)),
                goal=np.array([1.,0.,0.]),initial_state=state.to_array(),plan_issue_steps=np.array([0]),
                plan_issue_states=state.to_array()[None],local_prediction_0000=local,global_plan_0000=world,plan_dt_0000=.2)
            np.savez_compressed(Path(directory)/'zero-predictions.npz',points=np.zeros((1,320,3),dtype=np.float32),
                logits=np.ones((1,320),dtype=np.float32),lengths=np.array([1]),issue_steps=np.array([0]),pose=state.to_array()[None])
            self.assertEqual(audit_pair_arm(output,detail,case)['steps'],0)
            detail['failed_calls'][0]['reason']='forged'
            with self.assertRaisesRegex(ValueError,'unverified'):audit_pair_arm(output,detail,case)

if __name__=='__main__':unittest.main()
