"""REF-DIAG: one-variable pairing, honest classifications and saved-window alignment."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from controller.mpc import MPCController
from interfaces import Trajectory,VehicleState
from runtime import TerminalChecker
from scripts import diagnose_fixed_reference as study
from scripts.fixed_reference_analysis import (
    verify_fixed_prefix,classify_fixed_result,rolling_window_updates,execution_observation)
from tests.test_feasible_cem_mpc import FreeSpace


def fixed_row(index,success=False,slow=False):
    return dict(dataset_index=index,task_id=f'task-{index}',scene_name='scene',
        branch='network_fixed',profile='zero_rescue',success=success,collision=False,
        margin_violation_steps=0,control_timing=dict(over_100ms=int(slow)))


class ClassificationTests(unittest.TestCase):
    def test_fixed_success_requires_safety_and_latency_before_replanning_route(self):
        fixed=fixed_row(2,True);rolling=dict(success=False)
        self.assertEqual(classify_fixed_result(fixed,rolling,formal=True),'fixed_reference_recovers_rolling_failure')
        for key,value in [('collision',True),('margin_violation_steps',1),('control_timing',dict(over_100ms=1))]:
            changed=deepcopy(fixed);changed[key]=value
            self.assertEqual(classify_fixed_result(changed,rolling,formal=True),'fixed_arrival_unqualified_risk_or_latency')

    def test_fixed_failure_and_smoke_never_claim_replanning_cause(self):
        self.assertEqual(classify_fixed_result(fixed_row(2),dict(success=False),formal=True),'fixed_reference_execution_also_failed')
        self.assertEqual(classify_fixed_result(fixed_row(2,True),dict(success=False),formal=False),'truncated_unclassified')
        with self.assertRaises(ValueError):classify_fixed_result(fixed_row(2),dict(success=True),formal=True)

    def fixtures(self,success=False):
        rows=[fixed_row(i,success) for i in study.INDICES]
        diagnoses=[dict(dataset_index=r['dataset_index'],task_id=r['task_id'],
            classification=classify_fixed_result(r,dict(success=False),formal=True)) for r in rows]
        return rows,diagnoses

    def test_all_failure_all_recovery_and_mixed_route_keep_qualification_false(self):
        rows,diagnoses=self.fixtures()
        result=study.summarize(rows,diagnoses,formal=True)
        self.assertEqual(result['decision']['status'],'all_fixed_also_fail_check_reference_control_contract')
        self.assertFalse(result['training_unlocked']);self.assertFalse(result['candidate_qualified'])
        rows,diagnoses=self.fixtures(True)
        self.assertEqual(study.summarize(rows,diagnoses,formal=True)['decision']['status'],'fixed_recovers_all_prioritize_replanning_continuity')
        rows[0]['success']=False;diagnoses[0]['classification']='fixed_reference_execution_also_failed'
        self.assertEqual(study.summarize(rows,diagnoses,formal=True)['decision']['status'],'split_by_task_no_global_fix')

    def test_all_21_and_original_order_are_mandatory(self):
        rows,diagnoses=self.fixtures()
        self.assertEqual(len(study.required_files()),86);self.assertEqual(len(study.required_files(True)),10)
        self.assertEqual(len(set(study.INDICES)),21)
        for a,b in [(rows[:-1],diagnoses[:-1]),(rows[::-1],diagnoses[::-1]),([rows[0]]*21,diagnoses)]:
            with self.assertRaises(ValueError):study.summarize(a,b,formal=True)
        diagnoses[0]['task_id']='wrong'
        with self.assertRaises(ValueError):study.summarize(rows,diagnoses,formal=True)

    def test_freeze_overwrite_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prepared'
            with patch.object(study,'frozen_config',return_value=dict(code='a')):
                study.prepare(path);self.assertEqual(study.precheck(path),dict(code='a'))
                with self.assertRaises(FileExistsError):study.prepare(path)
            with patch.object(study,'frozen_config',return_value=dict(code='b')):
                with self.assertRaises(ValueError):study.precheck(path)

    def test_stage_failure_retains_incomplete_and_never_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared=Path(directory)/'prepared';prepared.mkdir();(prepared/'config.json').write_text('{}')
            output=Path(directory)/'output'
            with patch.object(study,'precheck',return_value={}),patch.object(study,'case_data',side_effect=RuntimeError('fault')):
                with self.assertRaisesRegex(RuntimeError,'fault'):study.run(prepared,output,smoke=True)
                self.assertEqual(study.read_json(output/'report.json')['status'],'incomplete')
                with self.assertRaises(FileExistsError):study.run(prepared,output,smoke=True)

    def test_powershell_bom_encodings_preserve_original_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log'
            path.write_text('AUDIT 1\n{"status":"verified"}\n',encoding='utf-16')
            self.assertEqual(study.read_shell_json(path,last_line=True),dict(status='verified'))
            path.write_text('{"exit_code":0}',encoding='utf-8-sig')
            self.assertEqual(study.read_shell_json(path),dict(exit_code=0))


class PairTests(unittest.TestCase):
    def write_pair(self,root):
        for branch in ('network_fixed','network_rolling'):
            path=root/branch;n=12
            detail=dict(branch=branch,profile='zero_rescue',task_id='task-2',dataset_index=2,
                predictions=1 if branch=='network_fixed' else 2,steps=n)
            path.with_suffix('.json').write_text(json.dumps(detail))
            np.savez_compressed(path.with_suffix('.npz'),initial_state=np.zeros(5),goal=np.ones(3),
                global_plan_0000=np.zeros((4,3)),local_prediction_0000=np.zeros((4,3)),
                plan_issue_steps=np.array([0]) if branch=='network_fixed' else np.array([0,10]),
                states_before=np.zeros((n,5)),states_after=np.zeros((n,5)),control_windows=np.zeros((n,10,3)),
                selected_control_sequences=np.zeros((n,10,2)))
            length=1 if branch=='network_fixed' else 2
            np.savez_compressed(root/f'{branch}-predictions.npz',points=np.zeros((length,320,3)),logits=np.zeros((length,320)),
                bev=np.zeros((length,2,4,4)),goal=np.zeros((length,3)),motion=np.zeros((length,2)),pose=np.zeros((length,5)),
                lengths=np.full(length,4),issue_steps=np.array([0]) if length==1 else np.array([0,10]))

    def test_first_native_prediction_and_actual_prefix_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_pair(root)
            self.assertEqual(verify_fixed_prefix(root/'network_fixed',root/'network_rolling')['shared_actual_steps'],10)
            path=root/'network_fixed-predictions.npz'
            with np.load(path,allow_pickle=False) as data:changed={k:data[k] for k in data.files}
            changed['logits'][0,6]=1.
            np.savez_compressed(path,**changed)
            with self.assertRaises(AssertionError):verify_fixed_prefix(root/'network_fixed',root/'network_rolling')

    def test_control_difference_before_first_replan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_pair(root);path=root/'network_fixed.npz'
            with np.load(path,allow_pickle=False) as data:changed={k:data[k] for k in data.files}
            changed['selected_control_sequences'][9,0,0]=1.
            np.savez_compressed(path,**changed)
            with self.assertRaises(AssertionError):verify_fixed_prefix(root/'network_fixed',root/'network_rolling')


class WindowTests(unittest.TestCase):
    def fixture(self,root,shift):
        output=root/'rolling';old=np.zeros((40,3));old[:,0]=np.arange(40)*.1
        fresh=old[10:].copy();fresh[:,1]+=shift
        states=np.zeros((12,5));states[:,0]=np.arange(12)*.1
        reference=MPCController(dt=.1,horizon=10);aligned=reference._align_dt(Trajectory(fresh,.1))
        index=reference._advance_progress(aligned,VehicleState.from_array(states[10]))
        windows=np.zeros((12,10,3));windows[10]=reference._reference_points(aligned,index)
        frames=[dict(plan_ordinal=0,progress_index=i) for i in range(10)]
        frames.extend([dict(plan_ordinal=1,progress_index=index),dict(plan_ordinal=1,progress_index=index+1)])
        np.savez_compressed(output.with_suffix('.npz'),plan_issue_steps=np.array([0,10]),states_before=states,
            control_windows=windows,global_plan_0000=old,global_plan_0001=fresh,plan_dt_0000=.1,plan_dt_0001=.1)
        case=dict(planner=SimpleNamespace(_collision_checker=FreeSpace()),engine_kwargs=dict(collision_checker=FreeSpace()))
        return output,dict(frames=frames),case

    def test_continuation_and_fresh_window_are_compared_at_same_actual_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output,detail,case=self.fixture(Path(directory),0.)
            result=rolling_window_updates(output,detail,case)
            self.assertAlmostEqual(result[0]['xy_rms_m'],0.)
            output,detail,case=self.fixture(Path(directory),.5)
            result=rolling_window_updates(output,detail,case)
            self.assertAlmostEqual(result[0]['xy_rms_m'],.5)
            self.assertEqual(result[0]['previous_progress_index'],10)
            self.assertEqual(result[0]['fresh_progress_index'],0)
            self.assertFalse(result[0]['safety_regression']['planning'])

    def test_forged_consumed_window_or_issue_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output,detail,case=self.fixture(Path(directory),0.)
            detail['frames'][10]['plan_ordinal']=0
            with self.assertRaises(ValueError):rolling_window_updates(output,detail,case)
            output,detail,case=self.fixture(Path(directory),0.)
            with np.load(output.with_suffix('.npz'),allow_pickle=False) as data:changed={k:data[k] for k in data.files}
            changed['control_windows'][10,0,0]+=1.
            np.savez_compressed(output.with_suffix('.npz'),**changed)
            with self.assertRaises(AssertionError):rolling_window_updates(output,detail,case)

    def test_actual_position_and_velocity_reversals_do_not_use_reference_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'episode';states=np.zeros((4,5));states[:,0]=[0.,.1,0.,.1]
            controls=np.ones((4,10,2));controls[:,0,0]=[.2,-.2,.2,-.2]
            np.savez_compressed(output.with_suffix('.npz'),states_after=states,selected_control_sequences=controls,goal=np.array([5.,0.,0.]))
            detail=dict(success=False,failure='oscillation',controller_stats={},first_prediction=dict(endpoint_joint_passed=True),control_timing={})
            result=execution_observation(output,detail,TerminalChecker())
            self.assertEqual(result['active_velocity_sign_flips'],3)
            self.assertAlmostEqual(result['closest_position_m'],4.9)
            self.assertFalse(result['position_threshold_visited'])
            self.assertIsNone(result['first_joint_arrival_step'])


if __name__=='__main__':unittest.main()
