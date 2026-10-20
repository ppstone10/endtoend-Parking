from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.closed_loop_benefit import summarize_pair, validate_report


def report(outcomes, scenes=None):
    scenes=scenes or ['S3']*len(outcomes)
    rows=[{'success':value=='success','collision':value=='collision',
           'failure':None if value=='success' else value,'steps':20,
           'final_pos_err':.1,'final_yaw_err':.02,
           'meta':{'dataset_index':i,'task_id':f'task-{i}','scene_name':scenes[i],
                   'tol_pos':.3,'tol_yaw':.17}} for i,value in enumerate(outcomes)]
    n=len(rows)
    return {'status':'completed','experiment':{'name':'E5','bev_source':'sensor','trajectory_source':'network',
            'executor':'mpc_vehicle','safety_mode':'none'},'vehicle_model':{'length':6},'reconstruct_failures':[],
            'protocol':{'data':'data.npz','checkpoint':'model.pt','model_config':{'stop_threshold':.26},
                        'evaluated_indices':list(range(n)),'selected_indices':list(range(n)),
                        'control_seed':0,'max_steps':600,'replan_every':10,'mpc_dt':.1},
            'overall':{'evaluated_samples':n,'requested_indices':n,'reconstruct_failures':0,
                       'success_rate':outcomes.count('success')/n,'collision_rate':outcomes.count('collision')/n},
            'episodes':rows}


class ClosedLoopBenefitTests(unittest.TestCase):
    def test_joint_benefit_and_transitions(self):
        a=report(['success','collision','collision','oscillation'])
        b=report(['success','success','collision','success'])
        result=summarize_pair(a,b,list(range(4)),meaningful_count=1)
        self.assertTrue(result['meaningful_joint_gain'])
        self.assertEqual(result['success_delta'],2)
        self.assertEqual(result['collision_delta'],-1)
        self.assertEqual(result['new_success_indices'],[1,3])
        self.assertEqual(result['transitions']['collision->success'],1)
        self.assertFalse(result['main_protocol_passed'])
        self.assertFalse(result['automatic_deployment'])

    def test_collision_replaced_with_timeout_is_not_joint_gain(self):
        a=report(['success','collision','collision'])
        b=report(['success','timeout','collision'])
        result=summarize_pair(a,b,[0,1,2],meaningful_count=1)
        self.assertFalse(result['joint_gain'])
        self.assertEqual(result['decision'],'no_joint_gain_stop')
        self.assertEqual(result['candidate']['failures']['timeout'],1)

    def test_scene_regression_is_visible_without_requiring_all_samples_retained(self):
        scenes=['S4','S3','S3','S8']
        a=report(['success','collision','collision','collision'],scenes)
        b=report(['collision','success','success','success'],scenes)
        result=summarize_pair(a,b,[0,1,2,3],meaningful_count=1)
        self.assertTrue(result['joint_gain'])
        self.assertEqual(result['lost_success_indices'],[0])
        self.assertEqual(result['scene_regressions'],['S4'])
        self.assertTrue(result['requires_scene_review'])

    def test_selection_target_never_means_main_pass_or_smoke_gain(self):
        a=report(['success','collision','collision','collision'])
        b=report(['success']*4)
        formal=summarize_pair(a,b,[0,1,2,3],meaningful_count=1)
        self.assertEqual(formal['decision'],'selection_target_met_requires_main_protocol')
        smoke=summarize_pair(a,b,[0,1,2,3],meaningful_count=1,smoke=True)
        self.assertFalse(smoke['selection_target_met'])
        self.assertFalse(smoke['joint_gain'])
        self.assertEqual(smoke['decision'],'bounded_smoke_not_formal_evidence')

    def test_small_gain_not_promoted_to_meaningful(self):
        a=report(['success','collision','collision'])
        b=report(['success','success','collision'])
        result=summarize_pair(a,b,[0,1,2],meaningful_count=9)
        self.assertEqual(result['decision'],'small_joint_gain_not_sufficient')

    def test_invalid_reports_rejected(self):
        source=report(['success','collision'])
        for mutate in [lambda r:r['episodes'].pop(),
                       lambda r:r['episodes'][1]['meta'].update(dataset_index=0),
                       lambda r:r['episodes'][0].update(final_pos_err=float('nan')),
                       lambda r:r['overall'].update(success_rate=0.),
                       lambda r:r['experiment'].update(safety_mode='expert_fallback'),
                       lambda r:r['reconstruct_failures'].append({'dataset_index':3}),
                       lambda r:r['episodes'][0].update(collision=True)]:
            bad=deepcopy(source);mutate(bad)
            with self.assertRaises(ValueError):validate_report(bad,[0,1])

    def test_identity_and_protocol_mismatch_rejected(self):
        source=report(['success','collision'])
        for mutate in [lambda r:r['protocol'].update(control_seed=1),
                       lambda r:r['protocol'].update(model_config={'stop_threshold':.5}),
                       lambda r:r['episodes'][0]['meta'].update(task_id='other'),
                       lambda r:r['vehicle_model'].update(length=7)]:
            bad=deepcopy(source);mutate(bad)
            with self.assertRaises(ValueError):summarize_pair(source,bad,[0,1])

    def test_zero_step_planning_failure_remains_in_denominator(self):
        source=report(['planning_failure','collision'])
        source['episodes'][0]['steps']=0
        self.assertEqual(len(validate_report(source,[0,1])),2)


class BenefitRunnerTests(unittest.TestCase):
    @staticmethod
    def config():
        return {'indices':[0,1,2],'smoke_indices':[0,1,2],'arms':['baseline','candidate'],
                'inputs':{'data':{'path':'data.npz'},'baseline':{'path':'model.pt'},'candidate':{'path':'model.pt'}},
                'smoke_max_steps':20,'max_steps':600,'replan_every':10,'seed':0,'meaningful_count':9,
                'model_config':{'stop_threshold':.26},'model_name':None,
                'task_identities':[{'task_id':f'task-{i}','scene_name':'S3'} for i in range(3)]}

    def test_prepare_rejects_existing_and_config_sha_drift(self):
        from scripts import validate_closed_loop_benefit as cli
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileExistsError):cli.prepare(name)
            path=Path(name)/'config.json';path.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):cli.load_config(path,'wrong-sha')

    def test_check_only_never_rolls_out_or_writes_output(self):
        from scripts import validate_closed_loop_benefit as cli
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'config.json'
            with patch.object(cli,'load_config',return_value=self.config()),patch.object(cli.matrix,'run_validation_experiment') as evaluation:
                cli.run(path,'sha',check_only=True)
                evaluation.assert_not_called()
            self.assertFalse((path.parent/'results').exists())

    def test_smoke_never_passes_and_refuses_overwrite(self):
        from scripts import validate_closed_loop_benefit as cli
        example=report(['success']*3);example['protocol']['max_steps']=20
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'config.json'
            with patch.object(cli,'load_config',return_value=self.config()),patch.object(cli.matrix,'run_validation_experiment',return_value=example):
                cli.run(path,'sha',smoke=True)
                with self.assertRaises(FileExistsError):cli.run(path,'sha',smoke=True)
            saved=json.loads((path.parent/'preflight/report.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['summary']['selection_target_met'])
            self.assertFalse(saved['summary']['joint_gain'])
            self.assertFalse((path.parent/'results').exists())

    def test_explicit_resume_reuses_complete_arm_and_checks_report_sha(self):
        from scripts import validate_closed_loop_benefit as cli
        example=report(['success','collision','oscillation'])
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'config.json'
            with patch.object(cli,'load_config',return_value=self.config()),patch.object(cli.matrix,'run_validation_experiment',side_effect=[example,RuntimeError('test failure')]):
                with self.assertRaises(RuntimeError):cli.run(path,'sha')
            self.assertFalse((path.parent/'results/run.lock').exists())
            with patch.object(cli,'load_config',return_value=self.config()),patch.object(cli.matrix,'run_validation_experiment',return_value=example) as evaluation:
                cli.run(path,'sha',resume=True)
                self.assertEqual(evaluation.call_count,1)
            self.assertEqual(json.loads((path.parent/'results/status.json').read_text())['status'],'completed')
            with patch.object(cli,'load_config',return_value=self.config()):
                with self.assertRaises(FileExistsError):cli.run(path,'sha',resume=True)

    def test_resume_rejects_tampered_completed_arm(self):
        from scripts import validate_closed_loop_benefit as cli
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'config.json'
            with patch.object(cli,'load_config',return_value=self.config()),patch.object(cli.matrix,'run_validation_experiment',side_effect=[report(['success','collision','oscillation']),RuntimeError('test failure')]):
                with self.assertRaises(RuntimeError):cli.run(path,'sha')
            (path.parent/'results/baseline.json').write_text('{}',encoding='utf-8')
            with patch.object(cli,'load_config',return_value=self.config()),patch.object(cli.matrix,'run_validation_experiment') as evaluation:
                with self.assertRaises(ValueError):cli.run(path,'sha',resume=True)
                evaluation.assert_not_called()


if __name__=='__main__':unittest.main()
