import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def row(i):
    return dict(dataset_index=i,task_id=f'S8-test-{i}',scene_name='S8_weigh_station',
        success=True,collision=False,reached_with_margin=True)


class SourceValidationTests(unittest.TestCase):
    def test_18_episode_orchestration_and_no_training(self):
        from scripts import validate_s8_preservation_sources as cli
        sources={20261020:[2338,2351,2375],20261021:[2330,2338,2351,2370,2378,2398]}
        cfg=dict(sources=sources,checkpoints={arm:{s:'checkpoint' for s in sources}
            for arm in ('LT18_policy','preservation_student')})
        context=dict(config=cfg,cached_C240={s:[row(i) for i in v] for s,v in sources.items()})
        with tempfile.TemporaryDirectory() as folder:
            prepared=Path(folder)/'prepared.json';prepared.write_text(json.dumps(cfg),encoding='utf-8')
            with patch.object(cli,'verify_config'),patch.object(cli,'prepare_context',return_value=context), \
                    patch.object(cli.DatasetGenerator,'load'),patch.object(cli.matrix,'load_dataset_manifest',return_value={'vehicle_model':{}}), \
                    patch.object(cli,'VehicleConfig'),patch.object(cli,'build_case'), \
                    patch.object(cli,'load_model_checkpoint',return_value=SimpleNamespace(model=None)), \
                    patch.object(cli,'load_candidate',return_value=(None,None)), \
                    patch.object(cli,'evaluate_cases',side_effect=lambda indices,*args:[row(i) for i in indices]) as evaluate:
                result=cli.run(prepared,Path(folder)/'new')
            self.assertEqual(evaluate.call_count,4);self.assertEqual(result['complete_episodes'],18)
            self.assertEqual(result['training_updates'],0);self.assertEqual(result['status'],'completed')
            self.assertFalse(result['allowed_to_prepare_training_plan'])

    def test_precheck_and_budget_failure_do_not_run_sources(self):
        from scripts import validate_s8_preservation_sources as cli
        with patch.object(cli,'prepare_context',return_value={'config':{}}),patch.object(cli,'evaluate_cases') as evaluate:
            cli.precheck();evaluate.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            prepared=Path(folder)/'prepared.json';prepared.write_text('{}',encoding='utf-8');target=Path(folder)/'new'
            with patch.object(cli,'verify_config'),patch.object(cli,'prepare_context',side_effect=TimeoutError('budget')), \
                    patch.object(cli,'evaluate_cases') as evaluate:
                with self.assertRaises(TimeoutError):cli.run(prepared,target)
                evaluate.assert_not_called()
            report=json.loads((target/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            with patch.object(cli,'verify_config'):
                with self.assertRaises(FileExistsError):cli.run(prepared,target)


if __name__=='__main__':unittest.main()
