"""FEAS-CEM-003..005: real execution capture, rejection and study boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from controller.constrained_mpc import ControlSafetyError
from interfaces import GoalPose, VehicleState
from model.variants import MineParkingNetV1
from planner import RectangleFootprintCollisionChecker
from runtime import TerminalChecker
from sim import DifferentialDriveModel, MINING_DRILL_RIG, ParkingEnvironment
from scripts.compare_feasible_cem import compare_rows, prepare, precheck, required_files, summaries, verify_initial_predictions
from scripts.feasible_cem_capture import record_feasible_branch, audit_execution, audit_predictions, latency_summary
from scripts.feasible_cem_mpc import FeasibleCEMController
from tests.test_emitted_execution_diagnosis import Pipeline, Budget


def task_row(index, success, *, collision=False, margin=0):
    return dict(task_id=f'task-{index}', dataset_index=index, success=success,
                collision=collision, margin_violation_steps=margin)


class FeasibleStudyTests(unittest.TestCase):
    def test_new_arrival_cannot_offset_old_success_loss(self):
        old = [task_row(1, True), task_row(2, False)]
        new = [task_row(1, False), task_row(2, True)]
        result = compare_rows(old, new)
        self.assertEqual(result['lost_successes'], [1])
        self.assertEqual(result['gained_successes'], [2])
        self.assertFalse(result['mechanism_gain'])

    def test_holds_and_margin_are_not_arrival_benefit(self):
        old = [task_row(1, False)]
        self.assertFalse(compare_rows(old, [task_row(1, False)])['mechanism_gain'])
        self.assertFalse(compare_rows(old, [task_row(1, True, margin=1)])['mechanism_gain'])
        self.assertTrue(compare_rows(old, [task_row(1, True)])['mechanism_gain'])

    def test_duplicate_missing_or_mismatched_task_denominators_fail(self):
        for old, new in [([], []), ([task_row(1, True)] * 2, [task_row(1, True)]),
                         ([task_row(1, True)], [task_row(2, True)])]:
            with self.assertRaises(ValueError):
                compare_rows(old, new)

    def test_prepare_never_overwrites_and_precheck_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'prepared'
            config = dict(schema='fixture', protocol={'required_margin_m': .2})
            with patch('scripts.compare_feasible_cem.frozen_config', return_value=config):
                prepare(path)
                self.assertEqual(precheck(path), config)
                with self.assertRaises(FileExistsError):
                    prepare(path)
                (path / 'config.json').write_text(json.dumps(dict(schema='fixture')), encoding='utf-8')
                with self.assertRaises(ValueError):
                    precheck(path)

    def test_required_files_have_all_24_branches_and_16_predictions(self):
        files = required_files(list(range(8)))
        self.assertEqual(len(files), 66)
        self.assertEqual(sum(name.endswith('-predictions.npz') for name in files), 16)

    def test_truncated_smoke_never_claims_full_episode_regression(self):
        row = dict(dataset_index=0, task_id='task-0')
        for branch in ('network_fixed', 'network_rolling', 'teacher_fixed'):
            row[branch] = task_row(0, False)
        result = summaries([row], formal=False)
        self.assertEqual(result['comparisons']['network_fixed']['status'], 'not_evaluated_truncated_smoke')
        self.assertNotIn('lost_successes', result['comparisons']['network_fixed'])
        self.assertFalse(result['new_contract_teacher_strict_pass'])

    def test_latency_counts_deadlines_and_rejects_bad_values(self):
        summary = latency_summary([1., 100., 101.])
        self.assertEqual(summary['over_100ms'], 1)
        self.assertEqual(summary['calls'], 3)
        self.assertIsNone(latency_summary([])['max_ms'])
        with self.assertRaises(ValueError):
            latency_summary([float('nan')])

    def test_prediction_audit_rejects_teacher_length_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'native'
            pose = np.zeros((1, 5))
            points = np.zeros((1, 320, 3), dtype=np.float32)
            logits = np.full((1, 320), -1., dtype=np.float32)
            logits[0, 2:] = 1.
            np.savez(output.with_suffix('.npz'), plan_issue_steps=np.array([0]),
                plan_issue_states=pose, local_prediction_0000=points[0, :2],
                global_plan_0000=points[0, :2], plan_dt_0000=np.array(.2))
            np.savez(Path(temporary) / 'native-predictions.npz', points=points, logits=logits,
                pose=pose, lengths=np.array([2]), issue_steps=np.array([0]))
            with self.assertRaises(AssertionError):
                audit_predictions(output, dict(branch='network_fixed', predictions=1))

    def test_same_prediction_cannot_hide_different_first_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'baseline'
            original.mkdir()
            prediction = {key: np.zeros((1, 2)) for key in ('points', 'logits', 'bev', 'goal', 'motion', 'pose', 'lengths')}
            np.savez(original / '0000-network_fixed-predictions.npz', **prediction)
            for branch in ('network_fixed', 'network_rolling'):
                np.savez(root / f'0000-{branch}-predictions.npz', **prediction)
                arrays = {key: np.zeros((2, 5)) for key in ('states_before', 'states_after', 'control_windows', 'selected_control_sequences')}
                arrays['global_plan_0000'] = np.zeros((2, 3))
                if branch == 'network_rolling':
                    arrays['selected_control_sequences'][0, 0] = .1
                np.savez(root / f'0000-{branch}.npz', **arrays)
            with patch('scripts.compare_feasible_cem.BASELINE', original):
                with self.assertRaises(AssertionError):
                    verify_initial_predictions(root, 0)

    def test_real_capture_reconstructs_h10_and_safety_stop_has_no_fake_action(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = MineParkingNetV1(hidden_dim=8, max_horizon=4, dt=.2)
        env = ParkingEnvironment(world_size=30.)
        def checker(margin):
            return RectangleFootprintCollisionChecker(env, vehicle_length=6., vehicle_width=3.,
                                                     collision_margin=margin, resolution=.1)
        class Planner:
            _collision_checker = checker(.2)
        case = dict(index=0, task_id='fixture', scene_name='fixture', vehicle=MINING_DRILL_RIG,
            pipeline=Pipeline(), planner=Planner(), start=VehicleState(0., 0., 0.),
            goal=GoalPose(8., 0., 0.), engine_kwargs=dict(vehicle_model=DifferentialDriveModel(),
                terminal=TerminalChecker(), env=env, vehicle_length=6., vehicle_width=3.,
                collision_checker=checker(0.)))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'capture'
            summary = record_feasible_branch(case, model, output, Budget(), branch='network_fixed', max_steps=2)
            self.assertEqual(summary['steps'], 2)
            detail = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual(audit_execution(output, detail, case)['selected_h10_transitions'], 20)
            broken = deepcopy(detail)
            broken['frames'][0]['command'][0] += .01
            with self.assertRaises(AssertionError):
                audit_execution(output, broken, case)
            broken = deepcopy(detail)
            broken['searches'][0]['generations'] = []
            with self.assertRaises(ValueError):
                audit_execution(output, broken, case)
            with np.load(output.with_suffix('.npz'), allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            arrays['selected_control_rollouts'][0, 9, 0] += .01
            np.savez_compressed(output.with_suffix('.npz'), **arrays)
            with self.assertRaises(AssertionError):
                audit_execution(output, detail, case)
            with patch.object(FeasibleCEMController, '_solve', side_effect=ControlSafetyError('fixture stop')):
                summary = record_feasible_branch(case, model, Path(temporary) / 'stop', Budget(),
                                                  branch='network_fixed', max_steps=2)
            self.assertFalse(summary['success'])
            self.assertTrue(summary['safety_stop'])
            self.assertEqual(summary['steps'], 0)
            self.assertEqual(summary['control_timing']['calls'], 1)


if __name__ == '__main__':
    unittest.main()
