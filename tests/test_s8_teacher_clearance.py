import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from experiments.s8_teacher_clearance import clearance_teacher_planner
from planner.hybrid_astar import HybridAStarPlanner
from sim import ParkingEnvironment


class TeacherClearanceTests(unittest.TestCase):
    def test_independent_teacher_keeps_execution_planner_and_metadata(self):
        original=HybridAStarPlanner(ParkingEnvironment(),collision_margin=.2,
            vehicle_model_metadata={'collision_margin':.2,'marker':'original'})
        teacher=clearance_teacher_planner(original)
        self.assertIsNot(teacher,original)
        self.assertIsNot(teacher.env,original.env)
        self.assertIsNot(teacher._collision_checker,original._collision_checker)
        self.assertEqual(original.collision_margin,.2)
        self.assertEqual(original.model_metadata()['collision_margin'],.2)
        self.assertEqual(teacher.collision_margin,.3)
        self.assertEqual(teacher.model_metadata()['collision_margin'],.3)
        self.assertAlmostEqual(teacher._collision_checker.half_width-original._collision_checker.half_width,.1)
        self.assertEqual(teacher.trajectory_dt,original.trajectory_dt)
        self.assertEqual(teacher.max_planning_time_s,original.max_planning_time_s)
        self.assertTrue(np.array_equal(teacher.omega_values,original.omega_values))

    def test_unexpected_baseline_margin_is_rejected(self):
        original=HybridAStarPlanner(ParkingEnvironment(),collision_margin=0.)
        with self.assertRaises(ValueError):clearance_teacher_planner(original)

    def test_inflated_start_or_goal_infeasible_never_executes(self):
        from scripts import preflight_s8_teacher_clearance as cli
        from interfaces import GoalPose,VehicleState
        original=HybridAStarPlanner(ParkingEnvironment(),collision_margin=.2)
        case={'planner':original,'goal':GoalPose(0,0,0)}
        context=SimpleNamespace(state=VehicleState(0,0,0),step=0)
        for safety in ((False,True),(True,False)):
            teacher=clearance_teacher_planner(original);row={'attempts':[]}
            with patch.object(cli,'clearance_teacher_planner',return_value=teacher), \
                    patch.object(teacher._collision_checker,'pose_free',side_effect=safety), \
                    patch.object(teacher,'plan') as plan,patch.object(cli,'run_case') as execute:
                result=cli.teacher_query(case,context,None,None,None,'test',None,None,row,lambda:None)
                self.assertIsNone(result);plan.assert_not_called();execute.assert_not_called()
                self.assertEqual(row['decision'],'inflated_start_or_goal_infeasible')

    def test_precheck_and_budget_failure_do_not_execute(self):
        from scripts import preflight_s8_teacher_clearance as cli
        with patch.object(cli,'prepare_config',return_value=({},{})),patch.object(cli,'run_case') as run:
            cli.precheck();run.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'new'
            with patch.object(cli,'prepare_config',side_effect=TimeoutError('budget')),patch.object(cli,'run_case') as run:
                with self.assertRaises(TimeoutError):cli.run(target)
                run.assert_not_called()
            report=json.loads((target/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertEqual(report['training_updates'],0)
            self.assertFalse(report['allowed_to_prepare_training_plan'])
            with self.assertRaises(FileExistsError):cli.run(target)


if __name__=='__main__':unittest.main()
