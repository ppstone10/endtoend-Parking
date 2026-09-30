"""闭环五组对照实验矩阵测试：组合校验、周期重建与端到端骨架。"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from experiments.closed_loop_evaluation import ReconstructedDatasetTask
from experiments.validation_matrix import (
    EXPERIMENT_SPECS,
    ExperimentSpec,
    _control_steps_per_point,
    _cycles_from_record,
    run_validation_experiment,
)
from interfaces import GoalPose, Trajectory, VehicleState
from metrics import EpisodeResult
from metrics.rollout import CycleTracker
from runtime import EpisodeRecord
from sim import MINING_DRILL_RIG, Maneuver, NoiseLevel, TaskSampler, TaskType


class _StraightLineModel:
    """测试替身：沿局部 +x 直线匀速 0.5m/步，dt=0.2s，用于理想执行口径回归。

    带一层 5 通道 Conv2d，使 ``validate_model_dataset`` 的通道/时间步长校验走真实路径。
    """

    dt = 0.2
    model_name = "stub-straight-line"
    model_config = {"bev_channels": 5, "max_horizon": 30, "dt": 0.2}
    epoch = 0

    def __init__(self) -> None:
        # checkpoint 载入结果的 .model 指向网络本体，本替身自指。
        self.model = self
        self.conv = torch.nn.Conv2d(5, 8, kernel_size=3, padding=1)

    def modules(self):
        return [self.conv]

    def predict(self, bev, goal, state):
        points = np.stack(
            [
                0.5 * np.arange(1, 31, dtype=np.float64),
                np.zeros(30),
                np.zeros(30),
            ],
            axis=1,
        )
        return Trajectory(points=points, dt=self.dt)


class TestExperimentSpecs(unittest.TestCase):
    def test_all_registered_specs_are_valid(self):
        for name, spec in EXPERIMENT_SPECS.items():
            with self.subTest(name=name):
                spec.validate()

    def test_five_experiment_matrix_is_present(self):
        self.assertEqual(
            {EXPERIMENT_SPECS[name].executor for name in ("E1", "E2", "E3", "E4", "E5")},
            {"mpc_vehicle", "ideal_path"},
        )
        self.assertEqual(EXPERIMENT_SPECS["E2"].bev_source, "gt")
        self.assertEqual(EXPERIMENT_SPECS["E2"].executor, "ideal_path")
        self.assertEqual(EXPERIMENT_SPECS["E5"].bev_source, "sensor")
        self.assertEqual(EXPERIMENT_SPECS["E5"].trajectory_source, "network")

    def test_rejects_gt_bev_with_expert_planning(self):
        spec = ExperimentSpec(
            name="bad", bev_source="gt", trajectory_source="expert", executor="mpc_vehicle"
        )
        with self.assertRaisesRegex(ValueError, "不消费 BEV"):
            spec.validate()

    def test_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="bad", bev_source="lidar", trajectory_source="expert", executor="mpc_vehicle"
            ).validate()
        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="bad",
                bev_source="sensor",
                trajectory_source="expert",
                executor="mpc_vehicle",
                safety_mode="magic",
            ).validate()

    def test_rejects_safety_mode_on_expert_source(self):
        with self.assertRaisesRegex(ValueError, "安全门禁"):
            ExperimentSpec(
                name="bad",
                bev_source="sensor",
                trajectory_source="expert",
                executor="mpc_vehicle",
                safety_mode="hierarchical",
            ).validate()


class TestControlStepsPerPoint(unittest.TestCase):
    def test_default_aligns_network_dt_with_mpc_dt(self):
        self.assertAlmostEqual(_control_steps_per_point(None, 0.2, 0.1), 2.0)

    def test_explicit_value_wins(self):
        self.assertAlmostEqual(_control_steps_per_point(5.0, 0.2, 0.1), 5.0)

    def test_missing_horizon_dt_falls_back_to_one(self):
        self.assertAlmostEqual(_control_steps_per_point(None, 0.0, 0.1), 1.0)

    def test_non_positive_explicit_value_rejected(self):
        with self.assertRaises(ValueError):
            _control_steps_per_point(0.0, 0.2, 0.1)


def _episode_with_plans(plans: list[np.ndarray]) -> EpisodeResult:
    record = EpisodeRecord()
    for index, plan in enumerate(plans):
        state = VehicleState(float(index), 0.0, 0.0)
        record.log(state, None, Trajectory(plan, dt=0.2), Trajectory(plan, dt=0.2), False)
    return EpisodeResult(
        success=False,
        failure="timeout",
        steps=len(plans),
        final_pos_err=1.0,
        final_yaw_err=0.0,
        path_length=1.0,
        parking_time=0.1 * len(plans),
        tracking_rms=0.0,
        inference_ms=0.0,
        record=record,
    )


class TestCycleReconstruction(unittest.TestCase):
    def test_new_plan_starts_new_cycle(self):
        plan_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        plan_b = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _episode_with_plans([plan_a, plan_a, plan_b, plan_b, plan_b])
        tracker = CycleTracker()
        tracker.begin(GoalPose(5.0, 0.0, 0.0), VehicleState(0.0, 0.0, 0.0))
        _cycles_from_record(result, tracker, reference_dt=0.2)
        cycles = tracker.cycles()
        self.assertEqual(len(cycles), 2)
        self.assertEqual(cycles[0]["steps"], 2)
        self.assertEqual(cycles[1]["steps"], 3)


class TestEndToEndSkeleton(unittest.TestCase):
    """E1 组合端到端骨架：不读数据集文件，验证报告结构与过程证据接通。"""

    def _task_and_goal(self):
        sampler = TaskSampler(
            seed=20260824,
            vehicle_length=MINING_DRILL_RIG.length,
            vehicle_width=MINING_DRILL_RIG.width,
            collision_margin=MINING_DRILL_RIG.collision_margin,
        )
        task = sampler.sample(
            "S1_parking_lot",
            TaskType.T1_NEAR,
            0,
            maneuver=Maneuver.FORWARD,
            adjacent_occupancy=0,
            noise_level=NoiseLevel.CLEAN,
        )
        return task

    def _fake_dataset(self):
        """构造与 schema v2 载荷同构的假数据与复原任务。"""
        task = self._task_and_goal()
        goal_pose = task.goal.as_goal_pose()
        state = VehicleState(float(task.start.x), float(task.start.y), float(task.start.yaw))
        restored = ReconstructedDatasetTask(
            task=task,
            goal=goal_pose,
            goal_meta=task.goal.to_metadata(),
            tol_pos=float(task.goal.tol_pos),
            tol_yaw=float(task.goal.tol_yaw),
        )
        metadata = [
            {
                "scene_name": task.scene_name,
                "task_type": task.task_type.value,
                "task_id": task.task_id,
                "difficulty": {
                    "maneuver": task.difficulty.maneuver.value,
                    "noise_level": task.difficulty.noise_level.value,
                    "adjacent_occupancy": int(task.difficulty.adjacent_occupancy),
                },
            }
        ]
        payload = {
            "schema_version": 2,
            "task_meta": metadata,
            "states": np.asarray([[state.x, state.y, state.yaw, state.v, state.omega]]),
            "goals": np.asarray([[goal_pose.x, goal_pose.y, goal_pose.yaw]]),
            "bevs": np.zeros((1, 5, 160, 160), dtype=np.float32),
            "dt": np.asarray([0.2], dtype=np.float64),
        }
        manifest = {
            "schema_version": 1,
            "seed": 20260824,
            "vehicle_model": MINING_DRILL_RIG.to_metadata(),
        }
        return restored, payload, manifest, state

    def _run_with_mocks(self, experiment: str, model=None, max_steps: int = 200):
        restored, payload, manifest, _ = self._fake_dataset()
        with mock.patch(
            "dataset.DatasetGenerator.load", return_value=payload
        ), mock.patch(
            "experiments.validation_matrix.load_dataset_manifest", return_value=manifest
        ), mock.patch(
            "experiments.validation_matrix.reconstruct_dataset_task",
            return_value=restored,
        ), mock.patch(
            "experiments.validation_matrix.load_model_checkpoint", return_value=model
        ):
            return run_validation_experiment(
                experiment,
                data_path="fake/val.npz",
                checkpoint_path=("fake/deployment.pt" if model is not None else None),
                samples=1,
                max_steps=max_steps,
                replan_every=10,
            )

    def test_expert_mpc_run_produces_report_with_cycles(self):
        report = self._run_with_mocks("E1")
        overall = report["overall"]
        self.assertEqual(report["experiment"]["name"], "E1")
        self.assertEqual(overall["evaluated_samples"], 1)
        self.assertEqual(report["protocol"]["evaluated_indices"], [0])
        self.assertEqual(report["protocol"]["mpc_dt"], 0.1)
        self.assertIn("cycles_divergence_events_mean", overall)
        self.assertNotIn("ideal_executor_advance_failures", overall)
        episode = report["episodes"][0]
        self.assertGreater(episode["cycles"]["cycles"], 0)
        self.assertEqual(episode["executor_audit"]["kind"], "mpc_vehicle")
        self.assertEqual(report["reconstruct_failures"], [])

    def test_ideal_executor_spec_is_actually_used_by_the_engine(self):
        """回归：executor 必须传给引擎，否则 E2 会静默退化成 MPC 口径。"""
        report = self._run_with_mocks("E2", model=_StraightLineModel())
        episode = report["episodes"][0]
        self.assertEqual(episode["executor_audit"]["kind"], "ideal_path")
        self.assertEqual(report["protocol"]["ideal_steps_per_point"], 2.0)
        self.assertIn("ideal_executor_advance_failures", report["overall"])
        self.assertIn("cycles_divergence_events_mean", report["overall"])
        # 理想执行 + 直线替身轨迹：应一路收敛到目标附近，无发散、无前进失败。
        self.assertEqual(episode["executor_audit"]["advance_failures"], 0)
        self.assertEqual(report["overall"]["cycles_divergence_events_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
