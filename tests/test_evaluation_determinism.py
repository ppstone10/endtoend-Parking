"""闭环评测确定性守卫测试。

背景：本项目曾误判"闭环评测有 3pt 随机噪声"。实测同一命令连续两次运行的
`summary.json` 完全一致（成功率、碰撞率、终点误差、失败分类逐项相同），
差异其实来自**换了 checkpoint**（`best_closed_loop.pt` 未校准阈值 vs
`deployment.pt` 已校准阈值，实测相差约 3pt）。

这组测试锁定"同一 checkpoint + 同一索引 → 同一结论"这一性质，
避免以后把 checkpoint 差异误读成评测噪声，也避免引入真正的非确定性。
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from controller import MPCController
from metrics import summarize
from runtime import ClosedLoopEngine, NetworkSource, TerminalChecker
from sim import DifferentialDriveModel, MINING_DRILL_RIG


class _ConstantSource:
    """给出固定直线的轨迹源，用于隔离控制层自身的确定性。"""

    def __init__(self, points: np.ndarray) -> None:
        from interfaces import Trajectory

        self._trajectory = Trajectory(points=np.asarray(points, dtype=np.float32), dt=0.2)

    def begin(self, start, goal) -> None:  # noqa: ANN001 - 协议签名
        return None

    def next_trajectory(self, state):  # noqa: ANN001 - 协议签名
        return self._trajectory, 0.0


def _run_episode(seed: int):
    from interfaces import GoalPose, VehicleState

    points = np.array(
        [[i * 0.5, 0.0, 0.0] for i in range(1, 41)], dtype=np.float64
    )
    source = _ConstantSource(points)
    engine = ClosedLoopEngine(
        vehicle_model=DifferentialDriveModel(**MINING_DRILL_RIG.vehicle_model_kwargs()),
        mpc=MPCController(
            dt=0.1,
            horizon=10,
            seed=seed,
            **MINING_DRILL_RIG.mpc_kwargs(),
        ),
        source=source,
        terminal=TerminalChecker(0.3, np.deg2rad(10.0)),
        replan_every=10,
        max_steps=120,
    )
    result = engine.run(VehicleState(0.0, 0.0, 0.0), GoalPose(18.0, 0.0, 0.0))
    result.record = None
    return result


class TestMpcDeterminism(unittest.TestCase):
    def test_same_seed_reproduces_identical_episode(self):
        first = _run_episode(7)
        second = _run_episode(7)
        self.assertEqual(first.steps, second.steps)
        self.assertEqual(first.failure, second.failure)
        self.assertAlmostEqual(first.final_pos_err, second.final_pos_err, places=12)
        self.assertAlmostEqual(first.final_yaw_err, second.final_yaw_err, places=12)

    def test_different_seed_can_change_the_rollout(self):
        """反向验证：seed 确实在起作用，否则"确定性"可能只是因为流程没用到随机源。"""
        signatures = set()
        for seed in range(6):
            result = _run_episode(seed)
            signatures.add(
                (
                    result.steps,
                    round(float(result.final_pos_err), 9),
                    round(float(result.final_yaw_err), 9),
                )
            )
        self.assertGreater(
            len(signatures), 1, "不同 seed 应产生不同结果，否则 RNG 未真正参与控制"
        )

    def test_summarize_is_deterministic_for_repeated_results(self):
        results = [_run_episode(seed) for seed in (3, 5)]
        first = summarize(results)
        second = summarize(results)
        for key in ("success_rate", "collision_rate", "final_pos_err_mean"):
            self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
