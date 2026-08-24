"""车辆参数配置与可视化冒烟测试。"""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # 无显示环境渲染

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from runtime.recorder import EpisodeRecord
from sim import MINING_TRUCK, ParkingEnvironment, RectangleObstacle, VehicleConfig, get_vehicle
from viz import draw_trajectory, render_episode, render_world


class TestVehicleConfig(unittest.TestCase):
    def test_presets_available(self):
        truck = get_vehicle("mining_truck")
        self.assertEqual((truck.length, truck.width), (6.0, 3.0))
        legacy = get_vehicle("legacy_4x2")
        self.assertEqual((legacy.length, legacy.width), (4.0, 2.0))

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            get_vehicle("no_such_vehicle")

    def test_kwargs_consistency(self):
        """三类 kwargs 与 config 字段一致，保证尺寸/上限统一注入。"""
        cfg = MINING_TRUCK
        self.assertEqual(cfg.planner_kwargs(), {"vehicle_length": 6.0, "vehicle_width": 3.0})
        self.assertEqual(cfg.mpc_kwargs(), {"max_v": 2.0, "max_omega": 1.0})
        self.assertEqual(cfg.vehicle_model_kwargs(), {"max_v": 2.0, "max_omega": 1.0})
        self.assertEqual(cfg.collision_kwargs(), {"vehicle_length": 6.0, "vehicle_width": 3.0})

    def test_frozen(self):
        cfg = VehicleConfig("t", 1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(Exception):
            cfg.length = 2.0


class TestVizSmoke(unittest.TestCase):
    def _env(self):
        return ParkingEnvironment(
            world_size=40.0,
            obstacles=[RectangleObstacle(-15.0, 15.0, -6.0, -2.0), RectangleObstacle(-15.0, 15.0, 2.0, 6.0)],
        )

    def _record(self):
        record = EpisodeRecord()
        from interfaces import ControlCmd

        traj = Trajectory(points=np.array([[0, 0, 0], [1, -1, -0.5], [2, -2, -1.0]]), dt=0.1)
        for i in range(10):
            state = VehicleState(i * 0.2, -i * 0.2, -0.05 * i)
            cmd = ControlCmd(0.2, -0.05)
            record.log(state, cmd, traj, traj, False)
        return record

    def test_render_world_and_traj(self):
        import matplotlib.pyplot as plt

        from viz import setup_style

        setup_style()
        fig, ax = plt.subplots()
        env = self._env()
        render_world(ax, env, spots=[GoalPose(5.0, 0.0, 0.0)])
        draw_trajectory(ax, np.array([[0, 0], [1, 1], [2, 2]]), kind="actual", label="t")
        plt.close(fig)

    def test_render_episode_saves_files(self):
        record = self._record()
        env = self._env()
        goal = GoalPose(2.0, -2.0, -1.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "episode")
            fig = render_episode(
                record, env, goal,
                vehicle_length=MINING_TRUCK.length, vehicle_width=MINING_TRUCK.width,
                out_path=out,
            )
            import matplotlib.pyplot as plt

            plt.close(fig)
            self.assertTrue(os.path.exists(out + ".png"))
            self.assertTrue(os.path.exists(out + ".pdf"))


if __name__ == "__main__":
    unittest.main()
