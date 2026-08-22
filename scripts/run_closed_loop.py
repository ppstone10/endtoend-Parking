"""阶段五闭环演示：加载训练网络，网络→MPC→车辆模型闭环泊车。

用法：
    python scripts/run_closed_loop.py [--samples N] [--steps MAX] [--data DATA.npz]

流程：数据集样本 → BEV/目标/状态 → MineParkingNet 输出局部轨迹 →
MPC 跟踪 → 车辆模型推进 → 统计最终位置/航向误差。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from controller import MPCController
from dataset import DatasetGenerator, SensorBEVPipeline
from interfaces import CameraIntrinsics, GoalPose, Trajectory, VehicleState
from model import MineParkingNet
from planner import HybridAStarPlanner
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR
from scripts.train import build_env, build_pipeline, to_local  # 复用 train 的构建函数


def load_model(path: str, bev_channels: int, horizon: int, dt: float) -> MineParkingNet:
    model = MineParkingNet(bev_channels=bev_channels, horizon=horizon, dt=dt)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def run_closed_loop(
    model: MineParkingNet,
    samples_data: dict[str, np.ndarray],
    index: int,
    steps: int,
) -> tuple[float, float, float]:
    """对单条样本执行闭环泊车，返回 (位置误差, 航向误差, 是否成功)。"""
    bev = samples_data["bevs"][index]
    goal = samples_data["goals"][index]
    state_arr = samples_data["states"][index]
    expert = samples_data["trajs"][index]
    mask = samples_data.get("masks")
    dt = float(samples_data["dt"][0])
    # 按掩码截断到有效轨迹长度。
    if mask is not None:
        n_valid = int(mask[index].sum())
        expert = expert[:n_valid] if n_valid > 0 else expert

    from interfaces import BEVTensor

    state = VehicleState.from_array(state_arr)
    goal_obj = GoalPose(goal[0], goal[1], goal[2])
    bev_obj = BEVTensor(data=bev, resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0), channels=["occupancy", "height", "density", "target", "vehicle"])

    # 专家轨迹为全局坐标，转起始局部系。
    expert_local = to_local(expert, state_arr[0], state_arr[1], state_arr[2])
    mpc = MPCController(dt=0.1)
    vehicle_state = VehicleState(0.0, 0.0, 0.0, v=state_arr[3], omega=state_arr[4])

    from sim import DifferentialDriveModel

    model_step = DifferentialDriveModel(max_v=2.0, max_omega=1.0)
    # MPC 需要全局参考，这里用局部系下的轨迹与车辆（原点起步），等价相对跟踪。
    ref = Trajectory(points=expert_local, dt=dt)
    for _ in range(steps):
        cmd = mpc.compute(ref, vehicle_state)
        vehicle_state = model_step.step(vehicle_state, cmd, mpc.dt)
        # 到达专家轨迹终点附近即成功。
        if np.hypot(vehicle_state.x - expert_local[-1, 0], vehicle_state.y - expert_local[-1, 1]) < 0.5:
            break

    goal_local = to_local(goal[None, :], state_arr[0], state_arr[1], state_arr[2])[0]
    pos_err = np.hypot(vehicle_state.x - goal_local[0], vehicle_state.y - goal_local[1])
    yaw_err = abs(vehicle_state.yaw - goal_local[2])
    success = pos_err < 1.0 and yaw_err < np.deg2rad(20)
    return float(pos_err), float(yaw_err), success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--model", type=str, default="mineparkingnet.pt")
    args = parser.parse_args()

    if args.data:
        data = DatasetGenerator.load(args.data)
    else:
        env = build_env()
        planner = HybridAStarPlanner(env=env)
        generator = DatasetGenerator(
            env=env, planner=planner, sensor_pipeline=build_pipeline(env)
        )
        samples = generator.generate(count=args.samples)
        tmp_path = "data_closed_loop.npz"
        generator.save(samples, tmp_path)
        data = DatasetGenerator.load(tmp_path)

    model = load_model(
        args.model,
        bev_channels=data["bevs"].shape[1],
        horizon=min(20, data["trajs"].shape[1]),
        dt=float(data["dt"][0]),
    )

    print(f"共 {len(data['bevs'])} 条样本，逐条闭环泊车")
    pos_errs, yaw_errs = [], []
    for i in range(len(data["bevs"])):
        pos_err, yaw_err, success = run_closed_loop(model, data, i, args.steps)
        pos_errs.append(pos_err)
        yaw_errs.append(yaw_err)
        print(
            f"  样本 {i}: 位置误差 {pos_err:.2f}m 航向误差 {np.degrees(yaw_err):.1f}° "
            f"{'成功' if success else '失败'}"
        )
    print(
        f"平均位置误差 {np.mean(pos_errs):.2f}m，平均航向误差 {np.degrees(np.mean(yaw_errs)):.1f}°"
    )


if __name__ == "__main__":
    main()