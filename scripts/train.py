"""阶段四演示：生成小批量数据并训练 MineParkingNet。

用法：
    python scripts/train.py [--samples N] [--epochs E] [--data DATA.npz]

数据坐标约定：专家轨迹与目标为全局坐标，训练前转换到车辆起始局部系
（与 BEV 一致）。网络输出局部轨迹点。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dataset import DatasetGenerator, SensorBEVPipeline
from interfaces import CameraIntrinsics
from model import MineParkingNet, loss_fn
from planner import HybridAStarPlanner
from sensor2bev import BEVFusion, Camera2BEV, LiDAR2BEV
from sim import ParkingEnvironment, RectangleObstacle, SimulatedCamera, SimulatedLiDAR


def build_env() -> ParkingEnvironment:
    return ParkingEnvironment(
        world_size=40.0,
        obstacles=[
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=-6.0, y_max=-2.0),
            RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=2.0, y_max=6.0),
        ],
    )


def build_pipeline(env) -> SensorBEVPipeline:
    intrinsics = CameraIntrinsics(
        fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
    )
    return SensorBEVPipeline(
        lidar_sensor=SimulatedLiDAR(env, beams=360, max_range=20.0),
        camera_sensor=SimulatedCamera(env, intrinsics),
        lidar2bev=LiDAR2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
        camera2bev=Camera2BEV(resolution=0.2, extent=(10.0, 10.0, 10.0, 10.0)),
        bev_fusion=BEVFusion(),
    )


def to_local(points: np.ndarray, sx: float, sy: float, syaw: float) -> np.ndarray:
    """全局点转换到以 (sx, sy, syaw) 为原点的局部系。"""
    dx = points[:, 0] - sx
    dy = points[:, 1] - sy
    cos_yaw, sin_yaw = np.cos(syaw), np.sin(syaw)
    local = np.empty_like(points)
    local[:, 0] = cos_yaw * dx + sin_yaw * dy
    local[:, 1] = -sin_yaw * dx + cos_yaw * dy
    local[:, 2] = points[:, 2] - syaw
    return local


def prepare_batches(data: dict[str, np.ndarray], horizon: int, batch_size: int, device):
    """将数据转为局部坐标并按 batch 划分。"""
    bevs = data["bevs"]
    goals = data["goals"]
    states = data["states"]
    trajs = data["trajs"]
    masks = data["masks"]
    dt = float(data["dt"][0])

    n = bevs.shape[0]
    traj_len = trajs.shape[1]
    horizon_eff = min(horizon, traj_len)

    # 每条样本以起始位姿为原点转换目标与轨迹。
    local_goals = np.empty_like(goals)
    local_trajs = np.empty_like(trajs)
    for i in range(n):
        sx, sy, syaw = states[i, 0], states[i, 1], states[i, 2]
        local_goals[i] = to_local(goals[i : i + 1], sx, sy, syaw)[0]
        local_trajs[i] = to_local(trajs[i], sx, sy, syaw)

    batches = []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        batches.append(
            (
                torch.as_tensor(bevs[sl], dtype=torch.float32, device=device),
                torch.as_tensor(local_goals[sl], dtype=torch.float32, device=device),
                torch.as_tensor(states[sl][:, 3:5], dtype=torch.float32, device=device),
                torch.as_tensor(local_trajs[sl][:, :horizon_eff], dtype=torch.float32, device=device),
                torch.as_tensor(masks[sl][:, :horizon_eff], dtype=torch.float32, device=device),
            )
        )
    return batches, horizon_eff, dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cpu")

    if args.data:
        from dataset import DatasetGenerator as _DG

        data = _DG.load(args.data)
    else:
        env = build_env()
        planner = HybridAStarPlanner(env=env)
        generator = DatasetGenerator(
            env=env, planner=planner, sensor_pipeline=build_pipeline(env)
        )
        samples = generator.generate(count=args.samples)
        print(f"生成 {len(samples)} 条样本，开始训练")
        np_path = "data_training.npz"
        generator.save(samples, np_path)
        print(f"数据集已保存到 {np_path}")
        data = DatasetGenerator.load(np_path)

    batches, horizon_eff, dt = prepare_batches(data, args.horizon, args.batch_size, device)
    model = MineParkingNet(bev_channels=data["bevs"].shape[1], horizon=horizon_eff, dt=dt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"样本 {data['bevs'].shape[0]} 条，batch {len(batches)} 个，horizon {horizon_eff}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bev, goal, state, traj, mask in batches:
            optimizer.zero_grad()
            pred = model.forward(bev, goal, state)
            loss = loss_fn(pred, traj, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {total_loss / len(batches):.4f}")

    torch.save(model.state_dict(), "mineparkingnet.pt")
    print("模型已保存到 mineparkingnet.pt")


if __name__ == "__main__":
    main()