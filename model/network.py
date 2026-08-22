"""MineParkingNet 端到端轨迹生成网络。

输入：BEV 环境表示 (C,H,W)、目标泊车位姿 (x,y,yaw)、车辆运动状态 (v,omega)。
输出：未来 N 个局部轨迹点 (N,3)，列 [x,y,yaw]。

结构：BEV 经 CNN 编码展平为特征向量，与目标位姿、运动状态拼接后经 MLP
回归 N×3 轨迹点。轨迹为车辆中心局部坐标。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState


class MineParkingNet(nn.Module):
    """端到端轨迹生成网络。"""

    def __init__(
        self,
        bev_channels: int = 5,
        horizon: int = 20,
        dt: float = 0.1,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.dt = dt

        # BEV 编码器：3 层卷积下采样。
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 目标位姿 (3) + 运动状态 (v, omega) + 全局场景特征：占位长度由编码输出决定。
        self.fc_out = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.LazyLinear(horizon * 3),
        )

    def forward(
        self,
        bev: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """前向计算。

        bev: (B, C, H, W)；goal: (B, 3)；state: (B, 2)（v, omega）。
        返回 (B, N, 3) 局部轨迹点。
        """
        features = self.bev_encoder(bev)
        features = features.flatten(1)
        cond = torch.cat([goal, state], dim=1)
        # 将条件广播到特征维度：先拼接再经全连接，LazyLinear 自动推断维度。
        x = torch.cat([features, cond], dim=1)
        out = self.fc_out(x)
        return out.view(-1, self.horizon, 3)

    # ------------------------------------------------------------------
    # 接口适配：numpy 数据进出
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, bev: BEVTensor, goal: GoalPose, state: VehicleState
    ) -> Trajectory:
        """预测未来轨迹（局部坐标）。"""
        self.eval()
        bev_t = torch.as_tensor(bev.data, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.as_tensor(
            [goal.x, goal.y, goal.yaw], dtype=torch.float32
        ).unsqueeze(0)
        state_t = torch.as_tensor([state.v, state.omega], dtype=torch.float32).unsqueeze(0)
        out = self.forward(bev_t, goal_t, state_t)
        return Trajectory(points=out[0].numpy(), dt=self.dt)


def loss_fn(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE 轨迹损失，按有效点掩码平均。

    pred/target: (B, N, 3)；mask: (B, N)（1 表示该点有效）。
    """
    diff = (pred - target) ** 2
    diff = diff * mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom