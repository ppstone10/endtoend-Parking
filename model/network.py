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
import torch.nn.functional as F

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


def endpoint_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    tail_points: int = 8,
) -> torch.Tensor:
    """近端终点对齐监督：对预测轨迹末端加权逼近专家轨迹末端。

    车辆接近目标时净空损失会诱导网络"缩短轨迹躲避障碍"，使停止长度
    提前触发、终点距目标与航向误差膨胀（闭环振荡根因）。此损失对 mask
    末端 ``tail_points`` 个有效点施加加权的专家轨迹模仿，恢复"逼近目标
    但保留 MPC 收敛余量"的 v7 轨迹模式；权重越高越强调近端收敛。
    使用专家轨迹 ``target`` 而非目标位姿做监督，避免强制轨迹塌缩到终点。
    """
    if pred.ndim != 3 or pred.shape[-1] != 3 or target.ndim != 3 or target.shape != pred.shape:
        raise ValueError("endpoint_alignment_loss 要求 pred/target=(B,T,3) 且形状一致")
    if mask.shape != pred.shape[:2]:
        raise ValueError("endpoint_alignment_loss batch/mask 形状不一致")
    if isinstance(tail_points, bool) or not isinstance(tail_points, int) or tail_points <= 0:
        raise ValueError("tail_points 必须为正整数")
    lengths = mask.sum(dim=1).long()
    steps = torch.arange(pred.shape[1], device=pred.device).unsqueeze(0)
    tail_begin = (lengths - tail_points).clamp(min=0).unsqueeze(1)
    tail_span = (steps >= tail_begin) & (steps < lengths.unsqueeze(1))
    end_idx = steps.masked_select(tail_span)
    batch_idx = tail_span.nonzero(as_tuple=False)[:, 0]
    tail_pred = pred[batch_idx, end_idx]
    tail_target = target[batch_idx, end_idx]
    pos_err = (tail_pred[..., :2] - tail_target[..., :2]).pow(2).sum(-1)
    yaw_err = (tail_pred[..., 2] - tail_target[..., 2]).pow(2)
    per_point = pos_err + yaw_err
    denominator = tail_span.sum().clamp(min=1.0)
    return per_point.sum() / denominator


def variable_loss_fn(
    pred: torch.Tensor,
    stop_logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    stop_weight: float = 0.2,
    balance_stop: bool = True,
    stop_target_mode: str = "terminal",
) -> torch.Tensor:
    """变长轨迹损失：掩码轨迹 MSE + 有效前缀终止 BCE。

    ``balance_stop`` 按当前 batch 有效前缀中的负/正样本比提高终点权重，
    避免长轨迹里单个终点被大量“继续”标签淹没。
    """
    trajectory_loss = loss_fn(pred, target, mask)
    if stop_target_mode not in {"terminal", "cumulative"}:
        raise ValueError("stop_target_mode 必须为 terminal 或 cumulative")
    lengths = mask.sum(dim=1).long()
    steps = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    valid_sequences = lengths > 0
    prefix_mask = steps < lengths.unsqueeze(1)
    stop_targets = torch.zeros_like(stop_logits)
    if stop_target_mode == "cumulative":
        stop_targets = (
            valid_sequences.unsqueeze(1)
            & (steps >= (lengths - 1).clamp(min=0).unsqueeze(1))
        ).to(stop_logits.dtype)
        supervision_mask = valid_sequences.unsqueeze(1).expand_as(stop_logits)
    else:
        rows = torch.nonzero(valid_sequences, as_tuple=False).squeeze(1)
        if rows.numel():
            stop_targets[rows, lengths[rows] - 1] = 1.0
        supervision_mask = prefix_mask
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits, stop_targets, reduction="none"
    )
    effective_weights = supervision_mask.to(stop_loss.dtype)
    if balance_stop:
        positive_count = (stop_targets * effective_weights).sum()
        negative_count = ((1.0 - stop_targets) * effective_weights).sum()
        positive_weight = torch.where(
            positive_count > 0,
            negative_count / positive_count.clamp(min=1.0),
            positive_count.new_tensor(1.0),
        ).clamp(min=1.0)
        effective_weights = effective_weights * torch.where(
            stop_targets > 0,
            positive_weight,
            stop_targets.new_tensor(1.0),
        )
    stop_denom = effective_weights.sum().clamp(min=1.0)
    stop_loss = (stop_loss * effective_weights).sum() / stop_denom
    return trajectory_loss + float(stop_weight) * stop_loss
