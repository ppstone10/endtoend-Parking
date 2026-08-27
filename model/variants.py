"""MineParkingNet v1/v2 变长轨迹模型。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from interfaces import BEVTensor, GoalPose, Trajectory, VehicleState

from .prediction import TrajectoryPrediction


class _ConditionedGRUDecoder(nn.Module):
    """以场景条件初始化并逐点解码的共享 GRU。"""

    def __init__(self, context_dim: int, hidden_dim: int, max_horizon: int) -> None:
        super().__init__()
        if max_horizon <= 0:
            raise ValueError("max_horizon 必须为正")
        self.max_horizon = int(max_horizon)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.input_projection = nn.Linear(context_dim + 3, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.point_head = nn.Linear(hidden_dim, 3)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        *,
        teacher_points: torch.Tensor | None = None,
    ) -> TrajectoryPrediction:
        if teacher_points is not None:
            if teacher_points.ndim != 3 or teacher_points.shape[2] != 3:
                raise ValueError("teacher_points 必须是 (B,N,3)")
            if teacher_points.shape[0] != context.shape[0]:
                raise ValueError("teacher_points batch 必须与 context 一致")
        hidden = torch.tanh(self.context_projection(context))
        previous = context.new_zeros((context.shape[0], 3))
        points: list[torch.Tensor] = []
        stop_logits: list[torch.Tensor] = []
        for step in range(self.max_horizon):
            decoder_input = torch.tanh(
                self.input_projection(torch.cat([context, previous], dim=1))
            )
            hidden = self.gru_cell(decoder_input, hidden)
            point = self.point_head(hidden)
            points.append(point)
            stop_logits.append(self.stop_head(hidden).squeeze(1))
            if teacher_points is not None and step < teacher_points.shape[1]:
                previous = teacher_points[:, step]
            else:
                previous = point
        return TrajectoryPrediction(
            points=torch.stack(points, dim=1),
            stop_logits=torch.stack(stop_logits, dim=1),
        )


class _VariableTrajectoryMixin:
    max_horizon: int
    dt: float
    stop_threshold: float

    @torch.no_grad()
    def predict(
        self, bev: BEVTensor, goal: GoalPose, state: VehicleState
    ) -> Trajectory:
        self.eval()
        device = next(self.parameters()).device
        bev_t = torch.as_tensor(bev.data, dtype=torch.float32, device=device).unsqueeze(0)
        goal_t = torch.as_tensor(
            [goal.x, goal.y, goal.yaw], dtype=torch.float32, device=device
        ).unsqueeze(0)
        state_t = torch.as_tensor(
            [state.v, state.omega], dtype=torch.float32, device=device
        ).unsqueeze(0)
        prediction = self.forward_with_stop(bev_t, goal_t, state_t)
        probabilities = torch.sigmoid(prediction.stop_logits[0])
        indices = torch.nonzero(probabilities >= self.stop_threshold, as_tuple=False)
        length = int(indices[0, 0]) + 1 if indices.numel() else self.max_horizon
        points = prediction.points[0, :length].detach().cpu().numpy().astype(np.float32)
        return Trajectory(points=points, dt=self.dt)


class MineParkingNetV1(_VariableTrajectoryMixin, nn.Module):
    """CNN 条件编码 + GRU 变长轨迹解码。"""

    def __init__(
        self,
        bev_channels: int = 5,
        max_horizon: int = 60,
        dt: float = 0.1,
        hidden_dim: int = 128,
        stop_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < stop_threshold < 1.0:
            raise ValueError("stop_threshold 必须在 (0,1) 内")
        self.max_horizon = int(max_horizon)
        self.dt = float(dt)
        self.stop_threshold = float(stop_threshold)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.condition = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = _ConditionedGRUDecoder(
            hidden_dim, hidden_dim, self.max_horizon
        )

    def _context(
        self, bev: torch.Tensor, goal: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        scene = self.bev_encoder(bev).flatten(1)
        return self.condition(torch.cat([scene, goal, state], dim=1))

    def forward_with_stop(
        self,
        bev: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
        *,
        teacher_points: torch.Tensor | None = None,
    ) -> TrajectoryPrediction:
        return self.decoder(
            self._context(bev, goal, state), teacher_points=teacher_points
        )

    def forward(
        self, bev: torch.Tensor, goal: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_with_stop(bev, goal, state).points


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class MineParkingNetV2(_VariableTrajectoryMixin, nn.Module):
    """U-Net 跳连空间编码 + goal/state 交叉注意力 + GRU 解码。"""

    def __init__(
        self,
        bev_channels: int = 5,
        max_horizon: int = 60,
        dt: float = 0.1,
        hidden_dim: int = 128,
        base_channels: int = 32,
        attention_heads: int = 4,
        stop_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim 必须能被 attention_heads 整除")
        if not 0.0 < stop_threshold < 1.0:
            raise ValueError("stop_threshold 必须在 (0,1) 内")
        self.max_horizon = int(max_horizon)
        self.dt = float(dt)
        self.stop_threshold = float(stop_threshold)
        self.enc1 = _ConvBlock(bev_channels, base_channels)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
        self.down = nn.MaxPool2d(2)
        self.fuse2 = _ConvBlock(base_channels * 6, base_channels * 2)
        self.fuse1 = _ConvBlock(base_channels * 3, base_channels)
        self.token_projection = nn.Conv2d(base_channels, hidden_dim, 1)
        self.condition_query = nn.Linear(5, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.decoder = _ConditionedGRUDecoder(
            hidden_dim, hidden_dim, self.max_horizon
        )

    def _context(
        self, bev: torch.Tensor, goal: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        skip1 = self.enc1(bev)
        skip2 = self.enc2(self.down(skip1))
        bottleneck = self.bottleneck(self.down(skip2))
        up2 = F.interpolate(
            bottleneck, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        fused2 = self.fuse2(torch.cat([up2, skip2], dim=1))
        up1 = F.interpolate(
            fused2, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        fused1 = self.fuse1(torch.cat([up1, skip1], dim=1))
        tokens = self.token_projection(fused1).flatten(2).transpose(1, 2)
        query = self.condition_query(torch.cat([goal, state], dim=1)).unsqueeze(1)
        attended, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        return self.context_norm(attended.squeeze(1) + query.squeeze(1))

    def forward_with_stop(
        self,
        bev: torch.Tensor,
        goal: torch.Tensor,
        state: torch.Tensor,
        *,
        teacher_points: torch.Tensor | None = None,
    ) -> TrajectoryPrediction:
        return self.decoder(
            self._context(bev, goal, state), teacher_points=teacher_points
        )

    def forward(
        self, bev: torch.Tensor, goal: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_with_stop(bev, goal, state).points
