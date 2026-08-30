"""基于 BEV occupancy 的可微完整车体连续扫掠安全损失。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SafetyGeometry:
    """从数据集元数据解析出的局部 BEV 与车辆安全几何。"""

    vehicle_length_m: float
    vehicle_width_m: float
    collision_margin_m: float
    bev_resolution_m: float
    bev_extent_m: tuple[float, float, float, float]
    occupancy_channel: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safety_geometry_from_dataset(data: dict) -> SafetyGeometry:
    """严格从 schema v2 元数据解析统一几何，不使用场景特例或默认车型。"""
    if int(data.get("schema_version", -1)) != 2:
        raise ValueError("碰撞安全损失要求 schema v2 数据集")
    bev_meta = data.get("bev_meta")
    task_meta = data.get("task_meta")
    if not isinstance(bev_meta, dict) or not isinstance(task_meta, list) or not task_meta:
        raise ValueError("碰撞安全损失要求完整 bev_meta 与 task_meta")
    channels = bev_meta.get("channels")
    extent = bev_meta.get("extent")
    if not isinstance(channels, list) or "occupancy" not in channels:
        raise ValueError("碰撞安全损失要求 BEV occupancy 通道")
    if not isinstance(extent, list) or len(extent) != 4:
        raise ValueError("碰撞安全损失要求四方向 BEV extent")

    models: list[dict[str, Any]] = []
    for index, metadata in enumerate(task_meta):
        dataset_meta = metadata.get("dataset") if isinstance(metadata, dict) else None
        vehicle = dataset_meta.get("vehicle_model") if isinstance(dataset_meta, dict) else None
        if not isinstance(vehicle, dict):
            raise ValueError(f"样本 {index} 缺少 dataset.vehicle_model")
        models.append(vehicle)
    first = models[0]
    if any(model != first for model in models[1:]):
        raise ValueError("同一训练归档不得混合车辆模型")
    try:
        values = {
            "vehicle_length_m": float(first["length"]),
            "vehicle_width_m": float(first["width"]),
            "collision_margin_m": float(first["collision_margin"]),
            "bev_resolution_m": float(bev_meta["resolution"]),
            "bev_extent_m": tuple(float(value) for value in extent),
            "occupancy_channel": channels.index("occupancy"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("数据集车辆或 BEV 安全几何无效") from exc
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            values["vehicle_length_m"],
            values["vehicle_width_m"],
            values["bev_resolution_m"],
            *values["bev_extent_m"],
        )
    ) or not math.isfinite(values["collision_margin_m"]) or values["collision_margin_m"] < 0.0:
        raise ValueError("数据集车辆或 BEV 安全几何必须为有限正数")
    return SafetyGeometry(**values)


class SweptFootprintLoss(nn.Module):
    """对局部轨迹的完整矩形和相邻点连续扫掠计算占用/越界惩罚。"""

    def __init__(
        self,
        geometry: SafetyGeometry,
        *,
        extra_margin_m: float = 0.1,
        sample_spacing_m: float = 0.5,
        max_swept_substeps: int = 16,
        out_of_bounds_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(extra_margin_m) or extra_margin_m < 0.0:
            raise ValueError("extra_margin_m 必须为有限非负数")
        if not math.isfinite(sample_spacing_m) or sample_spacing_m <= 0.0:
            raise ValueError("sample_spacing_m 必须为有限正数")
        if isinstance(max_swept_substeps, bool) or max_swept_substeps <= 0:
            raise ValueError("max_swept_substeps 必须为正整数")
        if not math.isfinite(out_of_bounds_weight) or out_of_bounds_weight < 0.0:
            raise ValueError("out_of_bounds_weight 必须为有限非负数")
        self.geometry = geometry
        self.extra_margin_m = float(extra_margin_m)
        self.sample_spacing_m = float(sample_spacing_m)
        self.max_swept_substeps = int(max_swept_substeps)
        self.out_of_bounds_weight = float(out_of_bounds_weight)
        footprint = self._build_footprint_samples()
        self.register_buffer("footprint_samples", footprint, persistent=False)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry.to_dict(),
            "extra_margin_m": self.extra_margin_m,
            "sample_spacing_m": self.sample_spacing_m,
            "max_swept_substeps": self.max_swept_substeps,
            "out_of_bounds_weight": self.out_of_bounds_weight,
        }

    def forward(
        self, bev: torch.Tensor, points: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if bev.ndim != 4 or points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("安全损失要求 bev=(B,C,H,W)、points=(B,T,3)")
        if mask.shape != points.shape[:2] or bev.shape[0] != points.shape[0]:
            raise ValueError("安全损失 batch/mask 形状不一致")
        if self.geometry.occupancy_channel >= bev.shape[1]:
            raise ValueError("occupancy 通道索引超出 BEV")

        poses, swept_mask = self._swept_poses(points, mask)
        footprint = self.footprint_samples.to(dtype=points.dtype, device=points.device)
        cos_yaw = torch.cos(poses[..., 2]).unsqueeze(-1)
        sin_yaw = torch.sin(poses[..., 2]).unsqueeze(-1)
        local_x = footprint[:, 0]
        local_y = footprint[:, 1]
        sample_x = poses[..., 0].unsqueeze(-1) + cos_yaw * local_x - sin_yaw * local_y
        sample_y = poses[..., 1].unsqueeze(-1) + sin_yaw * local_x + cos_yaw * local_y

        front, back, left, right = self.geometry.bev_extent_m
        grid_x = 2.0 * (sample_y + right) / (left + right) - 1.0
        grid_y = 2.0 * (front - sample_x) / (front + back) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)

        occupancy = bev[:, self.geometry.occupancy_channel : self.geometry.occupancy_channel + 1]
        dilation_cells = max(
            0,
            int(math.ceil(self.extra_margin_m / self.geometry.bev_resolution_m)),
        )
        if dilation_cells:
            kernel = 2 * dilation_cells + 1
            occupancy = F.max_pool2d(occupancy, kernel, stride=1, padding=dilation_cells)
        batch, swept_steps, samples = grid.shape[:3]
        sampled = F.grid_sample(
            occupancy,
            grid.reshape(batch, swept_steps * samples, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch, swept_steps, samples)
        collision_risk = sampled.amax(dim=-1)
        overflow = torch.maximum(grid.abs() - 1.0, torch.zeros_like(grid))
        boundary_risk = overflow.amax(dim=(-1, -2))
        per_pose = collision_risk + self.out_of_bounds_weight * boundary_risk
        denominator = swept_mask.sum().clamp_min(1.0)
        return (per_pose * swept_mask).sum() / denominator

    def _swept_poses(
        self, points: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin = torch.zeros_like(points[:, :1])
        starts = torch.cat((origin, points[:, :-1]), dim=1)
        delta_xy = points[..., :2] - starts[..., :2]
        delta_yaw = torch.atan2(
            torch.sin(points[..., 2] - starts[..., 2]),
            torch.cos(points[..., 2] - starts[..., 2]),
        )
        half_length = (
            self.geometry.vehicle_length_m / 2.0
            + self.geometry.collision_margin_m
            + self.extra_margin_m
        )
        half_width = (
            self.geometry.vehicle_width_m / 2.0
            + self.geometry.collision_margin_m
            + self.extra_margin_m
        )
        radius = math.hypot(half_length, half_width)
        swept_distance = torch.linalg.vector_norm(delta_xy, dim=-1) + radius * delta_yaw.abs()
        active_distance = swept_distance.detach()[mask > 0]
        if active_distance.numel() == 0:
            substeps = 1
        else:
            substeps = max(
                1,
                min(
                    self.max_swept_substeps,
                    int(math.ceil(float(active_distance.max().cpu()) / self.geometry.bev_resolution_m)),
                ),
            )
        fractions = torch.linspace(
            1.0 / substeps,
            1.0,
            substeps,
            dtype=points.dtype,
            device=points.device,
        )
        xy = starts[..., :2, None] + delta_xy[..., :, None] * fractions
        yaw = starts[..., 2, None] + delta_yaw[..., None] * fractions
        poses = torch.cat((xy, yaw.unsqueeze(-2)), dim=-2).permute(0, 1, 3, 2)
        poses = poses.reshape(points.shape[0], points.shape[1] * substeps, 3)
        swept_mask = mask.unsqueeze(-1).expand(-1, -1, substeps).reshape(mask.shape[0], -1)
        return poses, swept_mask

    def _build_footprint_samples(self) -> torch.Tensor:
        half_length = (
            self.geometry.vehicle_length_m / 2.0
            + self.geometry.collision_margin_m
            + self.extra_margin_m
        )
        half_width = (
            self.geometry.vehicle_width_m / 2.0
            + self.geometry.collision_margin_m
            + self.extra_margin_m
        )
        count_x = max(2, int(math.ceil(2.0 * half_length / self.sample_spacing_m)) + 1)
        count_y = max(2, int(math.ceil(2.0 * half_width / self.sample_spacing_m)) + 1)
        xs = torch.linspace(-half_length, half_length, count_x)
        ys = torch.linspace(-half_width, half_width, count_y)
        mesh_x, mesh_y = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack((mesh_x.reshape(-1), mesh_y.reshape(-1)), dim=1)


__all__ = ["SafetyGeometry", "SweptFootprintLoss", "safety_geometry_from_dataset"]
