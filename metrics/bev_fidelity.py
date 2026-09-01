"""感知→BEV 保真度指标（验证阶梯 L1）。

量化 Sensor2BEV 从传感器帧到统一 BEV 的过程中，对传感器可观测内容的还原
程度。用于在进入轨迹规划（L2）与跟踪（L3）之前，先隔离"感知/BEV 是否失真"。

比较口径：
- occupancy 通道：与"高分辨率 LiDAR 采样真值"（lidar truth）做 IoU/Precision/
  Recall；该真值由从同一位姿、相同 BEV 配置、高束数无噪声 LiDAR 投射得到，
  反映当前传感器配置下可感知占用的上限。几何真值（场景 emits_points 障碍全量
  栅格）作为辅助参考，反映"理想感知上限"，二者差异来自 LiDAR 稀疏采样与遮挡。
- target 通道：与目标车位矩形真值栅格做 IoU/命中率。
- height/density 通道：与真值栅格做逐栅格 MAE（诊断辅助，不作为主验收）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from interfaces import BEVConfig, BEVTensor, GoalPose
from sensor2bev import LiDAR2BEV
from sim.environment import ParkingEnvironment
from sim.sensor_sim import SimulatedLiDAR


@dataclass(frozen=True)
class BEVFidelityMetrics:
    """单个 BEV 帧的感知→BEV 保真度指标。"""

    samples: int = 0
    occupancy_iou: float = 0.0
    occupancy_precision: float = 0.0
    occupancy_recall: float = 0.0
    target_iou: float = 0.0
    target_hit_rate: float = 0.0
    height_mae: float = 0.0
    density_mae: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def rasterize_ground_truth_occupancy(
    env: ParkingEnvironment,
    x: float,
    y: float,
    yaw: float,
    bev_config: BEVConfig,
    *,
    max_range: float | None = None,
) -> np.ndarray:
    """把场景 emits_points 障碍栅格化为车辆中心局部 BEV 真值 occupancy（几何上限）。

    只把阻挡 LiDAR 射线的障碍（emits_points=True）计入真值占用：悬崖禁入但
    不产生点云、地面标线可通行，二者都不应在 LiDAR 源 occupancy 中出现。
    地图边界本身不视为障碍（边界外的点不存在点云）。
    """
    del max_range  # 几何真值覆盖整个局部 BEV 范围，不随传感器量程裁剪
    config = BEVConfig(resolution=bev_config.resolution, extent=bev_config.extent)
    h, w = config.shape
    front, back, left, right = config.extent
    res = config.resolution
    truth = np.zeros((h, w), dtype=np.float32)

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    local_x = front - (rows + 0.5) * res
    local_y = -right + (cols + 0.5) * res
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    global_x = x + cos_yaw * grid_x - sin_yaw * grid_y
    global_y = y + sin_yaw * grid_x + cos_yaw * grid_y

    for obs in env.obstacles:
        if not obs.emits_points:
            continue
        x_min, x_max, y_min, y_max = obs.bbox
        in_box = (
            (global_x >= x_min)
            & (global_x <= x_max)
            & (global_y >= y_min)
            & (global_y <= y_max)
        )
        rr, cc = np.nonzero(in_box)
        for r, c in zip(rr.tolist(), cc.tolist()):
            if obs.contains_point(float(global_x[r, c]), float(global_y[r, c])):
                truth[r, c] = 1.0
    return truth


def rasterize_lidar_truth_occupancy(
    env: ParkingEnvironment,
    x: float,
    y: float,
    yaw: float,
    bev_config: BEVConfig,
    *,
    beams: int = 3600,
    max_range: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """用高分辨率无噪声 LiDAR 从同一位姿投射，得到传感器可感知占用真值。

    beams 远大于生产配置（360），使采样稀疏性引起的召回缺口接近下限，
    从而 occupancy 保真度主要反映"生产 LiDAR + BEV 管道"对可感知内容的还原，
    而不是高束数 LiDAR 本身的不确定性。
    """
    config = BEVConfig(resolution=bev_config.resolution, extent=bev_config.extent)
    lidar = SimulatedLiDAR(
        env,
        beams=beams,
        max_range=(
            max_range if max_range is not None else _bev_diagonal(config)
        ),
        z=1.0,
        noise="clean",
        seed=seed,
    )
    frame = lidar.capture(x, y, yaw)
    bev = LiDAR2BEV(config=config).to_bev(frame, x, y, yaw)
    return np.asarray(bev.data[bev.channels.index("occupancy")], dtype=np.float32)


def rasterize_ground_truth_target(
    goal: GoalPose,
    length: float,
    width: float,
    x: float,
    y: float,
    yaw: float,
    bev_config: BEVConfig,
) -> np.ndarray:
    """把目标车位矩形栅格化为车辆中心局部 BEV 真值 target 通道。

    目标区域以车位位姿为中心的 length×width 定向矩形（与 SimulatedCamera
    渲染使用的 parking_area 语义一致）。
    """
    config = BEVConfig(resolution=bev_config.resolution, extent=bev_config.extent)
    h, w = config.shape
    front, back, left, right = config.extent
    res = config.resolution
    truth = np.zeros((h, w), dtype=np.float32)

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    local_x = front - (rows + 0.5) * res
    local_y = -right + (cols + 0.5) * res
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    global_x = x + cos_yaw * grid_x - sin_yaw * grid_y
    global_y = y + sin_yaw * grid_x + cos_yaw * grid_y

    cg, sg = np.cos(goal.yaw), np.sin(goal.yaw)
    dx = global_x - goal.x
    dy = global_y - goal.y
    goal_x = cg * dx + sg * dy
    goal_y = -sg * dx + cg * dy
    inside = (np.abs(goal_x) <= length / 2.0) & (np.abs(goal_y) <= width / 2.0)
    truth[inside] = 1.0
    return truth


def _bev_diagonal(config: BEVConfig) -> float:
    front, back, left, right = config.extent
    return float(np.hypot(front + back, left + right))


def _segmentation_metrics(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """对二值栅格计算 IoU/Precision/Recall。"""
    pred = np.asarray(prediction, dtype=np.float32) > 0.5
    gt = np.asarray(truth, dtype=np.float32) > 0.5
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    return iou, precision, recall


def compute_bev_fidelity_metrics(
    bev: BEVTensor,
    truth_occupancy: np.ndarray,
    truth_target: np.ndarray,
) -> BEVFidelityMetrics:
    """按通道名计算 BEV 与真值的保真度指标。"""
    channels = list(bev.channels)
    if "occupancy" not in channels or "target" not in channels:
        raise ValueError("BEV 必须包含 occupancy 与 target 通道")
    data = np.asarray(bev.data, dtype=np.float32)
    occupancy_idx = channels.index("occupancy")
    target_idx = channels.index("target")

    occ_iou, occ_precision, occ_recall = _segmentation_metrics(
        data[occupancy_idx], truth_occupancy
    )
    tgt_iou, _, tgt_recall = _segmentation_metrics(
        data[target_idx], truth_target
    )

    metrics = BEVFidelityMetrics(
        samples=1,
        occupancy_iou=float(occ_iou),
        occupancy_precision=float(occ_precision),
        occupancy_recall=float(occ_recall),
        target_iou=float(tgt_iou),
        target_hit_rate=float(tgt_recall),
    )
    if "height" in channels:
        height_idx = channels.index("height")
        metrics = BEVFidelityMetrics(
            **{**metrics.to_dict(),
               "height_mae": float(np.abs(data[height_idx] - truth_occupancy).mean())}
        )
    if "density" in channels:
        density_idx = channels.index("density")
        metrics = BEVFidelityMetrics(
            **{**metrics.to_dict(),
               "density_mae": float(np.abs(data[density_idx] - truth_occupancy).mean())}
        )
    return metrics


def aggregate_bev_fidelity(
    metrics_list: Sequence[BEVFidelityMetrics],
) -> BEVFidelityMetrics:
    """聚合多帧指标为均值。"""
    if not metrics_list:
        raise ValueError("不能聚合空指标列表")
    fields = ["occupancy_iou", "occupancy_precision", "occupancy_recall",
              "target_iou", "target_hit_rate", "height_mae", "density_mae"]
    aggregated = {
        "samples": sum(metrics.samples for metrics in metrics_list),
    }
    for field in fields:
        values = [getattr(metrics, field) for metrics in metrics_list]
        aggregated[field] = float(np.mean(values))
    return BEVFidelityMetrics(**aggregated)


__all__ = [
    "BEVFidelityMetrics",
    "aggregate_bev_fidelity",
    "compute_bev_fidelity_metrics",
    "rasterize_ground_truth_occupancy",
    "rasterize_ground_truth_target",
    "rasterize_lidar_truth_occupancy",
]