"""BEV 输入侧缺陷诊断（②）：量化 target 通道缺失的成因分布。

区分两类成因，避免把"相机视野不够"误诊成"投影实现有 bug"：

- **视野外**：目标在车体系下的方位角超出相机水平半视场，物理上看不见；
- **视野内但仍为空**：目标在视野内，却因为 `Camera2BEV` 的截断策略
  （任一角点投影出图即整帧置零）而丢失，属实现缺陷。

同时统计 height/density 通道的信息量，作为"网络输入表征不足"的量化证据。

运行：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/diagnose_bev_inputs.py \
        --data data/task_dataset/tracked_pivot_v7_3000/val.npz \
        --output runs/validation/bev-input-diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator
from interfaces import CameraIntrinsics
from sim.camera_model import CameraModel
from training.reporting import atomic_write_json


def _camera() -> CameraModel:
    """与 `dataset.components.build_task_components` 完全一致的相机参数。"""
    return CameraModel(
        CameraIntrinsics(
            fx=400.0, fy=400.0, cx=320.0, cy=240.0, image_width=640, image_height=480
        ),
        height=1.5,
        pitch=np.deg2rad(30.0),
    )


def _views_from_metadata(data: dict) -> tuple[float, ...] | None:
    """从 task_meta.dataset.camera 读取记录下来的相机视角配置。"""
    for meta in data.get("task_meta") or []:
        camera = ((meta or {}).get("dataset") or {}).get("camera")
        if isinstance(camera, dict) and camera.get("view_yaws_deg"):
            return tuple(float(value) for value in camera["view_yaws_deg"])
    return None


def _local_goal(goal: np.ndarray, state: np.ndarray) -> tuple[float, float, float]:
    dx, dy = goal[0] - state[0], goal[1] - state[1]
    cos_yaw, sin_yaw = np.cos(state[2]), np.sin(state[2])
    return (
        float(cos_yaw * dx + sin_yaw * dy),
        float(-sin_yaw * dx + cos_yaw * dy),
        float(np.arctan2(np.sin(goal[2] - state[2]), np.cos(goal[2] - state[2]))),
    )


def diagnose(data_path: Path, view_yaws_deg: tuple[float, ...] | None = None) -> dict:
    """诊断 BEV 输入质量。

    ``view_yaws_deg`` 给出实际相机视角；缺省读数据集 ``task_meta.dataset.camera``，
    再缺省按单前视（0°）处理。**注意**：若数据是环视配置，"目标在视野外"这一项
    必须按各视角分别判断，用单视角半视场去判定会把环视误报成视野外。
    """
    data = DatasetGenerator.load(data_path)
    bevs = np.asarray(data["bevs"])
    channels = [str(name) for name in np.asarray(data["bev_meta"]["channels"])]
    states = np.asarray(data["states"])
    goals = np.asarray(data["goals"])
    camera = _camera()

    if view_yaws_deg is None:
        view_yaws_deg = _views_from_metadata(data) or (0.0,)

    target = bevs[:, channels.index("target")]
    non_zero = (target != 0).reshape(len(target), -1).sum(axis=1)
    height = bevs[:, channels.index("height")]
    density = bevs[:, channels.index("density")]

    half_fov = np.arctan(640.0 / 2.0 / 400.0)  # 单视角水平半视场（弧度）
    azimuths: list[float] = []
    visible_goal: list[bool] = []
    projected_corners_in: list[int] = []
    length, width = 6.0, 3.0
    for index in range(len(bevs)):
        gx, gy, gyaw = _local_goal(goals[index], states[index])
        azimuth = float(np.arctan2(gy, gx))
        azimuths.append(azimuth)
        # 环视下"能被看到"= 至少一个视角的目标在前方且在水平视场内。
        sees = False
        for view_yaw in view_yaws_deg:
            theta = np.deg2rad(view_yaw)
            view_x = float(np.cos(theta) * gx + np.sin(theta) * gy)
            view_y = float(-np.sin(theta) * gx + np.cos(theta) * gy)
            if view_x > 0.0 and abs(np.arctan2(view_y, view_x)) <= half_fov:
                sees = True
                break
        visible_goal.append(sees)
        cos_y, sin_y = np.cos(gyaw), np.sin(gyaw)
        inside = 0
        for sx, sy in ((length / 2, width / 2), (length / 2, -width / 2),
                       (-length / 2, -width / 2), (-length / 2, width / 2)):
            px = gx + cos_y * sx - sin_y * sy
            py = gy + sin_y * sx + cos_y * sy
            projected = camera.project(px, py)
            if projected is None:
                continue
            u, v = projected
            if 0.0 <= u < 640.0 and 0.0 <= v < 480.0:
                inside += 1
        projected_corners_in.append(inside)

    azimuths_arr = np.asarray(azimuths)
    visible_arr = np.asarray(visible_goal)
    empty_arr = non_zero == 0
    in_view_but_empty = int(np.sum(visible_arr & empty_arr))
    out_of_view = int(np.sum(~visible_arr))
    all_corners_fail = np.asarray(projected_corners_in) == 0

    report = {
        "schema_version": 1,
        "data": str(data_path),
        "samples": int(len(bevs)),
        "channels": channels,
        "camera": {
            "half_fov_deg": float(np.degrees(half_fov)),
            "full_fov_deg": float(np.degrees(2.0 * half_fov)),
            "height_m": camera.height,
            "pitch_deg": float(np.degrees(camera.pitch)),
        },
        "target_channel": {
            "empty_samples": int(empty_arr.sum()),
            "empty_rate": float(empty_arr.mean()),
            "non_zero_median": float(np.median(non_zero)),
            "goal_within_fov_samples": int(visible_arr.sum()),
            "goal_out_of_fov_samples": out_of_view,
            "goal_out_of_fov_rate": float((~visible_arr).mean()),
            "in_view_but_empty_samples": in_view_but_empty,
            "in_view_empty_rate_of_in_view": (
                float(in_view_but_empty / max(int(visible_arr.sum()), 1))
            ),
            "all_corners_out_of_image_samples": int(all_corners_fail.sum()),
            "explained_by_fov_only": int(np.sum(~visible_arr & empty_arr)),
            "unexplained_in_view_empty": in_view_but_empty,
        },
        "goal_azimuth_deg": {
            "median": float(np.degrees(np.median(azimuths_arr))),
            "p05": float(np.degrees(np.percentile(azimuths_arr, 5))),
            "p95": float(np.degrees(np.percentile(azimuths_arr, 95))),
            "abs_ge_90_rate": float((np.abs(azimuths_arr) >= np.pi / 2).mean()),
        },
        "channel_information": {
            "height_unique_values": sorted(float(v) for v in np.unique(height)),
            "density_unique_values": sorted(float(v) for v in np.unique(density)),
        },
        "camera_views_deg": [float(value) for value in view_yaws_deg],
        "horizontal_full_fov_per_view_deg": float(np.degrees(2.0 * half_fov)),
        "conclusion": (
            "target 缺失按视角配置判定：单前视（77.3°）下主因是视野不够；"
            "环视（多视角）下应看 empty_rate 与 in_view_but_empty_samples。"
            "height 通道无高度信息属输入表征缺陷"
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/task_dataset/tracked_pivot_v7_3000/val.npz"
    )
    parser.add_argument("--output", default="runs/validation/bev-input-diagnosis.json")
    args = parser.parse_args()
    report = diagnose(Path(args.data).resolve())
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, report)
    target = report["target_channel"]
    print(f"报告：{destination}")
    print(
        f"样本 {report['samples']}；target 全空 {target['empty_samples']} "
        f"({target['empty_rate']:.1%})；目标在视野外 {target['goal_out_of_fov_samples']} "
        f"({target['goal_out_of_fov_rate']:.1%})；"
        f"视野内仍为空 {target['in_view_but_empty_samples']}"
    )
    print(
        f"相机水平全视场 {report['camera']['full_fov_deg']:.1f}°；"
        f"height 唯一值 {report['channel_information']['height_unique_values']}；"
        f"density 唯一值个数 {len(report['channel_information']['density_unique_values'])}"
    )


if __name__ == "__main__":
    main()
