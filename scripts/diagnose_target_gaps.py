"""诊断剩余 target 空帧：区分"目标在 BEV 覆盖范围外"与"渲染退化"。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/diagnose_target_gaps.py \
        --data runs/validation/rebuild-val/val-360.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dataset import DatasetGenerator
from sim.sensor_camera import DEFAULT_VIEW_YAWS_DEG


def diagnose(data_path: Path, views: tuple[float, ...]) -> dict:
    data = DatasetGenerator.load(data_path)
    channels = [str(c) for c in np.asarray(data["bev_meta"]["channels"])]
    bevs = np.asarray(data["bevs"])
    target = bevs[:, channels.index("target")]
    empty = ~np.any(target.reshape(len(target), -1), axis=1)
    states = np.asarray(data["states"])
    goals = np.asarray(data["goals"])
    extent = [float(v) for v in np.asarray(data["bev_meta"]["extent"])]
    front, back, left, right = extent

    records: list[dict] = []
    for index in np.flatnonzero(empty):
        dx, dy = goals[index][0] - states[index][0], goals[index][1] - states[index][1]
        cos_yaw, sin_yaw = np.cos(states[index][2]), np.sin(states[index][2])
        gx = float(cos_yaw * dx + sin_yaw * dy)
        gy = float(-sin_yaw * dx + cos_yaw * dy)
        azimuth = float(np.degrees(np.arctan2(gy, gx)))
        half_diagonal = 0.5 * float(np.hypot(6.0, 3.0))
        inside = (
            -back - half_diagonal <= gx <= front + half_diagonal
            and -right - half_diagonal <= gy <= left + half_diagonal
        )
        # 该目标在哪个视角的相机前方（用于判断是否被所有视角的相机平面裁掉）。
        front_views = []
        for view_yaw in views:
            theta = np.deg2rad(view_yaw)
            local_x = np.cos(theta) * gx + np.sin(theta) * gy
            if local_x > -0.87:  # 相机平面前方
                front_views.append(view_yaw)
        records.append(
            {
                "index": int(index),
                "task_id": str(data["task_meta"][index].get("task_id")),
                "scene": str(data["task_meta"][index].get("scene_name")),
                "distance_m": float(np.hypot(gx, gy)),
                "azimuth_deg": round(azimuth, 1),
                "inside_bev_extent": bool(inside),
                "views_with_goal_in_front": front_views,
            }
        )
    outside = sum(1 for item in records if not item["inside_bev_extent"])
    no_front_view = sum(1 for item in records if not item["views_with_goal_in_front"])
    return {
        "schema_version": 1,
        "data": str(data_path),
        "samples": int(len(bevs)),
        "empty_samples": len(records),
        "empty_rate": len(records) / len(bevs),
        "outside_bev_extent": outside,
        "no_view_sees_goal": no_front_view,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="runs/validation/rebuild-val/val-360.npz")
    parser.add_argument("--output", default="runs/validation/rebuild-val/target-gaps.json")
    args = parser.parse_args()
    report = diagnose(Path(args.data).resolve(), DEFAULT_VIEW_YAWS_DEG)
    import json

    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"样本 {report['samples']}；target 空帧 {report['empty_samples']} "
        f"（{report['empty_rate']:.1%}）；其中目标在 BEV 覆盖外 {report['outside_bev_extent']}，"
        f"无任何视角可见 {report['no_view_sees_goal']}"
    )
    for item in report["records"][:12]:
        print(
            f"  #{item['index']} {item['scene']} 距 {item['distance_m']:.2f}m "
            f"方位 {item['azimuth_deg']}° BEV内={item['inside_bev_extent']} "
            f"前方视角={item['views_with_goal_in_front']}"
        )
    print(f"报告：{destination}")


if __name__ == "__main__":
    main()
