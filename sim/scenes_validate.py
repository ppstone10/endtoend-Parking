"""场景库自检：注册即校验（几何合法性、车位无碰撞、可达性抽验）。

每个场景构造器在测试与实验加载前运行自检，尽早暴露构造错误：
- 车位目标位姿的车身矩形无碰撞（占用位除外，占用者本身即障碍）；
- 起点采样区内存在自由位姿；
- 至少一个空闲车位；
- 图幅合法（障碍全部在地图内）。
"""

from __future__ import annotations

import numpy as np

from sim.scenes import SceneBundle, build_scene
from sim.spots import ParkingSpot


def _vehicle_free(bundle: SceneBundle, x: float, y: float, yaw: float, length: float = 6.0, width: float = 3.0) -> bool:
    """车辆矩形四角（含中心）是否全部自由。"""
    half_l, half_w = length / 2.0, width / 2.0
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    pts = [(x, y)]
    for cl, cw in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)):
        pts.append((x + cl * cos_yaw - cw * sin_yaw, y + cl * sin_yaw + cw * cos_yaw))
    return all(bundle.env.is_free(px, py) for px, py in pts)


def validate_scene(bundle: SceneBundle, spot: ParkingSpot | None = None) -> list[str]:
    """返回错误列表（空列表 = 通过）。spot 指定时仅检查该车位。"""
    errors: list[str] = []
    env = bundle.env

    # 1. 障碍在图内（按包围盒检查，允许贴边）。
    half = env.world_size / 2.0
    for obs in env.obstacles:
        x_min, x_max, y_min, y_max = obs.bbox
        if x_min < -half - 1e-6 or x_max > half + 1e-6 or y_min < -half - 1e-6 or y_max > half + 1e-6:
            errors.append(f"障碍 {obs} 越出地图边界")

    # 2. 空闲车位目标位姿无碰撞。
    targets = [spot] if spot is not None else [s for s in bundle.spots if not s.occupied]
    for s in targets:
        if not _vehicle_free(bundle, s.pose.x, s.pose.y, s.pose.yaw):
            errors.append(f"车位 {s.id} 目标位姿碰撞")

    # 3. 至少一个空闲车位（指定单车位检查时跳过）。
    if spot is None and not bundle.free_spots():
        errors.append("场景无空闲车位")

    # 4. 起点采样区内存在自由位姿。
    if bundle.spawn_zones:
        rng = np.random.default_rng(0)
        found = False
        for _ in range(200):
            zone = bundle.spawn_zones[rng.integers(len(bundle.spawn_zones))]
            x = rng.uniform(zone[0], zone[1])
            y = rng.uniform(zone[2], zone[3])
            yaw = rng.uniform(-np.pi, np.pi)
            if _vehicle_free(bundle, x, y, yaw):
                found = True
                break
        if not found:
            errors.append("起点采样区 200 次采样未找到自由位姿")
    return errors


def validate_all_registered() -> dict[str, list[str]]:
    """构建并自检全部注册场景（默认参数），返回 场景名 → 错误列表。"""
    from sim.scenes import SCENE_REGISTRY

    report: dict[str, list[str]] = {}
    for name in sorted(SCENE_REGISTRY):
        bundle = build_scene(name)
        report[name] = validate_scene(bundle)
    return report
