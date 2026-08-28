"""矿区泊车场景库（REQUIREMENTS §3：S1–S9）。

每个场景 = 参数化构造函数 + 元数据（类型/难度旋钮/精度标准），注册进
SCENE_REGISTRY。SceneBundle 携带环境、车位组、起终点采样区与配置，
供任务层（sim/tasks.py）与实验 runner 消费。

场景几何规格见 docs/REQUIREMENTS.md §3；精度值即车位容差默认值。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from interfaces import BEVConfig, GoalPose
from sim.obstacles import (
    KIND_BERM,
    KIND_CLIFF,
    KIND_EQUIPMENT,
    KIND_ROCK,
    KIND_WALL,
    CircleObstacle,
    Obstacle,
    PolygonObstacle,
    RectangleObstacle,
)
from sim.spots import ParkingSpot, make_spot_row

__all__ = ["SceneBundle", "SCENE_REGISTRY", "build_scene", "register_scene"]


DUMP_GEOMETRY_PROFILE = "vehicle_relative_v1"
DUMP_STOP_CLEARANCE = 0.3
DUMP_BAY_CENTER_SPACING_WIDTHS = 3.0
TWO_WAY_HAUL_ROAD_WIDTHS = 3.5


def _validate_vehicle_scale(
    vehicle_length: float,
    vehicle_width: float,
    collision_margin: float,
) -> None:
    if vehicle_length <= 0.0 or vehicle_width <= 0.0:
        raise ValueError("车辆尺寸必须为正")
    if collision_margin < 0.0:
        raise ValueError("collision_margin 不能为负")


def _dump_goal_y(
    berm_y: float,
    vehicle_length: float,
    collision_margin: float,
) -> float:
    """车尾朝 +y 时的中心 y；安全膨胀后仍保留卸载停车余量。"""
    return berm_y - (
        vehicle_length / 2.0 + collision_margin + DUMP_STOP_CLEARANCE
    )


@dataclass
class SceneBundle:
    """场景实例包：环境 + 车位 + 元数据 + 任务采样区。

    spawn_zones 为任务层提供合法起点采样区 [(x_min,x_max,y_min,y_max)]；
    difficulty_knobs 记录本场景的难度旋钮取值（写入任务元数据）。
    """

    name: str
    env: "ParkingEnvironment"  # noqa: F821（避免循环导入，运行时为 sim.environment 类型）
    spots: list[ParkingSpot]
    spawn_zones: list[tuple[float, float, float, float]] = field(default_factory=list)
    difficulty_knobs: dict = field(default_factory=dict)
    description: str = ""
    title_en: str = ""
    bev_config: BEVConfig = field(default_factory=BEVConfig)

    def free_spots(self) -> list[ParkingSpot]:
        return [s for s in self.spots if not s.occupied]


def register_scene(name: str):
    """场景构造器注册装饰器。"""

    def _wrap(fn):
        SCENE_REGISTRY[name] = fn
        return fn

    return _wrap


SCENE_REGISTRY: dict[str, callable] = {}


def build_scene(name: str, **kwargs) -> SceneBundle:
    """按名称与参数构建场景实例。"""
    if name not in SCENE_REGISTRY:
        raise ValueError(f"未知场景 {name}，可选：{sorted(SCENE_REGISTRY)}")
    return SCENE_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# S1 驻地停车场：2 排垂直车位 + 通道
# ---------------------------------------------------------------------------

@register_scene("S1_parking_lot")
def s1_parking_lot(
    spots_per_row: int = 6,
    aisle_width: float = 12.0,
    occupied_pattern: list[int] | None = None,
    seed: int = 0,
) -> SceneBundle:
    """驻地停车场：两排垂直车位相对，中间通道。

    车位 7×3.5m、车 6×3m；通道宽 9/12m 两档（难度旋钮）；occupied_pattern
    指定被占用车位索引（相邻占用为难度旋钮）。
    """
    rng = np.random.default_rng(seed)
    vehicle_l, vehicle_w = 6.0, 3.0
    pitch = 3.5
    row_len = spots_per_row * pitch
    half = max(25.0, row_len / 2 + 8.0)
    depth = 9.0
    world = 2 * (half + 6.0)

    def _occupied(idx: int, row: int) -> bool:
        if not occupied_pattern:
            return False
        return (row * spots_per_row + idx) in occupied_pattern

    spots: list[ParkingSpot] = []
    obstacles: list[Obstacle] = []
    # 北排（车位朝 +y，车头朝北）与南排（朝 -y）。
    row_y_n = depth / 2 + aisle_width / 2
    row_y_s = -(depth / 2 + aisle_width / 2)
    for idx in range(spots_per_row):
        px = -row_len / 2 + pitch / 2 + idx * pitch
        spot_n = ParkingSpot(
            id=f"N{idx}", pose=GoalPose(px, row_y_n, np.pi / 2),
            tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
            kind="perpendicular_bay",
            occupied=_occupied(idx, 0),
        )
        spot_s = ParkingSpot(
            id=f"S{idx}", pose=GoalPose(px, row_y_s, -np.pi / 2),
            tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
            kind="perpendicular_bay",
            occupied=_occupied(idx, 1),
        )
        spots.extend([spot_n, spot_s])
    for s in spots:
        if s.occupied:
            obstacles.append(s.occupant_obstacle(vehicle_l, vehicle_w))

    # 停车场边界墙（四周围合，留北侧道路开口供进入——简化为全围合）。
    wall_t = 0.5
    y_top = row_y_n + depth / 2
    y_bot = row_y_s - depth / 2
    obstacles += [
        RectangleObstacle(-half - wall_t, half + wall_t, y_top, y_top + wall_t, kind=KIND_WALL),
        RectangleObstacle(-half - wall_t, half + wall_t, y_bot - wall_t, y_bot, kind=KIND_WALL),
        RectangleObstacle(-half - wall_t, -half, y_bot - wall_t, y_top + wall_t, kind=KIND_WALL),
        RectangleObstacle(half, half + wall_t, y_bot - wall_t, y_top + wall_t, kind=KIND_WALL),
    ]

    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=world, obstacles=obstacles)
    return SceneBundle(
        name="S1_parking_lot",
        env=env,
        spots=spots,
        spawn_zones=[(-half + 2, half - 2, row_y_s + 1.5, row_y_n - 1.5)],
        difficulty_knobs={"spots_per_row": spots_per_row, "aisle_width": aisle_width,
                          "occupied": len(occupied_pattern or [])},
        description="驻地停车场：两排垂直车位，通道宽度与占用可调",
        title_en="Campus parking lot: two rows of perpendicular spots"
    )


# ---------------------------------------------------------------------------
# S2 斜列停车场：45° 斜位
# ---------------------------------------------------------------------------

@register_scene("S2_diagonal_lot")
def s2_diagonal_lot(
    spots_count: int = 6,
    occupied_pattern: list[int] | None = None,
    seed: int = 0,
) -> SceneBundle:
    """斜列停车场：单排 45° 斜位 + 单向通道。"""
    pitch = 4.5
    angle = np.pi / 4.0
    row_len = spots_count * pitch
    half = max(20.0, row_len / 2 + 6.0)
    world = 2 * (half + 5.0)
    spots: list[ParkingSpot] = []
    obstacles: list[Obstacle] = []
    for idx in range(spots_count):
        px = -row_len / 2 + pitch / 2 + idx * pitch
        spot = ParkingSpot(
            id=f"D{idx}", pose=GoalPose(px, 3.0, angle),
            tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
            kind="diagonal_bay",
            occupied=(idx in (occupied_pattern or [])),
        )
        spots.append(spot)
    for s in spots:
        if s.occupied:
            obstacles.append(s.occupant_obstacle(6.0, 3.0))
    wall_t = 0.5
    obstacles += [
        RectangleObstacle(-half - wall_t, half + wall_t, 7.5, 8.0, kind=KIND_WALL),
        RectangleObstacle(-half - wall_t, -half, -8.0, 8.0, kind=KIND_WALL),
        RectangleObstacle(half, half + wall_t, -8.0, 8.0, kind=KIND_WALL),
    ]
    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=world, obstacles=obstacles)
    return SceneBundle(
        name="S2_diagonal_lot",
        env=env,
        spots=spots,
        spawn_zones=[(-half + 2, half - 2, -6.0, 1.0)],
        difficulty_knobs={"spots_count": spots_count, "occupied": len(occupied_pattern or [])},
        description="斜列停车场：45° 斜位，单向通道",
        title_en="Diagonal parking lot: 45-degree spots"
    )


# ---------------------------------------------------------------------------
# S3 维修保养区：三面墙 bay + 前方维修作业道
# ---------------------------------------------------------------------------

MAINT_GEOMETRY_PROFILE = "vehicle_relative_v1"
MAINT_SIDE_CLEARANCE = 0.6


@register_scene("S3_maintenance")
def s3_maintenance(
    bay_count: int = 3,
    seed: int = 0,
    vehicle_length: float = 6.0,
    vehicle_width: float = 3.0,
    collision_margin: float = 0.2,
) -> SceneBundle:
    """维修保养区：紧净空 bay（三面墙+入口立柱）+ 前方维修作业道。

    真实矿区语义：维修车间前的硬化作业道，卡车沿作业道对齐 bay 轴线后
    倒车入位（yaw -90° 车头朝外）。作业道深度按 3.5 倍车长保证
    T3(15–30m) 远距接近可落地；bay 单侧净空 0.6m（需求 0.5~0.8m）。
    """
    _validate_vehicle_scale(vehicle_length, vehicle_width, collision_margin)
    clearance = MAINT_SIDE_CLEARANCE
    bay_w = vehicle_width + 2 * clearance
    row_len = bay_count * bay_w
    wall_t = 0.4
    bay_depth = vehicle_length + 2.0
    goal_y = bay_depth - (vehicle_length / 2.0 + 1.0)
    apron_depth = max(18.0, 3.5 * vehicle_length)
    half_x = max(row_len / 2 + 8.0, 15.0)
    half_y = apron_depth + 2.0
    world = 2 * (max(half_x, half_y) + wall_t + 3.0)

    obstacles: list[Obstacle] = []
    spots: list[ParkingSpot] = []
    # bay 隔墙：bay_count 个车位需要 bay_count+1 道墙（沿 y 从 0 到 bay_depth）。
    for i in range(bay_count + 1):
        wx = -row_len / 2 + i * bay_w - wall_t / 2
        obstacles.append(RectangleObstacle(wx, wx + wall_t, 0.0, bay_depth, kind=KIND_WALL))
    # 后墙。
    obstacles.append(RectangleObstacle(-row_len / 2 - wall_t, row_len / 2 + wall_t, bay_depth, bay_depth + wall_t, kind=KIND_WALL))
    # 车位：倒车入位，车位朝向 -y（车头朝外）。
    for i in range(bay_count):
        cx = -row_len / 2 + (i + 0.5) * bay_w
        spots.append(
            ParkingSpot(
                id=f"M{i}", pose=GoalPose(cx, goal_y, -np.pi / 2),
                size=(vehicle_length + 1.0, bay_w),
                tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
                kind="maintenance_bay",
            )
        )
    # 立柱：bay 入口两角（小圆障碍，分隔墙线上）。
    for i in range(1, bay_count):
        px = -row_len / 2 + i * bay_w
        obstacles.append(CircleObstacle(x=px, y=-0.6, radius=0.3, kind=KIND_EQUIPMENT))
    # 作业道外围墙（南墙远离 bay，保证远距起点有旋转空间）。
    obstacles += [
        RectangleObstacle(-half_x - wall_t, half_x + wall_t, -apron_depth - wall_t, -apron_depth, kind=KIND_WALL),
        RectangleObstacle(-half_x - wall_t, -half_x, -apron_depth, bay_depth + wall_t, kind=KIND_WALL),
        RectangleObstacle(half_x, half_x + wall_t, -apron_depth, bay_depth + wall_t, kind=KIND_WALL),
    ]
    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=world, obstacles=obstacles)
    return SceneBundle(
        name="S3_maintenance",
        env=env,
        spots=spots,
        spawn_zones=[(-half_x + 4.0, half_x - 4.0, -apron_depth + 4.0, -3.0)],
        difficulty_knobs={
            "bay_count": bay_count,
            "clearance": clearance,
            "geometry_profile": MAINT_GEOMETRY_PROFILE,
            "vehicle_length": vehicle_length,
            "vehicle_width": vehicle_width,
            "collision_margin": collision_margin,
        },
        description="维修保养区：紧净空倒车入位 + 前方维修作业道，入口立柱",
        title_en="Maintenance bays: tight reversing off a service apron",
    )


# ---------------------------------------------------------------------------
# S4 排土场卸载区：倒车贴挡墙，挡墙外悬崖
# ---------------------------------------------------------------------------

@register_scene("S4_dump_area")
def s4_dump_area(
    bay_count: int = 4,
    occupied_pattern: list[int] | None = None,
    seed: int = 0,
    vehicle_length: float = 6.0,
    vehicle_width: float = 3.0,
    collision_margin: float = 0.2,
) -> SceneBundle:
    """排土场卸载区：沿卸载边缘的挡墙条带，墙外悬崖禁区。

    任务语义：倒车至挡墙前停稳；航向指向车头，因此 -90° 表示车尾朝
    +y 挡墙。目标同时保留规划碰撞裕量和 0.3m 停车余量。
    """
    _validate_vehicle_scale(vehicle_length, vehicle_width, collision_margin)
    pitch = DUMP_BAY_CENTER_SPACING_WIDTHS * vehicle_width
    row_len = bay_count * pitch
    half = max(22.0, row_len / 2 + 8.0)
    world = 2 * (half + 8.0)
    world_half = world / 2.0
    wall_t = 0.6

    # 挡墙沿 y = 6 处（东西向）；悬崖在挡墙北侧（y > 6.6）。
    berm_y = 6.0
    obstacles: list[Obstacle] = [
        RectangleObstacle(-row_len / 2, row_len / 2, berm_y, berm_y + wall_t, kind=KIND_BERM),
        # 悬崖：挡墙外条带，禁入不挡射线。
        PolygonObstacle(
            vertices=[(-world_half, berm_y + wall_t), (world_half, berm_y + wall_t), (world_half, world_half), (-world_half, world_half)],
            kind=KIND_CLIFF, emits_points=False, forbidden=True,
        ),
    ]
    spots: list[ParkingSpot] = []
    for i in range(bay_count):
        cx = -row_len / 2 + pitch / 2 + i * pitch
        # 车尾朝墙（+y），车头朝 -y。
        cy = _dump_goal_y(berm_y, vehicle_length, collision_margin)
        spots.append(
            ParkingSpot(
                id=f"B{i}", pose=GoalPose(cx, cy, -np.pi / 2),
                size=(vehicle_length + 1.0, vehicle_width + 1.0),
                tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
                kind="berm_bay",
                occupied=(i in (occupied_pattern or [])),
            )
        )
    for s in spots:
        if s.occupied:
            obstacles.append(s.occupant_obstacle(vehicle_length, vehicle_width))
    # 场地侧墙。
    obstacles += [
        RectangleObstacle(-half - wall_t, -half + wall_t, -14.0, berm_y, kind=KIND_WALL),
        RectangleObstacle(half - wall_t, half + wall_t, -14.0, berm_y, kind=KIND_WALL),
        RectangleObstacle(-half - wall_t, half + wall_t, -14.0, -13.4, kind=KIND_WALL),
    ]
    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=world, obstacles=obstacles)
    return SceneBundle(
        name="S4_dump_area",
        env=env,
        spots=spots,
        spawn_zones=[(-row_len / 2, row_len / 2, -12.0, -4.0)],
        difficulty_knobs={
            "bay_count": bay_count,
            "occupied": len(occupied_pattern or []),
            "geometry_profile": DUMP_GEOMETRY_PROFILE,
            "vehicle_length": vehicle_length,
            "vehicle_width": vehicle_width,
            "collision_margin": collision_margin,
            "dump_stop_clearance": DUMP_STOP_CLEARANCE,
            "dump_bay_pitch": pitch,
        },
        description="排土场卸载区：倒车至挡墙前安全停稳，挡墙外悬崖禁入",
        title_en="Dump area: reverse parking against berm, cliff beyond"
    )


# ---------------------------------------------------------------------------
# S5 破碎站卸料口：倒车入窄槽居中
# ---------------------------------------------------------------------------

@register_scene("S5_crusher")
def s5_crusher(
    slot_count: int = 2,
    seed: int = 0,
) -> SceneBundle:
    """破碎站卸料口：两侧混凝土墙窄槽（间距 4.2m，单侧余量 0.6m）+ 入口挡柱。

    目标：倒车入槽居中停稳；槽底料口为禁区（禁入不挡射线）。
    """
    slot_w = 4.2
    slot_depth = 8.0
    pitch = slot_w + 0.8  # 槽间隔墙厚
    row_len = slot_count * pitch
    half = max(16.0, row_len / 2 + 6.0)
    world = 2 * (half + 5.0)
    wall_t = 0.4

    obstacles: list[Obstacle] = []
    spots: list[ParkingSpot] = []
    for i in range(slot_count):
        cx = -row_len / 2 + pitch / 2 + i * pitch
        # 槽侧墙（东西两道，沿 y∈[0, slot_depth]）。
        obstacles.append(RectangleObstacle(cx - slot_w / 2 - wall_t, cx - slot_w / 2, 0.0, slot_depth, kind=KIND_WALL))
        obstacles.append(RectangleObstacle(cx + slot_w / 2, cx + slot_w / 2 + wall_t, 0.0, slot_depth, kind=KIND_WALL))
        # 槽底料口（禁区、无点云）。
        obstacles.append(
            RectangleObstacle(cx - slot_w / 2, cx + slot_w / 2, slot_depth, slot_depth + 1.2,
                              kind=KIND_CLIFF, emits_points=False, forbidden=True)
        )
        # 入口挡柱。
        obstacles.append(CircleObstacle(x=cx - slot_w / 2 - 0.6, y=-0.5, radius=0.25, kind=KIND_EQUIPMENT))
        obstacles.append(CircleObstacle(x=cx + slot_w / 2 + 0.6, y=-0.5, radius=0.25, kind=KIND_EQUIPMENT))
        spots.append(
            ParkingSpot(
                id=f"C{i}", pose=GoalPose(cx, slot_depth - 3.0 - 0.5, -np.pi / 2),
                size=(6.5, 4.0), tol_pos=0.2, tol_yaw=np.deg2rad(5.0),
                kind="crusher_slot",
            )
        )
    # 外围。
    obstacles += [
        RectangleObstacle(-half - wall_t, half + wall_t, -10.0, -9.4, kind=KIND_WALL),
        RectangleObstacle(-half - wall_t, -half, -9.4, slot_depth + 1.2, kind=KIND_WALL),
        RectangleObstacle(half, half + wall_t, -9.4, slot_depth + 1.2, kind=KIND_WALL),
    ]
    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=world, obstacles=obstacles)
    return SceneBundle(
        name="S5_crusher",
        env=env,
        spots=spots,
        spawn_zones=[(-row_len / 2 - 1, row_len / 2 + 1, -8.0, -2.0)],
        difficulty_knobs={"slot_count": slot_count, "clearance": 0.6},
        description="破碎站卸料口：4.2m 窄槽倒车居中，槽底料口禁区",
        title_en="Crusher station: reverse into 4.2m slot, chute forbidden"
    )


# ---------------------------------------------------------------------------
# S6 装载工作面：前进对位（电铲 + 矿堆）
# ---------------------------------------------------------------------------

@register_scene("S6_loading_face")
def s6_loading_face(
    rock_count: int = 3,
    seed: int = 0,
) -> SceneBundle:
    """装载工作面：挖掘机/装载机多边形 + 台阶坡面 + 不规则矿堆；前进对位停稳。

    装载设备为挖掘机/装载机量级（约 6×3m，与 6×3 矿卡配套），
    现实参照 20~30t 级液压挖掘机或 ZL-50 装载机。
    """
    rng = np.random.default_rng(seed)
    obstacles: list[Obstacle] = []
    # 台阶坡面（北侧长墙）。
    obstacles.append(RectangleObstacle(-30.0, 30.0, 10.0, 13.0, kind=KIND_WALL))
    # 挖掘机：履带底盘 + 回转平台多边形 footprint（约 6m × 2.8m）。
    excavator = PolygonObstacle(
        vertices=[(2.5, 6.0), (7.5, 6.0), (8.0, 7.4), (7.2, 8.8), (2.8, 8.8), (2.0, 7.4)],
        kind=KIND_EQUIPMENT,
    )
    obstacles.append(excavator)
    # 装载位：电铲正前方，车头朝 +y（面向铲）。
    spots = [
        ParkingSpot(
            id="L0", pose=GoalPose(5.0, 2.0, np.pi / 2),
            size=(7.0, 4.0), tol_pos=0.5, tol_yaw=np.deg2rad(15.0),
            kind="loading_point",
        )
    ]
    # 不规则矿堆（随机凸多边形，放在装载区东西两侧空地，
    # 避开南侧起点采样区与 L0 前进对位走廊，保证 T3 远距接近不被堵死）。
    for i in range(rock_count):
        if rng.random() < 0.5:
            cx = rng.uniform(-30.0, -18.0)
        else:
            cx = rng.uniform(16.0, 30.0)
        cy = rng.uniform(-8.0, 6.0)
        r = rng.uniform(1.0, 2.2)
        n = 6
        angles = np.sort(rng.uniform(0.0, 2 * np.pi, n))
        verts = tuple(
            (float(cx + r * np.cos(a)), float(cy + r * np.sin(a))) for a in angles
        )
        obstacles.append(PolygonObstacle(vertices=verts, kind=KIND_ROCK))
    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=70.0, obstacles=obstacles)
    return SceneBundle(
        name="S6_loading_face",
        env=env,
        spots=spots,
        spawn_zones=[(-16.0, 14.0, -10.0, -2.0)],
        difficulty_knobs={"rock_count": rock_count},
        description="装载工作面：挖掘机/装载机 + 随机矿堆，前进对位（±0.5m/±15°）",
        title_en="Loading face: excavator alignment with random rock piles"
    )


# ---------------------------------------------------------------------------
# S7 加油/加水站：平行停靠
# ---------------------------------------------------------------------------

@register_scene("S7_fuel_station")
def s7_fuel_station(
    bay_count: int = 2,
    occupied_pattern: list[int] | None = None,
    seed: int = 0,
    vehicle_length: float = 6.0,
    vehicle_width: float = 3.0,
    collision_margin: float = 0.2,
) -> SceneBundle:
    """加油/加水站：中央加油岛，两侧平行停靠位，后侧挡墙。

    平行泊位：车位长 8m（车 6m + 前后余量 1m）、岛宽 2.4m；
    目标位姿平行于岛，与岛间距保持 0.3m 物理余量并计入碰撞裕量。
    """
    island_w = 2.4
    bay_len = 8.0
    pitch = 9.0
    row_len = bay_count * pitch
    half = max(18.0, row_len / 2 + 6.0)
    world = 2 * (half + 5.0)
    wall_t = 0.5

    obstacles: list[Obstacle] = [
        # 加油岛（东西向中央）。
        RectangleObstacle(-row_len / 2 - 1.0, row_len / 2 + 1.0, -island_w / 2, island_w / 2, kind=KIND_EQUIPMENT),
        # 后侧挡墙（北侧）。
        RectangleObstacle(-half - wall_t, half + wall_t, 6.0, 6.5, kind=KIND_WALL),
        RectangleObstacle(-half - wall_t, -half, -6.5, 6.5, kind=KIND_WALL),
        RectangleObstacle(half, half + wall_t, -6.5, 6.5, kind=KIND_WALL),
    ]
    spots: list[ParkingSpot] = []
    # 北侧平行位（岛与挡墙之间，车头朝 +x；靠岛侧计入碰撞裕量，保证实际净空 ≥0.3m）。
    for i in range(bay_count):
        cx = -row_len / 2 + pitch / 2 + i * pitch
        cy = island_w / 2 + 0.3 + collision_margin + vehicle_width / 2.0
        spots.append(
            ParkingSpot(
                id=f"F{i}", pose=GoalPose(cx, cy, 0.0),
                size=(8.0, 3.5), tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
                kind="fuel_bay",
                occupied=(i in (occupied_pattern or [])),
            )
        )
    for s in spots:
        if s.occupied:
            obstacles.append(s.occupant_obstacle(vehicle_length, vehicle_width))
    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=world, obstacles=obstacles)
    return SceneBundle(
        name="S7_fuel_station",
        env=env,
        spots=spots,
        spawn_zones=[(-half + 2, half - 2, -5.0, -2.0)],
        difficulty_knobs={
            "bay_count": bay_count,
            "occupied": len(occupied_pattern or []),
            "vehicle_length": vehicle_length,
            "vehicle_width": vehicle_width,
            "collision_margin": collision_margin,
        },
        description="加油/加水站：中央加油岛两侧平行停靠，后侧挡墙",
        title_en="Fuel station: parallel parking beside central island"
    )


# ---------------------------------------------------------------------------
# S8 称重站：前进精准停线
# ---------------------------------------------------------------------------

@register_scene("S8_weigh_station")
def s8_weigh_station(seed: int = 0) -> SceneBundle:
    """称重站：车道两侧导向墙 + 称重台标线；前进停在台心（±0.2m）。

    称重台沿行驶方向 7m，完整覆盖 6m 车身（停稳时前后轴均在台上）。
    """
    wall_t = 0.5
    lane_half = 3.0  # 车道半宽（车宽 3m + 余量）
    pad_x_min, pad_x_max = 2.0, 9.0  # 台心 5.5，两侧各留 0.5m 余量
    obstacles: list[Obstacle] = [
        RectangleObstacle(-25.0, 25.0, lane_half, lane_half + wall_t, kind=KIND_WALL),
        RectangleObstacle(-25.0, 25.0, -lane_half - wall_t, -lane_half, kind=KIND_WALL),
        # 称重台（地面标线：可通行、不挡射线、非碰撞）。
        RectangleObstacle(pad_x_min, pad_x_max, -lane_half, lane_half, kind="line", emits_points=False, forbidden=False),
    ]
    spots = [
        ParkingSpot(
            id="W0", pose=GoalPose(5.5, 0.0, 0.0),
            size=(7.0, 3.5), tol_pos=0.2, tol_yaw=np.deg2rad(10.0),
            kind="weigh_pad",
        )
    ]
    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=55.0, obstacles=obstacles)
    return SceneBundle(
        name="S8_weigh_station",
        env=env,
        spots=spots,
        spawn_zones=[(-20.0, -6.0, -1.5, 1.5)],
        difficulty_knobs={},
        description="称重站：导向墙车道内前进精准停线（±0.2m）",
        title_en="Weigh station: precise stop on weigh pad (+-0.2m)"
    )


# ---------------------------------------------------------------------------
# S9 综合矿场地图：道路串联停车场/卸载区/破碎站
# ---------------------------------------------------------------------------

@register_scene("S9_mine_complex")
def s9_mine_complex(
    rock_count: int = 6,
    occupied_pattern: list[int] | None = None,
    seed: int = 0,
    vehicle_length: float = 6.0,
    vehicle_width: float = 3.0,
    collision_margin: float = 0.2,
) -> SceneBundle:
    """综合矿场：100×60m，运矿道路（12m 宽、弯道）串联 S1 停车场、
    S4 卸载区、S5 破碎站三功能区，路侧散布岩石。"""
    _validate_vehicle_scale(vehicle_length, vehicle_width, collision_margin)
    rng = np.random.default_rng(seed)
    obstacles: list[Obstacle] = []
    spots: list[ParkingSpot] = []

    # 布局：道路沿 L 形（西边南北向 → 北部东西向），三区挂靠。
    # 道路挡墙：西道（x∈[-50,-38]，y∈[-30,10]）与北道（x∈[-50,50]，y∈[8,20]）。
    road_w = max(12.0, TWO_WAY_HAUL_ROAD_WIDTHS * vehicle_width)
    west_road_inner_x = -50.0 + road_w
    north_road_top_y = 8.0 + road_w
    obstacles += [
        RectangleObstacle(-52.0, -50.0, -30.0, 10.0, kind=KIND_WALL),   # 西道西墙
        RectangleObstacle(west_road_inner_x - 0.5, west_road_inner_x, -30.0, 10.0, kind=KIND_WALL),  # 西道东墙
        RectangleObstacle(-50.0, 50.0, north_road_top_y, north_road_top_y + 2.0, kind=KIND_WALL),     # 北道北墙
        RectangleObstacle(-50.0, 50.0, 8.0 - 0.5, 8.0, kind=KIND_WALL),  # 北道南墙（东段）
    ]
    # 弯道开口（去掉西北角内角墙的重叠区已由矩形布置自然形成）。

    # S1 停车场（东南角）：单排垂直车位，朝 +y（车头朝北），面向南部通道。
    lot_x0, lot_y = 18.0, -18.0
    pitch = max(3.5, vehicle_width + 0.5)
    n_lot = 5
    for i in range(n_lot):
        px = lot_x0 + i * pitch
        occ = i in (occupied_pattern or [])
        spot = ParkingSpot(
            # 车位南侧有围挡、入口在北侧；基准航向 -90° 表示前进入位。
            # 倒车任务由 TaskSampler 将目标航向翻转 180°，使车头朝入口。
            id=f"P{i}", pose=GoalPose(px, lot_y, -np.pi / 2),
            tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
            kind="perpendicular_bay", occupied=occ,
        )
        spots.append(spot)
        if occ:
            obstacles.append(spot.occupant_obstacle(vehicle_length, vehicle_width))
    # 停车场围挡。
    obstacles += [
        RectangleObstacle(lot_x0 - 1.5, lot_x0 + n_lot * pitch + 1.5, lot_y - 4.5, lot_y - 4.0, kind=KIND_WALL),
        RectangleObstacle(lot_x0 - 2.0, lot_x0 - 1.5, lot_y - 4.0, lot_y + 4.0, kind=KIND_WALL),
        RectangleObstacle(lot_x0 + n_lot * pitch + 1.5, lot_x0 + n_lot * pitch + 2.0, lot_y - 4.0, lot_y + 4.0, kind=KIND_WALL),
    ]
    # 停车场首末车位中心向内收半格（车宽 3m < pitch 3.5m，靠墙侧留余量）。
    spots_lot = [s for s in spots if s.id.startswith("P")]
    for i, s in enumerate(spots_lot):
        shift = 0.25 if i == 0 else (-0.25 if i == len(spots_lot) - 1 else 0.0)
        s.pose = GoalPose(s.pose.x + shift, s.pose.y, s.pose.yaw)

    # S4 卸载区（北部东端）：挡墙沿 y=2（北道南侧），车尾朝北贴墙。
    berm_y = 2.0
    berm_pitch = DUMP_BAY_CENTER_SPACING_WIDTHS * vehicle_width
    n_berm = 3
    bx0 = 22.0
    berm_x_min = bx0 - vehicle_width / 2.0
    berm_x_max = bx0 + (n_berm - 1) * berm_pitch + vehicle_width / 2.0
    obstacles.append(RectangleObstacle(berm_x_min, berm_x_max, berm_y, berm_y + 0.6, kind=KIND_BERM))
    obstacles.append(
        PolygonObstacle(
            vertices=[(berm_x_min, berm_y + 0.6), (berm_x_max, berm_y + 0.6),
                      (berm_x_max, 8.0), (berm_x_min, 8.0)],
            kind=KIND_CLIFF, emits_points=False, forbidden=True,
        )
    )
    for i in range(n_berm):
        cx = bx0 + i * berm_pitch
        spots.append(
            ParkingSpot(
                id=f"DB{i}",
                pose=GoalPose(
                    cx,
                    _dump_goal_y(berm_y, vehicle_length, collision_margin),
                    -np.pi / 2,
                ),
                size=(vehicle_length + 1.0, vehicle_width + 1.0),
                tol_pos=0.3, tol_yaw=np.deg2rad(10.0),
                kind="berm_bay",
            )
        )

    # S5 破碎站（西部南端）：单个窄槽。
    slot_w = 4.2
    sx, sy0 = -30.0, -20.0
    obstacles += [
        RectangleObstacle(sx - slot_w / 2 - 0.4, sx - slot_w / 2, sy0, sy0 + 8.0, kind=KIND_WALL),
        RectangleObstacle(sx + slot_w / 2, sx + slot_w / 2 + 0.4, sy0, sy0 + 8.0, kind=KIND_WALL),
        RectangleObstacle(sx - slot_w / 2, sx + slot_w / 2, sy0 + 8.0, sy0 + 9.2,
                          kind=KIND_CLIFF, emits_points=False, forbidden=True),
    ]
    spots.append(
        ParkingSpot(
            id="CS0", pose=GoalPose(sx, sy0 + 5.0 - 0.5, -np.pi / 2),
            size=(6.5, 4.0), tol_pos=0.2, tol_yaw=np.deg2rad(5.0),
            kind="crusher_slot",
        )
    )

    # 路侧岩石（随机圆障碍，避开功能区）。
    placed = 0
    while placed < rock_count:
        cx = rng.uniform(-46.0, 46.0)
        cy = rng.uniform(-28.0, 6.0)
        r = rng.uniform(0.8, 1.8)
        # 避开道路（西道 x∈[-50,-38]，北道 y∈[8,20]）与三功能区包围盒。
        if -50 <= cx <= -37 and -30 <= cy <= 10:
            continue
        if 15 <= cx <= 42 and -24 <= cy <= -13:
            continue
        if 20 <= cx <= 44 and -4 <= cy <= 10:
            continue
        if -34 <= cx <= -26 and -22 <= cy <= -11:
            continue
        obstacles.append(CircleObstacle(x=float(cx), y=float(cy), radius=float(r), kind=KIND_ROCK))
        placed += 1

    from sim.environment import ParkingEnvironment

    env = ParkingEnvironment(world_size=110.0, obstacles=obstacles)
    return SceneBundle(
        name="S9_mine_complex",
        env=env,
        spots=spots,
        spawn_zones=[(-48.0, -40.0, -26.0, 6.0), (-40.0, 40.0, -26.0, 6.0)],
        difficulty_knobs={
            "rock_count": rock_count,
            "occupied": len(occupied_pattern or []),
            "geometry_profile": DUMP_GEOMETRY_PROFILE,
            "vehicle_length": vehicle_length,
            "vehicle_width": vehicle_width,
            "collision_margin": collision_margin,
            "dump_stop_clearance": DUMP_STOP_CLEARANCE,
            "dump_bay_pitch": berm_pitch,
            "haul_road_width": road_w,
        },
        description="综合矿场：道路串联停车场/卸载区/破碎站，路侧岩石，跨区远距任务",
        title_en="Mine complex: road-linked lot, dump area and crusher",
        bev_config=BEVConfig(resolution=0.5, extent=(40.0, 40.0, 40.0, 40.0)),
    )
