"""Hybrid A* 专家轨迹生成器。

从起始状态到目标泊车位姿规划履带钻机低速运动学可行轨迹。
算法要点（参考成熟 Hybrid A*/State Lattice 实现并适配履带底盘）：
- 运动基元：离散 omega 前后行驶，并支持绕两履带几何中心原地旋转；
- 状态离散：x/y 栅格 + yaw 扇区，用于 closed/open 集合判重；
- 碰撞检测：沿路径采样车辆矩形四角，检查是否在自由空间；
- 启发式：欧氏距离下界（admissible），保证搜索趋近目标。
"""

from __future__ import annotations

import heapq

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from sim import ParkingEnvironment

from .collision import RectangleFootprintCollisionChecker
from .reeds_shepp import reeds_shepp_paths


class _Node:
    """搜索节点：记录连续位姿、累计成本、父节点与插值路径。"""

    __slots__ = ("x", "y", "yaw", "g", "parent", "xs", "ys", "yaws")

    def __init__(
        self,
        x: float,
        y: float,
        yaw: float,
        g: float,
        parent: "_Node | None" = None,
    ) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.g = g
        self.parent = parent
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.yaws: list[float] = []


class HybridAStarPlanner:
    """Hybrid A* 规划器。

    env 为目标环境；vehicle_length/width 为车辆矩形尺寸用于碰撞检测；
    xy_resolution 为栅格分辨率，yaw_resolution 为 yaw 离散扇区；
    plan_v 为规划参考速度，max_omega 为最大角速度，omega_steps 为离散档数。
    """

    def __init__(
        self,
        env: ParkingEnvironment,
        vehicle_length: float = 4.0,
        vehicle_width: float = 2.0,
        xy_resolution: float = 0.5,
        yaw_resolution: float = np.deg2rad(30.0),
        motion_resolution: float = 0.1,
        plan_v: float = 0.5,
        max_omega: float = 0.8,
        omega_steps: int = 5,
        search_margin: float = 4.0,
        max_expansions: int = 20000,
        collision_margin: float = 0.0,
        analytic_expansion_distance: float | None = None,
        analytic_expansion_interval: int = 5,
        min_turning_radius: float | None = None,
        enable_pivot: bool = False,
        pivot_omega: float | None = None,
        rotation_penalty: float = 5.0,
        collision_check_resolution: float = 0.1,
        vehicle_model_name: str = "tracked_kinematic",
        vehicle_model_version: str = "tracked_pivot_v1",
        vehicle_model_metadata: dict | None = None,
    ) -> None:
        self.env = env
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.collision_margin = collision_margin
        self.xy_resolution = xy_resolution
        self.yaw_resolution = yaw_resolution
        self.motion_resolution = motion_resolution
        self.plan_v = plan_v
        self.max_omega = max_omega
        if plan_v == 0.0 or max_omega == 0.0:
            raise ValueError("plan_v 与 max_omega 不能为 0")
        if analytic_expansion_distance is not None and analytic_expansion_distance < 0.0:
            raise ValueError("analytic_expansion_distance 不能为负数")
        if analytic_expansion_interval < 1:
            raise ValueError("analytic_expansion_interval 至少为 1")
        if min_turning_radius is not None and min_turning_radius <= 0.0:
            raise ValueError("min_turning_radius 必须为正数")
        if collision_check_resolution <= 0.0:
            raise ValueError("collision_check_resolution 必须为正数")
        if rotation_penalty <= 0.0:
            raise ValueError("rotation_penalty 必须为正数")
        resolved_pivot_omega = max_omega if pivot_omega is None else float(pivot_omega)
        if resolved_pivot_omega <= 0.0 or resolved_pivot_omega > abs(max_omega):
            raise ValueError("pivot_omega 必须为正且不超过 max_omega")
        # 矿卡紧凑入位需在约两个车长的范围内尝试解析机动；
        # 固定 6m 门限会使 6m 车身在 S3/S5 先陷入离散搜索。
        self.analytic_expansion_distance = (
            max(6.0, 2.0 * vehicle_length)
            if analytic_expansion_distance is None
            else float(analytic_expansion_distance)
        )
        self.analytic_expansion_interval = analytic_expansion_interval
        self.min_turning_radius = (
            float(min_turning_radius)
            if min_turning_radius is not None
            else abs(plan_v) / abs(max_omega)
        )
        self.enable_pivot = bool(enable_pivot)
        self.pivot_omega = resolved_pivot_omega
        self.rotation_penalty = float(rotation_penalty)
        self.collision_check_resolution = float(collision_check_resolution)
        self.trajectory_dt = self.motion_resolution / abs(self.plan_v)
        self.vehicle_model_name = str(vehicle_model_name)
        self.vehicle_model_version = str(vehicle_model_version)
        self.vehicle_model_metadata = dict(vehicle_model_metadata or {})
        self._collision_checker = RectangleFootprintCollisionChecker(
            env,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            collision_margin=collision_margin,
            resolution=collision_check_resolution,
        )
        self.num_yaw = int(round(2.0 * np.pi / yaw_resolution))
        self.omega_values = np.linspace(-max_omega, max_omega, omega_steps)
        # 搜索范围限制在起点-目标包围盒外加 search_margin，避免远距离发散。
        self.search_margin = search_margin
        # 探索节点上限，防止难解位姿组合导致发散；超限视为规划失败。
        self.max_expansions = max_expansions

        half = env.world_size / 2.0
        self.min_x = -half
        self.max_x = half
        self.min_y = -half
        self.max_y = half

    # ------------------------------------------------------------------
    # 规划入口
    # ------------------------------------------------------------------

    def plan(self, start: VehicleState, goal: GoalPose) -> Trajectory:
        """从起始状态规划到目标位姿，返回全局坐标轨迹点 (N, 3)。"""
        if not self._pose_free(start.x, start.y, start.yaw):
            raise ValueError("起始位姿与障碍物冲突")
        if not self._pose_free(goal.x, goal.y, goal.yaw):
            raise ValueError("目标位姿与障碍物冲突")

        start_node = _Node(start.x, start.y, start.yaw, 0.0)
        goal_idx = self._pose_index(goal.x, goal.y, goal.yaw)
        self._search_bounds = self._compute_search_bounds(start, goal)

        open_heap: list[tuple[float, int, _Node]] = []
        counter = 0
        heapq.heappush(open_heap, (0.0, counter, start_node))
        counter += 1
        closed: dict[tuple[int, int, int], float] = {}
        expansions = 0

        while open_heap:
            if expansions >= self.max_expansions:
                raise RuntimeError("Hybrid A* 探索节点数超出上限，规划失败")
            expansions += 1
            _, _, current = heapq.heappop(open_heap)
            cur_idx = self._pose_index(current.x, current.y, current.yaw)
            if cur_idx in closed and closed[cur_idx] < current.g:
                continue
            closed[cur_idx] = current.g

            analytic_due = (
                expansions == 1
                or expansions % self.analytic_expansion_interval == 0
                or cur_idx == goal_idx
            )
            if (
                analytic_due
                and self._heuristic_distance(current, goal) <= self.analytic_expansion_distance
            ):
                analytic_node = self._analytic_connection(current, goal)
                if analytic_node is not None:
                    return self._extract_trajectory(analytic_node, goal)

            if self.analytic_expansion_distance == 0.0 and cur_idx == goal_idx:
                return self._extract_trajectory(current, goal)

            for node in self._expand(current):
                n_idx = self._pose_index(node.x, node.y, node.yaw)
                if n_idx in closed and closed[n_idx] <= node.g:
                    continue
                heuristic = self._heuristic(node, goal)
                heapq.heappush(open_heap, (node.g + heuristic, counter, node))
                counter += 1

        raise RuntimeError("Hybrid A* 未能找到可行轨迹")

    # ------------------------------------------------------------------
    # 搜索辅助
    # ------------------------------------------------------------------

    def _compute_search_bounds(self, start: VehicleState, goal: GoalPose) -> tuple[float, float, float, float]:
        """计算搜索范围 (x_min, x_max, y_min, y_max)，为起点-目标包围盒加边距。"""
        cx = (start.x + goal.x) / 2.0
        cy = (start.y + goal.y) / 2.0
        span = max(
            abs(goal.x - start.x) / 2.0 + self.search_margin,
            abs(goal.y - start.y) / 2.0 + self.search_margin,
            self.search_margin,
        )
        return (cx - span, cx + span, cy - span, cy + span)

    def _within_search_bounds(self, x: float, y: float) -> bool:
        """判断点位是否落在搜索范围内。"""
        x_min, x_max, y_min, y_max = self._search_bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _expand(self, node: _Node) -> list[_Node]:
        """按运动基元扩展邻居节点，过滤碰撞与越界。"""
        children: list[_Node] = []
        arc_len = self.xy_resolution * 1.5
        n_steps = max(1, int(np.ceil(arc_len / self.motion_resolution)))
        dt = self.trajectory_dt
        for omega in self.omega_values:
            for direction in (1.0, -1.0):
                v = direction * self.plan_v
                xs: list[float] = []
                ys: list[float] = []
                yaws: list[float] = []
                x, y, yaw = node.x, node.y, node.yaw
                collision = False
                for _ in range(n_steps):
                    previous = (x, y, yaw)
                    x += v * np.cos(yaw) * dt
                    y += v * np.sin(yaw) * dt
                    yaw = self._norm_angle(yaw + omega * dt)
                    if (
                        not self._within_search_bounds(x, y)
                        or not self._swept_segment_free(previous, (x, y, yaw))
                    ):
                        collision = True
                        break
                    xs.append(float(x))
                    ys.append(float(y))
                    yaws.append(float(yaw))
                if collision:
                    continue
                duration = n_steps * dt
                child = _Node(x, y, yaw, node.g + duration, parent=node)
                child.xs, child.ys, child.yaws = xs, ys, yaws
                children.append(child)
        if self.enable_pivot:
            for direction in (-1.0, 1.0):
                delta_yaw = direction * self.yaw_resolution
                points = self._sample_pivot(
                    (node.x, node.y, node.yaw), delta_yaw
                )
                if not self._path_free(points):
                    continue
                child = _Node(
                    float(points[-1, 0]),
                    float(points[-1, 1]),
                    float(points[-1, 2]),
                    node.g
                    + self.rotation_penalty
                    * abs(delta_yaw)
                    / self.pivot_omega,
                    parent=node,
                )
                child.xs = points[1:, 0].astype(float).tolist()
                child.ys = points[1:, 1].astype(float).tolist()
                child.yaws = points[1:, 2].astype(float).tolist()
                children.append(child)
        return children

    def _heuristic(self, node: _Node, goal: GoalPose) -> float:
        """欧氏距离下界，除以参考速度换算为成本。"""
        dist = self._heuristic_distance(node, goal)
        return dist / abs(self.plan_v)

    @staticmethod
    def _heuristic_distance(node: _Node, goal: GoalPose) -> float:
        return float(np.hypot(goal.x - node.x, goal.y - node.y))

    def _analytic_connection(self, node: _Node, goal: GoalPose) -> _Node | None:
        """比较履带直接候选与 48 词族候选，返回最低成本可行连接。"""
        start_pose = (node.x, node.y, node.yaw)
        goal_pose = (goal.x, goal.y, goal.yaw)
        feasible: list[tuple[float, np.ndarray]] = []
        if self.enable_pivot:
            for cost, points in self._tracked_direct_candidates(start_pose, goal_pose):
                if self._path_free(points):
                    feasible.append((cost, points))
            if self._heuristic_distance(node, goal) <= 1e-9 and feasible:
                feasible.sort(key=lambda item: item[0])
                return self._terminal_node(node, goal, *feasible[0])
        for candidate in reeds_shepp_paths(start_pose, goal_pose, self.min_turning_radius):
            points, _directions = candidate.sample(start_pose, self.motion_resolution)
            # 公式误差已在 reeds_shepp_paths 中限定；此处压到调用方的精确目标值，
            # 避免 _extract_trajectory 为 1e-15 量级残差再添加一个伪短段。
            points[-1] = goal_pose
            if not self._path_free(points):
                continue
            feasible.append((candidate.total_length / abs(self.plan_v), points))
        if not feasible:
            return None
        feasible.sort(key=lambda item: item[0])
        return self._terminal_node(node, goal, *feasible[0])

    def _terminal_node(
        self, node: _Node, goal: GoalPose, cost: float, points: np.ndarray
    ) -> _Node:
        terminal = _Node(goal.x, goal.y, goal.yaw, node.g + cost, parent=node)
        terminal.xs = points[1:, 0].astype(float).tolist()
        terminal.ys = points[1:, 1].astype(float).tolist()
        terminal.yaws = points[1:, 2].astype(float).tolist()
        return terminal

    def _tracked_direct_candidates(
        self,
        start_pose: tuple[float, float, float],
        goal_pose: tuple[float, float, float],
    ) -> list[tuple[float, np.ndarray]]:
        """生成前/倒车的“旋转—直行—旋转”履带解析候选。"""
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose
        dx, dy = gx - sx, gy - sy
        distance = float(np.hypot(dx, dy))
        headings = [syaw] if distance <= 1e-9 else [
            float(np.arctan2(dy, dx)),
            self._norm_angle(float(np.arctan2(dy, dx)) + np.pi),
        ]
        candidates: list[tuple[float, np.ndarray]] = []
        for heading in headings:
            first_delta = self._angle_delta(heading, syaw)
            last_delta = self._angle_delta(gyaw, heading)
            pieces = [self._sample_pivot(start_pose, first_delta)]
            if distance > 1e-9:
                pieces.append(
                    self._sample_straight(
                        tuple(pieces[-1][-1]), (gx, gy, heading)
                    )
                )
            pieces.append(
                self._sample_pivot(tuple(pieces[-1][-1]), last_delta)
            )
            points = pieces[0]
            for piece in pieces[1:]:
                points = np.vstack((points, piece[1:]))
            points[-1] = goal_pose
            cost = distance / abs(self.plan_v) + self.rotation_penalty * (
                abs(first_delta) + abs(last_delta)
            ) / self.pivot_omega
            candidates.append((cost, points))
        return candidates

    def _sample_pivot(
        self, start_pose: tuple[float, float, float], delta_yaw: float
    ) -> np.ndarray:
        """以固定几何中心采样原地旋转，限制角速度与最远角点扫掠步长。"""
        x, y, yaw = (float(value) for value in start_pose)
        corner_radius = self._collision_checker.corner_radius
        time_steps = abs(delta_yaw) / (self.pivot_omega * self.trajectory_dt)
        sweep_steps = corner_radius * abs(delta_yaw) / self.collision_check_resolution
        steps = max(1, int(np.ceil(max(time_steps, sweep_steps))))
        fractions = np.linspace(0.0, 1.0, steps + 1)
        points = np.empty((steps + 1, 3), dtype=np.float64)
        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = [self._norm_angle(yaw + fraction * delta_yaw) for fraction in fractions]
        return points

    def _sample_straight(
        self,
        start_pose: tuple[float, float, float],
        goal_pose: tuple[float, float, float],
    ) -> np.ndarray:
        sx, sy, syaw = (float(value) for value in start_pose)
        gx, gy, _ = (float(value) for value in goal_pose)
        distance = float(np.hypot(gx - sx, gy - sy))
        steps = max(1, int(np.ceil(distance / self.motion_resolution)))
        fractions = np.linspace(0.0, 1.0, steps + 1)
        points = np.empty((steps + 1, 3), dtype=np.float64)
        points[:, 0] = sx + fractions * (gx - sx)
        points[:, 1] = sy + fractions * (gy - sy)
        points[:, 2] = syaw
        return points

    def _splice_valid(self, x: float, y: float, yaw: float, goal: GoalPose) -> bool:
        """校验位姿到目标的直线拼接段（按 motion_resolution 加密）。"""
        dist = float(np.hypot(goal.x - x, goal.y - y))
        n_splice = max(1, int(np.ceil(dist / self.motion_resolution)))
        dyaw = float(np.arctan2(np.sin(goal.yaw - yaw), np.cos(goal.yaw - yaw)))
        for k in range(1, n_splice + 1):
            t = k / n_splice
            sx = x + t * (goal.x - x)
            sy = y + t * (goal.y - y)
            syaw = self._norm_angle(yaw + t * dyaw)
            if not self._pose_free(sx, sy, syaw):
                return False
        return True

    def _extract_trajectory(self, goal_node: _Node, goal: GoalPose) -> Trajectory:
        """回溯节点链合并插值点，并拼接目标点。"""
        xs: list[float] = []
        ys: list[float] = []
        yaws: list[float] = []
        chain: list[_Node] = []
        node: _Node | None = goal_node
        while node is not None:
            chain.append(node)
            node = node.parent
        ordered = list(reversed(chain))
        root = ordered[0]
        xs.append(root.x)
        ys.append(root.y)
        yaws.append(root.yaw)
        for n in ordered:
            xs.extend(n.xs)
            ys.extend(n.ys)
            yaws.extend(n.yaws)
        # 终点对齐到精确目标位姿。
        if not np.allclose(
            [xs[-1], ys[-1], yaws[-1]],
            [goal.x, goal.y, goal.yaw],
            atol=1e-9,
        ):
            # 拼接段（节点末位姿 → 精确目标）不经过运动基元的碰撞检查，
            # 须逐点校验，防止贴墙目标产出带碰撞的拼接线。
            if not self._splice_valid(xs[-1], ys[-1], yaws[-1], goal):
                raise RuntimeError("终点拼接段与障碍物冲突，规划失败")
            xs.append(goal.x)
            ys.append(goal.y)
            yaws.append(goal.yaw)
        # 去重相邻重复点。
        pts = np.stack([np.array(xs), np.array(ys), np.array(yaws)], axis=1)
        keep = np.ones(pts.shape[0], dtype=bool)
        keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
        pts = pts[keep]
        return Trajectory(points=pts, dt=self.trajectory_dt)

    # ------------------------------------------------------------------
    # 离散化与碰撞
    # ------------------------------------------------------------------

    def _pose_index(self, x: float, y: float, yaw: float) -> tuple[int, int, int]:
        """连续位姿映射到离散索引（x/y 栅格 + yaw 扇区）。"""
        x_idx = int(round((x - self.min_x) / self.xy_resolution))
        y_idx = int(round((y - self.min_y) / self.xy_resolution))
        yaw_idx = int(round(self._norm_angle(yaw) / self.yaw_resolution)) % self.num_yaw
        return x_idx, y_idx, yaw_idx

    def _pose_free(self, x: float, y: float, yaw: float) -> bool:
        """兼容既有调用，完整外廓判定由独立碰撞检查器拥有。"""
        return self._collision_checker.pose_free(x, y, yaw)

    def _path_free(self, points: np.ndarray) -> bool:
        """检查路径采样点、搜索边界及相邻位姿连续扫掠。"""
        for x, y, _yaw in points:
            if not self._within_search_bounds(float(x), float(y)):
                return False
        if not self._pose_free(*map(float, points[0])):
            return False
        for index in range(len(points) - 1):
            if not self._swept_segment_free(points[index], points[index + 1]):
                return False
        return True

    def _swept_segment_free(
        self,
        start_pose: tuple[float, float, float] | np.ndarray,
        end_pose: tuple[float, float, float] | np.ndarray,
    ) -> bool:
        """兼容既有调用，连续扫掠由独立碰撞检查器拥有。"""
        return self._collision_checker.swept_segment_free(
            np.asarray(start_pose, dtype=np.float64),
            np.asarray(end_pose, dtype=np.float64),
        )

    def model_metadata(self) -> dict:
        """返回生成数据用于版本门禁的车辆/规划模型元数据。"""
        if self.vehicle_model_metadata:
            return dict(self.vehicle_model_metadata)
        return {
            "name": self.vehicle_model_name,
            "model_version": self.vehicle_model_version,
            "length": self.vehicle_length,
            "width": self.vehicle_width,
            "plan_v": self.plan_v,
            "plan_max_omega": self.max_omega,
            "collision_margin": self.collision_margin,
            "xy_resolution": self.xy_resolution,
            "yaw_resolution_deg": float(np.degrees(self.yaw_resolution)),
            "motion_resolution": self.motion_resolution,
            "collision_check_resolution": self.collision_check_resolution,
            "enable_pivot": self.enable_pivot,
            "pivot_omega": self.pivot_omega,
            "rotation_penalty": self.rotation_penalty,
        }

    @staticmethod
    def _norm_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi)。"""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _angle_delta(cls, target: float, source: float) -> float:
        return cls._norm_angle(target - source)
