"""Hybrid A* 专家轨迹生成器。

从起始状态到目标泊车位姿规划差分驱动可行的轨迹，作为训练数据标签。
算法要点（参考 PythonRobotics 的成熟实现并适配差分驱动）：
- 运动基元：离散 omega 与前后方向，沿弧长插值积分运动学；
- 状态离散：x/y 栅格 + yaw 扇区，用于 closed/open 集合判重；
- 碰撞检测：沿路径采样车辆矩形四角，检查是否在自由空间；
- 启发式：欧氏距离下界（admissible），保证搜索趋近目标。
"""

from __future__ import annotations

import heapq

import numpy as np

from interfaces import GoalPose, Trajectory, VehicleState
from sim import ParkingEnvironment


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

            if cur_idx == goal_idx:
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
        dt = self.motion_resolution / abs(self.plan_v)
        for omega in self.omega_values:
            for direction in (1.0, -1.0):
                v = direction * self.plan_v
                xs: list[float] = []
                ys: list[float] = []
                yaws: list[float] = []
                x, y, yaw = node.x, node.y, node.yaw
                collision = False
                for _ in range(n_steps):
                    x += v * np.cos(yaw) * dt
                    y += v * np.sin(yaw) * dt
                    yaw = self._norm_angle(yaw + omega * dt)
                    if not self._pose_free(x, y, yaw) or not self._within_search_bounds(x, y):
                        collision = True
                        break
                    xs.append(float(x))
                    ys.append(float(y))
                    yaws.append(float(yaw))
                if collision:
                    continue
                child = _Node(x, y, yaw, node.g + arc_len, parent=node)
                child.xs, child.ys, child.yaws = xs, ys, yaws
                children.append(child)
        return children

    def _heuristic(self, node: _Node, goal: GoalPose) -> float:
        """欧氏距离下界，除以参考速度换算为成本。"""
        dist = float(np.hypot(goal.x - node.x, goal.y - node.y))
        return dist / abs(self.plan_v)

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
        for n in reversed(chain):
            xs.extend(n.xs)
            ys.extend(n.ys)
            yaws.extend(n.yaws)
        # 终点对齐到精确目标位姿。
        if not xs:
            xs = [n.x for n in chain]
            ys = [n.y for n in chain]
            yaws = [n.yaw for n in chain]
        else:
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
        dt = self.motion_resolution / abs(self.plan_v)
        return Trajectory(points=pts, dt=dt)

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
        """判断车辆矩形四角（及中心）是否全部位于自由空间。

        collision_margin > 0 时将矩形各向外膨胀该裕度（C-space 膨胀），
        使规划出的轨迹与障碍保持至少 margin 的净空，吸收跟踪误差。
        """
        half_l = self.vehicle_length / 2.0 + self.collision_margin
        half_w = self.vehicle_width / 2.0 + self.collision_margin
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        corners = [
            (x + half_l * cos_yaw - half_w * sin_yaw, y + half_l * sin_yaw + half_w * cos_yaw),
            (x + half_l * cos_yaw + half_w * sin_yaw, y + half_l * sin_yaw - half_w * cos_yaw),
            (x - half_l * cos_yaw - half_w * sin_yaw, y - half_l * sin_yaw + half_w * cos_yaw),
            (x - half_l * cos_yaw + half_w * sin_yaw, y - half_l * sin_yaw - half_w * cos_yaw),
        ]
        return all(self.env.is_free(cx, cy) for cx, cy in corners)

    @staticmethod
    def _norm_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi)。"""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi