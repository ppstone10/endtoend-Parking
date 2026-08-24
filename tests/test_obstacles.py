"""障碍物体系测试：形状几何、解析 raycast 与语义属性。"""

import unittest

import numpy as np

from sim import (
    KIND_CLIFF,
    KIND_LINE,
    CircleObstacle,
    ParkingEnvironment,
    PolygonObstacle,
    RectangleObstacle,
)


def _stepping_raycast(env: ParkingEnvironment, origin: np.ndarray, angle: float, max_range: float, step: float = 0.05) -> float:
    """旧步进式 raycast 参考实现（重构前 environment.raycast 原文）。"""
    distance = 0.0
    direction = np.array([np.cos(angle), np.sin(angle)])
    while distance < max_range:
        distance += step
        probe = origin + direction * distance
        if not env.is_free(float(probe[0]), float(probe[1])):
            return distance
    return max_range


class TestRectangleGeometry(unittest.TestCase):
    def setUp(self):
        self.rect = RectangleObstacle(x_min=2.0, x_max=5.0, y_min=-1.0, y_max=1.0)

    def test_contains(self):
        self.assertTrue(self.rect.contains_point(3.0, 0.0))
        self.assertTrue(self.rect.contains_point(2.0, -1.0))  # 边界含入
        self.assertFalse(self.rect.contains_point(5.5, 0.0))
        self.assertFalse(self.rect.contains_point(3.0, 2.0))

    def test_ray_entry_axis(self):
        # 从原点沿 +x 射线，应在 x=2 处进入。
        self.assertAlmostEqual(self.rect.ray_entry_distance(0.0, 0.0, 1.0, 0.0), 2.0)

    def test_ray_entry_diagonal(self):
        # 浅斜射线（斜率 0.25）从 (0,0)：在 x=2 进入且 y=0.5 在矩形内。
        angle = np.arctan(0.25)
        dx, dy = np.cos(angle), np.sin(angle)
        t = self.rect.ray_entry_distance(0.0, 0.0, dx, dy)
        self.assertAlmostEqual(t, 2.0 / dx, places=6)

    def test_ray_steep_diagonal_misses(self):
        # 45° 对角线 y=x：穿过 y=1 时 x=1 < 2，不与矩形相交。
        d = np.sqrt(0.5)
        self.assertIsNone(self.rect.ray_entry_distance(0.0, 0.0, d, d))

    def test_ray_miss(self):
        self.assertIsNone(self.rect.ray_entry_distance(0.0, 5.0, 1.0, 0.0))

    def test_ray_behind(self):
        # 障碍在射线反方向。
        self.assertIsNone(self.rect.ray_entry_distance(0.0, 0.0, -1.0, 0.0))

    def test_origin_inside_returns_zero(self):
        self.assertEqual(self.rect.ray_entry_distance(3.0, 0.0, 1.0, 0.0), 0.0)

    def test_legacy_signature_compat(self):
        """四字段构造保持兼容，语义字段取默认。"""
        r = RectangleObstacle(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
        self.assertEqual(r.kind, "wall")
        self.assertTrue(r.emits_points)
        self.assertTrue(r.forbidden)


class TestCircleGeometry(unittest.TestCase):
    def setUp(self):
        self.circle = CircleObstacle(x=4.0, y=0.0, radius=1.0)

    def test_contains(self):
        self.assertTrue(self.circle.contains_point(4.0, 0.5))
        self.assertTrue(self.circle.contains_point(5.0, 0.0))  # 边界
        self.assertFalse(self.circle.contains_point(6.0, 0.0))

    def test_ray_entry(self):
        self.assertAlmostEqual(self.circle.ray_entry_distance(0.0, 0.0, 1.0, 0.0), 3.0)

    def test_ray_miss(self):
        self.assertIsNone(self.circle.ray_entry_distance(0.0, 5.0, 1.0, 0.0))

    def test_origin_inside(self):
        self.assertEqual(self.circle.ray_entry_distance(4.5, 0.0, 1.0, 0.0), 0.0)

    def test_negative_radius_rejected(self):
        with self.assertRaises(ValueError):
            CircleObstacle(x=0.0, y=0.0, radius=-1.0)


class TestPolygonGeometry(unittest.TestCase):
    def setUp(self):
        # 凹多边形（带一个朝内的缺口），验证非凸正确性。
        # 顶点顺序：(2,-2) → (6,-2) → (6,2) → (4,2) → (4,0) → (2,0) 闭合。
        # 形状：右半为 6×4 矩形，左半仅 y∈[-2,0] 条带；y∈(0,2)、x∈[2,4) 为缺口。
        self.poly = PolygonObstacle(
            vertices=[(2.0, -2.0), (6.0, -2.0), (6.0, 2.0), (4.0, 2.0), (4.0, 0.0), (2.0, 0.0)]
        )

    def test_contains(self):
        self.assertTrue(self.poly.contains_point(3.0, -1.0))  # 左下条带内部
        self.assertFalse(self.poly.contains_point(3.0, 1.0))  # 缺口内
        self.assertTrue(self.poly.contains_point(5.0, 1.0))  # 右半内部
        self.assertFalse(self.poly.contains_point(0.0, 0.0))

    def test_ray_entry_lower_band(self):
        # 从原点沿 +x 在 y=-1（条带内）：x=2 进入。
        self.assertAlmostEqual(self.poly.ray_entry_distance(0.0, -1.0, 1.0, 0.0), 2.0)

    def test_ray_through_notch(self):
        # 沿 y=0.5 穿过缺口：与缺口右边界（边 (4,2)-(4,0)）相交于 x=4。
        t = self.poly.ray_entry_distance(0.0, 0.5, 1.0, 0.0)
        self.assertAlmostEqual(t, 4.0)

    def test_ray_miss(self):
        self.assertIsNone(self.poly.ray_entry_distance(0.0, 5.0, 1.0, 0.0))

    def test_min_vertices(self):
        with self.assertRaises(ValueError):
            PolygonObstacle(vertices=[(0.0, 0.0), (1.0, 1.0)])


class TestRaycastRegression(unittest.TestCase):
    """解析 raycast 与旧步进实现的一致性回归。"""

    def _env(self):
        return ParkingEnvironment(
            world_size=40.0,
            obstacles=[
                RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=-6.0, y_max=-2.0),
                RectangleObstacle(x_min=-15.0, x_max=15.0, y_min=2.0, y_max=6.0),
                CircleObstacle(x=8.0, y=0.0, radius=1.5),
                PolygonObstacle(vertices=[(-8.0, 8.0), (-4.0, 8.0), (-6.0, 12.0)]),
            ],
        )

    def test_matches_stepping_reference(self):
        env = self._env()
        rng = np.random.default_rng(42)
        max_diff = 0.0
        n_checked = 0
        for _ in range(60):
            ox, oy = rng.uniform(-18.0, 18.0, size=2)
            for angle in rng.uniform(0.0, 2.0 * np.pi, size=8):
                # 起点在障碍内时旧实现返回 0.05、新实现 0.0，属已知差异，跳过。
                if not env.is_free(ox, oy):
                    continue
                old = _stepping_raycast(env, np.array([ox, oy]), float(angle), 20.0)
                new = env.raycast(np.array([ox, oy]), float(angle), 20.0)
                max_diff = max(max_diff, abs(old - new))
                n_checked += 1
        self.assertGreater(n_checked, 300)
        # 步进量化误差上界为 step=0.05（首个非自由采样点）。
        self.assertLessEqual(max_diff, 0.05 + 1e-9)

    def test_boundary_exit_is_hit(self):
        env = ParkingEnvironment(world_size=40.0, obstacles=[])
        # 朝 +x 从原点：20m 处仍在图内（半宽 20），恰为 max_range → 无命中。
        self.assertEqual(env.raycast(np.array([0.0, 0.0]), 0.0, 20.0), 20.0)
        # 从 x=5 朝 +x：15m 后出界。
        self.assertAlmostEqual(env.raycast(np.array([5.0, 0.0]), 0.0, 30.0), 15.0, places=6)


class TestCliffSemantics(unittest.TestCase):
    """悬崖：禁止进入（碰撞）但不挡 LiDAR 射线。"""

    def _env(self):
        cliff = PolygonObstacle(
            vertices=[(10.0, -4.0), (18.0, -4.0), (18.0, 4.0), (10.0, 4.0)],
            kind=KIND_CLIFF,
            emits_points=False,
            forbidden=True,
        )
        wall = RectangleObstacle(x_min=10.0, x_max=10.5, y_min=-4.0, y_max=4.0)  # 悬崖近侧的挡墙
        return ParkingEnvironment(world_size=40.0, obstacles=[cliff, wall])

    def test_cliff_forbidden(self):
        env = self._env()
        self.assertFalse(env.is_free(12.0, 0.0))
        self.assertFalse(env.is_free(10.2, 0.0))  # 与墙重叠区仍禁止

    def test_ray_hits_wall_not_cliff(self):
        env = self._env()
        # 从原点朝 +x：墙（emits_points）在 x=10 挡住射线。
        dist = env.raycast(np.array([0.0, 0.0]), 0.0, 30.0)
        self.assertAlmostEqual(dist, 10.0, places=6)

    def test_cliff_alone_does_not_block(self):
        cliff = PolygonObstacle(
            vertices=[(10.0, -4.0), (18.0, -4.0), (18.0, 4.0), (10.0, 4.0)],
            kind=KIND_CLIFF,
            emits_points=False,
            forbidden=True,
        )
        env = ParkingEnvironment(world_size=40.0, obstacles=[cliff])
        # 只有悬崖时：射线穿过崖区直到出图边界（x=20）。
        dist = env.raycast(np.array([0.0, 0.0]), 0.0, 30.0)
        self.assertAlmostEqual(dist, 20.0, places=6)

    def test_line_marking_traversable(self):
        line = RectangleObstacle(
            x_min=5.0, x_max=6.0, y_min=-3.0, y_max=3.0,
            kind=KIND_LINE, emits_points=False, forbidden=False,
        )
        env = ParkingEnvironment(world_size=40.0, obstacles=[line])
        self.assertTrue(env.is_free(5.5, 0.0))  # 标线可通行
        self.assertEqual(env.raycast(np.array([0.0, 0.0]), 0.0, 10.0), 10.0)  # 不挡射线


if __name__ == "__main__":
    unittest.main()
