"""模拟相机传感器。

通过相机模型把泊车位目标区域渲染到图像平面，生成 CameraFrame。

环视帧保留每个视角的图像与标定，Camera2BEV逐视角变换到统一车体系。
CameraFrame.image仅为兼容预览，不能用它恢复环视方向。
"""

from __future__ import annotations

import numpy as np

from interfaces import CameraFrame, CameraIntrinsics, CameraView, GoalPose
from .camera_model import CameraModel
from .environment import ParkingEnvironment
from .noise import NoiseLevel, NoiseProfile, get_noise_profile

# 默认环视视角（车体坐标系下相机朝向，0° 为车前，逆时针为正）。
DEFAULT_VIEW_YAWS_DEG = (0.0, 90.0, -90.0, 180.0)


class SimulatedCamera:
    """基于相机模型的模拟环视相机。

    intrinsics 为相机内参，image 尺寸为 (width, height)，height/pitch 为相机
    位姿参数（见 CameraModel）。parking_area 为 (length, width) 米，目标区域
    以此为边长绘制。view_yaws_deg 为各视角相机朝向（度）。
    """

    def __init__(
        self,
        env: ParkingEnvironment,
        intrinsics: CameraIntrinsics,
        height: float = 1.5,
        pitch: float = np.deg2rad(30.0),
        parking_area: tuple[float, float] = (6.0, 3.0),
        *,
        noise: NoiseLevel | str | NoiseProfile = NoiseLevel.CLEAN,
        seed: int = 0,
        view_yaws_deg: tuple[float, ...] = DEFAULT_VIEW_YAWS_DEG,
    ) -> None:
        self.env = env
        self.intrinsics = intrinsics
        self.height = float(height)
        self.pitch = float(pitch)
        self.view_yaws_deg = tuple(float(value) for value in view_yaws_deg)
        if not self.view_yaws_deg:
            raise ValueError("环视相机至少需要一个视角")
        if (not np.isfinite([self.height, self.pitch, *self.view_yaws_deg]).all()
                or self.height <= 0):
            raise ValueError("相机高度必须为正，视角与外参必须有限")
        self.model = CameraModel(intrinsics, height=height, pitch=pitch)
        self.parking_area = parking_area
        self.noise_profile = get_noise_profile(noise)
        self.rng = np.random.default_rng(seed)

    def capture(self, x: float, y: float, yaw: float) -> CameraFrame:
        """采集各视角；目标漏检事件共享，像素/伪目标噪声逐视角生成。"""
        config = self.noise_profile.camera
        missed = (
            bool(self.rng.random() < config.false_negative_rate)
            if self.env.parking_spots and config.false_negative_rate > 0.0 else False
        )
        views = []
        for view_yaw in self.view_yaws_deg:
            image = np.zeros((self.intrinsics.image_height, self.intrinsics.image_width), dtype=np.uint8)
            if self.env.parking_spots and not missed:
                rect = self._parking_rectangle(self.env.parking_spots[0])
                local = [self._to_view_local(px, py, x, y, yaw, view_yaw) for px, py in rect]
                visible = self._clip_to_camera_front(local)
                pixels = [self.model.project(*point) for point in visible]
                pixels = [point for point in pixels if point is not None]
                if len(pixels) >= 3:
                    self._fill_polygon(image, self._convex_hull(pixels), 255)
            if config.false_positive_rate > 0.0 and self.rng.random() < config.false_positive_rate:
                self._add_false_positive(image)
            if config.pixel_std > 0.0:
                image = np.clip(image.astype(np.float32)
                                + self.rng.normal(0.0, config.pixel_std, size=image.shape),
                                0.0, 255.0).astype(np.uint8)
            views.append(CameraView(image[:, :, None], self.intrinsics,
                                    yaw=float(np.deg2rad(view_yaw)), height=self.height, pitch=self.pitch))
        preview = np.maximum.reduce([view.image for view in views])
        return CameraFrame(image=preview, intrinsics=self.intrinsics, views=tuple(views))

    def _add_false_positive(self, image: np.ndarray) -> None:
        """在图像内增加一个随机矩形伪目标。"""
        h, w = image.shape
        patch_w = int(self.rng.integers(max(2, w // 30), max(3, w // 12)))
        patch_h = int(self.rng.integers(max(2, h // 30), max(3, h // 12)))
        x0 = int(self.rng.integers(0, max(1, w - patch_w + 1)))
        y0 = int(self.rng.integers(0, max(1, h - patch_h + 1)))
        image[y0 : y0 + patch_h, x0 : x0 + patch_w] = 255

    def _parking_rectangle(self, goal: GoalPose) -> list[tuple[float, float]]:
        """计算目标位姿对应的全局矩形四角（沿 yaw 方向）。"""
        length, width = self.parking_area
        cos_yaw, sin_yaw = np.cos(goal.yaw), np.sin(goal.yaw)
        fx = np.array([cos_yaw, sin_yaw])
        fy = np.array([-sin_yaw, cos_yaw])
        cx, cy = goal.x, goal.y
        corners = [
            np.array([cx, cy]) + (length / 2.0) * fx + (width / 2.0) * fy,
            np.array([cx, cy]) + (length / 2.0) * fx - (width / 2.0) * fy,
            np.array([cx, cy]) - (length / 2.0) * fx - (width / 2.0) * fy,
            np.array([cx, cy]) - (length / 2.0) * fx + (width / 2.0) * fy,
        ]
        return [(float(c[0]), float(c[1])) for c in corners]

    def _to_local(self, px: float, py: float, x: float, y: float, yaw: float) -> np.ndarray:
        """全局坐标变换到车辆中心局部系（X 前向、Y 左向）。"""
        dx, dy = px - x, py - y
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        return np.array(
            [cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy]
        )

    def _to_view_local(
        self, px: float, py: float, x: float, y: float, yaw: float, view_yaw_deg: float
    ) -> np.ndarray:
        """全局坐标变换到某个视角相机的局部系（X 为该相机前向）。"""
        base = self._to_local(px, py, x, y, yaw)
        theta = np.deg2rad(view_yaw_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        return np.array(
            [
                cos_t * base[0] + sin_t * base[1],
                -sin_t * base[0] + cos_t * base[1],
            ]
        )

    def _clip_to_camera_front(
        self, polygon: list[np.ndarray], *, eps: float = 1e-3
    ) -> list[tuple[float, float]]:
        """按相机前方条件 ``cosP·X + h·sinP > 0`` 对地面多边形做 Sutherland–Hodgman 裁剪。

        相机斜向下看时，车辆后方约 0.87m 以外的地面点落在相机平面之后。此处对
        这些边求与相机平面的交点并插值到同一正深度eps平面，避免透视除法遇到零深度。
        """
        cos_p, sin_p = np.cos(self.pitch), np.sin(self.pitch)
        offset = self.height * sin_p

        def depth(point: np.ndarray) -> float:
            return cos_p * float(point[0]) + offset

        output: list[np.ndarray] = []
        count = len(polygon)
        for index in range(count):
            current = polygon[index]
            following = polygon[(index + 1) % count]
            d_current, d_following = depth(current), depth(following)
            current_in = d_current > eps
            following_in = d_following > eps
            if current_in:
                output.append(current)
            if current_in != following_in:
                span = d_current - d_following
                if abs(span) > 1e-12:
                    ratio = (d_current - eps) / span
                    output.append(current + ratio * (following - current))
        return [(float(point[0]), float(point[1])) for point in output]

    def _convex_hull(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Andrew 单调链凸包；点数不足 3 时原样返回。"""
        unique = sorted({(float(px), float(py)) for px, py in points})
        if len(unique) < 3:
            return unique

        def cross(o, a, b) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple[float, float]] = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[float, float]] = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    def _fill_polygon(
        self, image: np.ndarray, vertices: list[tuple[float, float]], value: int
    ) -> None:
        """用扫描线法填充凸多边形，交点裁剪到图像范围内。"""
        h, w = image.shape
        ys = [v[1] for v in vertices]
        xs = [v[0] for v in vertices]
        y_min, y_max = max(0, int(np.floor(min(ys)))), min(h - 1, int(np.ceil(max(ys))))
        for row in range(y_min, y_max + 1):
            intersections = []
            for i in range(len(vertices)):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % len(vertices)]
                if (y1 <= row < y2) or (y2 <= row < y1):
                    t = (row - y1) / (y2 - y1)
                    intersections.append(x1 + t * (x2 - x1))
            if len(intersections) < 2:
                continue
            intersections.sort()
            x_start = max(0, int(np.floor(intersections[0])))
            x_end = min(w - 1, int(np.ceil(intersections[-1])))
            if x_start > x_end:
                continue
            image[row, x_start : x_end + 1] = value
