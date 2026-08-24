"""模拟相机传感器。

通过相机模型将泊车位目标区域渲染到图像平面，生成 CameraFrame。
车位区域以全局坐标矩形表示，投影到图像后填充为高亮目标区域。
"""

from __future__ import annotations

import numpy as np

from interfaces import CameraFrame, CameraIntrinsics, GoalPose
from .camera_model import CameraModel
from .environment import ParkingEnvironment
from .noise import NoiseLevel, NoiseProfile, get_noise_profile


class SimulatedCamera:
    """基于相机模型的模拟相机。

    intrinsics 为相机内参，image 尺寸为 (width, height)，height/pitch 为相机
    位姿参数（见 CameraModel）。parking_area 为 (length, width) 米，目标区域
    以此为边长绘制在当前车辆前方视野内的图像中。
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
    ) -> None:
        self.env = env
        self.intrinsics = intrinsics
        self.model = CameraModel(intrinsics, height=height, pitch=pitch)
        self.parking_area = parking_area
        self.noise_profile = get_noise_profile(noise)
        self.rng = np.random.default_rng(seed)

    def capture(self, x: float, y: float, yaw: float) -> CameraFrame:
        """采集一帧图像。

        将环境中的第一个泊车位目标区域投影到图像并填充为白色（255），
        其余像素为黑色。图像为灰度单通道。
        """
        w = self.intrinsics.image_width
        h = self.intrinsics.image_height
        image = np.zeros((h, w), dtype=np.uint8)

        config = self.noise_profile.camera
        missed = (
            bool(self.rng.random() < config.false_negative_rate)
            if self.env.parking_spots and config.false_negative_rate > 0.0 else False
        )
        if self.env.parking_spots and not missed:
            goal = self.env.parking_spots[0]
            rect = self._parking_rectangle(goal)
            pixels = []
            for px, py in rect:
                local = self._to_local(px, py, x, y, yaw)
                proj = self.model.project(float(local[0]), float(local[1]))
                if proj is None:
                    pixels = []
                    break
                pixels.append(proj)
            if pixels:
                self._fill_polygon(image, pixels, 255)

        if config.false_positive_rate > 0.0 and self.rng.random() < config.false_positive_rate:
            self._add_false_positive(image)
        if config.pixel_std > 0.0:
            image = np.clip(
                image.astype(np.float32)
                + self.rng.normal(0.0, config.pixel_std, size=image.shape),
                0.0,
                255.0,
            ).astype(np.uint8)

        return CameraFrame(image=image[:, :, None], intrinsics=self.intrinsics)

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
