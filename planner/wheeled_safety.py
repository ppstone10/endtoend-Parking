"""对同一轮式执行弧线作车型余量下的完整矩形转移检查。"""

from __future__ import annotations

import math

import numpy as np

from interfaces.wheeled import WheeledControlCmd, WheeledVehicleState
from .collision import RectangleFootprintCollisionChecker


class WheeledTransitionChecker:
    def __init__(self, model, env, *, dt=.1, resolution=.1):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (dt, resolution)):
            raise ValueError("碰撞检查dt和resolution必须有限且为正")
        self.model, self.env, self.dt, self.resolution = model, env, float(dt), float(resolution)
        self._base = self._checker(0.)

    def _checker(self, extra_margin):
        c = self.model.config
        return RectangleFootprintCollisionChecker(self.env, vehicle_length=c.length, vehicle_width=c.width,
                                                  collision_margin=c.collision_margin+extra_margin,
                                                  resolution=self.resolution)

    def pose_free(self, state):
        return self._base.pose_free(state.x, state.y, state.yaw)

    def __call__(self, before, after):
        if not isinstance(before, WheeledVehicleState) or not isinstance(after, WheeledVehicleState):
            raise TypeError("轮式转移检查要求完整转角状态")
        # after的实际v/delta作为请求重放，能直接检查变化率与ZOH弧线是否一致。
        expected = self.model.step(before, WheeledControlCmd(after.v, after.steering_angle), self.dt)
        if not np.allclose(expected.to_array(), after.to_array(), rtol=0, atol=1e-9):
            return False
        turn = after.yaw-before.yaw
        if abs(turn) > math.pi:
            return False
        curvature = math.tan(after.steering_angle)/self.model.config.wheelbase
        extra = 0.
        if turn != 0 and curvature != 0:
            radius = math.hypot(1/curvature, self.model.config.center_from_rear)
            extra = 2*radius*math.sin(turn/4)**2
        # 原扫掠器在离散姿态间采样，额外半分辨率覆盖任意两采样之间的车体移动。
        moved = before.x != after.x or before.y != after.y or turn != 0
        checker = self._checker(extra+self.resolution/2) if moved else self._base
        return checker.swept_segment_free(np.asarray([before.x, before.y, before.yaw]),
                                          np.asarray([after.x, after.y, after.yaw]))
