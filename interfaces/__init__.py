"""统一接口包：三阶段共用的数据契约。"""

from .bev import BEVConfig, BEVTensor
from .control import ControlCmd
from .sensor import CameraFrame, CameraIntrinsics, CameraView, LiDARFrame
from .state import GoalPose, VehicleState
from .trajectory import Trajectory, TrajectoryPoint

__all__ = [
    "BEVConfig",
    "BEVTensor",
    "ControlCmd",
    "CameraFrame",
    "CameraView",
    "CameraIntrinsics",
    "LiDARFrame",
    "GoalPose",
    "VehicleState",
    "Trajectory",
    "TrajectoryPoint",
]
