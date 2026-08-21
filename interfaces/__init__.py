"""统一接口包：三阶段共用的数据契约。"""

from .bev import BEVTensor
from .control import ControlCmd
from .sensor import CameraFrame, CameraIntrinsics, LiDARFrame
from .state import GoalPose, VehicleState
from .trajectory import Trajectory, TrajectoryPoint

__all__ = [
    "BEVTensor",
    "ControlCmd",
    "CameraFrame",
    "CameraIntrinsics",
    "LiDARFrame",
    "GoalPose",
    "VehicleState",
    "Trajectory",
    "TrajectoryPoint",
]