"""Python 仿真环境包。"""

from .camera_model import CameraModel
from .environment import ParkingEnvironment, RectangleObstacle
from .sensor_camera import SimulatedCamera
from .sensor_sim import SimulatedLiDAR
from .vehicle_config import (
    LEGACY_4X2,
    MINING_TRUCK,
    VEHICLE_PRESETS,
    VehicleConfig,
    get_vehicle,
)
from .vehicle_model import DifferentialDriveModel

__all__ = [
    "ParkingEnvironment",
    "RectangleObstacle",
    "CameraModel",
    "SimulatedCamera",
    "SimulatedLiDAR",
    "DifferentialDriveModel",
    "VehicleConfig",
    "MINING_TRUCK",
    "LEGACY_4X2",
    "VEHICLE_PRESETS",
    "get_vehicle",
]