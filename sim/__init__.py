"""Python 仿真环境包。"""

from .camera_model import CameraModel
from .environment import ParkingEnvironment
from .noise import (
    NOISE_PROFILES,
    CameraNoiseConfig,
    LiDARNoiseConfig,
    NoiseLevel,
    NoiseProfile,
    get_noise_profile,
)
from .obstacles import (
    KIND_BERM,
    KIND_CLIFF,
    KIND_EQUIPMENT,
    KIND_LINE,
    KIND_ROCK,
    KIND_VEHICLE,
    KIND_WALL,
    CircleObstacle,
    Obstacle,
    PolygonObstacle,
    RectangleObstacle,
)
from .sensor_camera import SimulatedCamera
from .sensor_sim import SimulatedLiDAR
from .tasks import (
    DistanceTier,
    DynamicObstacleEvent,
    Maneuver,
    Task,
    TaskCapability,
    TaskDifficulty,
    TaskGoal,
    TaskSampler,
    TaskType,
    UnsupportedTaskError,
)
from .vehicle_config import (
    LEGACY_4X2,
    MINING_DRILL_RIG,
    MINING_TRUCK,
    VEHICLE_PRESETS,
    VehicleConfig,
    get_vehicle,
    load_vehicle_config,
)
from .vehicle_model import DifferentialDriveModel

__all__ = [
    "ParkingEnvironment",
    "NOISE_PROFILES",
    "CameraNoiseConfig",
    "LiDARNoiseConfig",
    "NoiseLevel",
    "NoiseProfile",
    "get_noise_profile",
    "Obstacle",
    "RectangleObstacle",
    "PolygonObstacle",
    "CircleObstacle",
    "KIND_WALL",
    "KIND_BERM",
    "KIND_CLIFF",
    "KIND_ROCK",
    "KIND_VEHICLE",
    "KIND_EQUIPMENT",
    "KIND_LINE",
    "CameraModel",
    "SimulatedCamera",
    "SimulatedLiDAR",
    "DifferentialDriveModel",
    "VehicleConfig",
    "MINING_DRILL_RIG",
    "MINING_TRUCK",
    "LEGACY_4X2",
    "VEHICLE_PRESETS",
    "get_vehicle",
    "load_vehicle_config",
    "DistanceTier",
    "DynamicObstacleEvent",
    "Maneuver",
    "Task",
    "TaskCapability",
    "TaskDifficulty",
    "TaskGoal",
    "TaskSampler",
    "TaskType",
    "UnsupportedTaskError",
]
