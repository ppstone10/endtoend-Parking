"""滚动闭环运行时包。"""

from .engine import ClosedLoopEngine
from .execution import (
    ExecutionError,
    IdealPathExecutor,
    MpcVehicleExecutor,
    TrajectoryExecutor,
)
from .recorder import EpisodeRecord
from .safety import FootprintTrajectorySafetyChecker, SafetyDecision, SafetyShieldStats
from .sources import (
    ExpertSource,
    GeometricFilterSource,
    HierarchicalPlanningSource,
    NetworkSource,
    ReplanningExpertSource,
    SafetyStopError,
    SafetyShieldSource,
    TrajectorySource,
)
from .termination import TerminalChecker
from .trajectory_repair import (
    FootprintFreeSpace,
    GeometricFilterStats,
    RepairOutcome,
    SweptFootprintProjector,
)

__all__ = [
    "ClosedLoopEngine",
    "EpisodeRecord",
    "ExecutionError",
    "ExpertSource",
    "FootprintFreeSpace",
    "GeometricFilterSource",
    "GeometricFilterStats",
    "HierarchicalPlanningSource",
    "IdealPathExecutor",
    "MpcVehicleExecutor",
    "NetworkSource",
    "RepairOutcome",
    "ReplanningExpertSource",
    "SafetyStopError",
    "SafetyShieldSource",
    "SweptFootprintProjector",
    "FootprintTrajectorySafetyChecker",
    "SafetyDecision",
    "SafetyShieldStats",
    "TrajectoryExecutor",
    "TrajectorySource",
    "TerminalChecker",
]
