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
    HierarchicalPlanningSource,
    NetworkSource,
    ReplanningExpertSource,
    SafetyStopError,
    SafetyShieldSource,
    TrajectorySource,
)
from .termination import TerminalChecker

__all__ = [
    "ClosedLoopEngine",
    "EpisodeRecord",
    "ExecutionError",
    "ExpertSource",
    "HierarchicalPlanningSource",
    "IdealPathExecutor",
    "MpcVehicleExecutor",
    "NetworkSource",
    "ReplanningExpertSource",
    "SafetyStopError",
    "SafetyShieldSource",
    "FootprintTrajectorySafetyChecker",
    "SafetyDecision",
    "SafetyShieldStats",
    "TrajectoryExecutor",
    "TrajectorySource",
    "TerminalChecker",
]
