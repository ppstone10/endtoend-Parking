"""数据集包。"""

from .build import BuildReport, build_task_plan, expert_maneuvers, generate_with_retries
from .generator import DatasetGenerator, TaskGenerationError, TrainingSample
from .inspection import (
    render_sample_overlay,
    select_representative_indices,
    summarize_dataset,
)
from .maneuver import (
    ManeuverAudit,
    audit_maneuver_consistency,
    require_maneuver_consistency,
)
from .pipeline import SensorBEVPipeline
from .splits import DatasetSplits, split_tasks

__all__ = [
    "BuildReport",
    "build_task_plan",
    "expert_maneuvers",
    "generate_with_retries",
    "DatasetGenerator",
    "TaskGenerationError",
    "TrainingSample",
    "SensorBEVPipeline",
    "DatasetSplits",
    "split_tasks",
    "render_sample_overlay",
    "require_maneuver_consistency",
    "select_representative_indices",
    "summarize_dataset",
    "ManeuverAudit",
    "audit_maneuver_consistency",
]
