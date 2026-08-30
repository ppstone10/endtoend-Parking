"""MineParkingNet 训练基础设施。"""

from .config import TrainingRunConfig, load_training_run_config
from .data import epoch_batches, model_horizon, prepare_batches, to_local, validate_model_dataset
from .reporting import TrainingArtifacts, save_training_artifacts
from .trainer import Trainer, TrainerConfig, TrainingHistory
from .safety import SafetyGeometry, SweptFootprintLoss, safety_geometry_from_dataset
from .stop_calibration import calibrate_stop_threshold, write_deployment_checkpoint

__all__ = [
    "Trainer",
    "TrainerConfig",
    "SafetyGeometry",
    "SweptFootprintLoss",
    "safety_geometry_from_dataset",
    "TrainingHistory",
    "TrainingRunConfig",
    "TrainingArtifacts",
    "load_training_run_config",
    "epoch_batches",
    "model_horizon",
    "prepare_batches",
    "save_training_artifacts",
    "to_local",
    "validate_model_dataset",
    "calibrate_stop_threshold",
    "write_deployment_checkpoint",
]
