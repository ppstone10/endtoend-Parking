"""配置化训练运行编排。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset import DatasetGenerator
from metrics.open_loop import evaluate_open_loop
from metrics.prediction_analysis import collect_open_loop_predictions
from model import build_model

from .config import TrainingRunConfig, load_training_run_config
from .data import model_horizon, prepare_batches, validate_model_dataset
from .reporting import atomic_write_json, save_training_artifacts
from .stop_calibration import calibrate_stop_threshold, write_deployment_checkpoint
from .trainer import Trainer, TrainingHistory
from .safety import SweptFootprintLoss, safety_geometry_from_dataset


def run_training(config: TrainingRunConfig) -> dict[str, Any]:
    """执行一次配置化训练，并返回最终可序列化报告。"""
    torch.manual_seed(config.trainer.seed)
    train_data = DatasetGenerator.load(config.train_data)
    val_data = DatasetGenerator.load(config.val_data)
    _validate_dataset_pair(train_data, val_data)
    model = build_model(config.model_name, config.model_config)
    validate_model_dataset(model, train_data)
    validate_model_dataset(model, val_data)
    horizon = model_horizon(model)
    train_batches = prepare_batches(
        train_data, horizon=horizon, batch_size=config.batch_size
    )
    val_batches = prepare_batches(
        val_data, horizon=horizon, batch_size=config.batch_size
    )
    safety_loss = None
    if config.trainer.collision_loss_weight > 0.0:
        train_geometry = safety_geometry_from_dataset(train_data)
        val_geometry = safety_geometry_from_dataset(val_data)
        if train_geometry != val_geometry:
            raise ValueError("启用碰撞损失时 train/val 安全几何必须一致")
        safety_loss = SweptFootprintLoss(
            train_geometry,
            extra_margin_m=config.trainer.safety_extra_margin_m,
            sample_spacing_m=config.trainer.safety_sample_spacing_m,
            max_swept_substeps=config.trainer.safety_max_swept_substeps,
            out_of_bounds_weight=config.trainer.safety_out_of_bounds_weight,
        )
    trainer = Trainer(
        model,
        config.trainer,
        model_name=config.model_name,
        model_config=config.model_config,
        safety_loss=safety_loss,
    )

    def record_progress(epoch: int, history: TrainingHistory) -> None:
        atomic_write_json(config.output_dir / "history.json", history.to_dict())
        stop_rate = history.val_stop_found_rate[-1]
        stop_summary = (
            "" if stop_rate is None else f" stop={stop_rate:.3f}"
        )
        early_stop_summary = (
            "active" if history.early_stopping_active[-1] else "warmup"
        )
        print(
            f"epoch {epoch + 1}/{config.trainer.epochs} "
            f"train={history.train_loss[-1]:.6f} "
            f"val={history.val_loss[-1]:.6f} "
            f"collision={history.val_collision_loss[-1]:.4f} "
            f"rollout_val_ade={history.val_rollout_ade_m[-1]:.3f}m "
            f"rollout_val_fde={history.val_rollout_fde_m[-1]:.3f}m "
            f"teacher={history.teacher_forcing_ratio[-1]:.3f} "
            f"best={history.best_val_loss:.6f}{stop_summary} "
            f"early_stop={early_stop_summary}",
            flush=True,
        )

    history = trainer.fit(
        train_batches,
        val_batches,
        resume_from=config.resume_from,
        on_epoch_end=record_progress,
    )
    best_checkpoint = config.output_dir / "best.pt"
    if not best_checkpoint.is_file():
        raise RuntimeError("训练结束但 best checkpoint 不存在")
    trainer.load_checkpoint(best_checkpoint)
    deployment_checkpoint = best_checkpoint
    calibration: dict[str, Any] = {"status": "not_applicable"}
    calibration_artifact: str | None = None
    if callable(getattr(trainer.model, "forward_with_stop", None)):
        predictions = collect_open_loop_predictions(
            trainer.model, val_batches, device=config.trainer.device
        )
        if predictions.stop_logits is None:
            raise RuntimeError("变长模型未返回停止 logits")
        calibration_result = calibrate_stop_threshold(
            predictions.stop_logits, predictions.masks
        )
        selected_threshold = float(calibration_result["selected_threshold"])
        calibration_path = atomic_write_json(
            config.output_dir / "stop_threshold_calibration.json",
            calibration_result,
        )
        deployment_checkpoint = write_deployment_checkpoint(
            best_checkpoint,
            config.output_dir / "deployment.pt",
            threshold=selected_threshold,
            calibration=calibration_result,
        )
        setattr(trainer.model, "stop_threshold", selected_threshold)
        calibration_artifact = str(calibration_path)
        calibration = {
            "status": "calibrated_on_validation",
            "stop_threshold": selected_threshold,
            "length_mae_points": calibration_result["selected"]["length_mae_points"],
            "length_bias_points": calibration_result["selected"]["length_bias_points"],
            "stop_found_rate": calibration_result["selected"]["stop_found_rate"],
            "artifact": calibration_artifact,
        }
    metrics = evaluate_open_loop(
        trainer.model, val_batches, device=config.trainer.device
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "config": str(config.source),
        "model_name": config.model_name,
        "model_config": config.model_config,
        "trainer_config": asdict(config.trainer),
        "data": {
            "train": str(config.train_data),
            "val": str(config.val_data),
            "train_samples": int(train_data["bevs"].shape[0]),
            "val_samples": int(val_data["bevs"].shape[0]),
        },
        "history": history.to_dict(),
        "metrics": metrics.to_dict(),
        "calibration": calibration,
        "checkpoints": {
            "best": str(best_checkpoint),
            "last": str(config.output_dir / "last.pt"),
            "deployment": str(deployment_checkpoint),
        },
        "artifacts": {
            "history": str(config.output_dir / "history.json"),
            "curve_png": str(config.output_dir / "training_curve.png"),
            "curve_pdf": str(config.output_dir / "training_curve.pdf"),
            "stop_threshold_calibration": calibration_artifact,
        },
    }
    save_training_artifacts(history, report, config.output_dir)
    return report


def run_training_from_yaml(path: str | Path) -> dict[str, Any]:
    return run_training(load_training_run_config(path))


def _validate_dataset_pair(train_data: dict, val_data: dict) -> None:
    if train_data["bevs"].shape[1:] != val_data["bevs"].shape[1:]:
        raise ValueError("train/val 的 BEV shape 不一致")
    train_dt = float(np.asarray(train_data["dt"]).reshape(-1)[0])
    val_dt = float(np.asarray(val_data["dt"]).reshape(-1)[0])
    if not np.isclose(train_dt, val_dt):
        raise ValueError("train/val 的轨迹 dt 不一致")
