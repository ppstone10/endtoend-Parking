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
from .checkpoint import initialize_model_from_checkpoint
from .closed_loop_selection import ClosedLoopSelector, select_best_checkpoint

from .config import TrainingRunConfig, load_training_run_config
from .data import (
    model_horizon,
    prepare_batches,
    recovery_sample_groups,
    validate_model_dataset,
)
from .reporting import atomic_write_json, save_training_artifacts
from .stop_calibration import calibrate_stop_threshold, write_deployment_checkpoint
from .trainer import Trainer, TrainingHistory
from .safety import (
    SweptFootprintLoss,
    build_clearance_fields,
    safety_geometry_from_dataset,
)


def run_training(config: TrainingRunConfig) -> dict[str, Any]:
    """执行一次配置化训练，并返回最终可序列化报告。"""
    torch.manual_seed(config.trainer.seed)
    train_data = DatasetGenerator.load(config.train_data)
    val_data = DatasetGenerator.load(config.val_data)
    _validate_dataset_pair(train_data, val_data)
    model = build_model(config.model_name, config.model_config)
    validate_model_dataset(model, train_data)
    validate_model_dataset(model, val_data)
    initialization = None
    if config.initialize_from is not None:
        initialization = initialize_model_from_checkpoint(
            model,
            config.initialize_from,
            model_name=config.model_name,
            model_config=config.model_config,
        )
    horizon = model_horizon(model)
    safety_loss = None
    train_clearance = None
    val_clearance = None
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
            mode=config.trainer.safety_loss_mode,
            goal_exempt_radius_m=config.trainer.safety_goal_exempt_radius_m,
            goal_exempt_weight=config.trainer.safety_goal_exempt_weight,
        )
        if config.trainer.safety_loss_mode in {"clearance_field", "tiered_clearance"}:
            train_clearance = build_clearance_fields(
                train_data["bevs"],
                train_geometry,
                extra_margin_m=config.trainer.safety_extra_margin_m,
            )
            val_clearance = build_clearance_fields(
                val_data["bevs"],
                val_geometry,
                extra_margin_m=config.trainer.safety_extra_margin_m,
            )
    train_groups = None
    if config.trainer.balance_recovery_batches:
        train_groups = recovery_sample_groups(train_data["task_meta"])
    train_batches = prepare_batches(
        train_data,
        horizon=horizon,
        batch_size=config.batch_size,
        clearance_fields=train_clearance,
        sample_groups=train_groups,
    )
    val_batches = prepare_batches(
        val_data,
        horizon=horizon,
        batch_size=config.batch_size,
        clearance_fields=val_clearance,
    )
    trainer = Trainer(
        model,
        config.trainer,
        model_name=config.model_name,
        model_config=config.model_config,
        safety_loss=safety_loss,
        initialization=initialization,
    )

    # 闭环选型：按"闭环成功率"另存 best_closed_loop.pt，并据此早停/部署。
    selector = None
    selection_state: dict[str, Any] = {
        "enabled": False,
        "best_score": None,
        "best_epoch": None,
        "stale_epochs": 0,
        "stopped_early": False,
        "history": [],
    }
    if config.closed_loop_selection is not None and config.closed_loop_selection.enabled:
        selection_config = config.closed_loop_selection
        selector = ClosedLoopSelector(
            selection_config,
            data_path=(selection_config.data or str(config.val_data)),
            device=config.trainer.device,
        )
        selection_state["enabled"] = True
        selection_state["data"] = str(selector.data_path)
        selection_state["selected_indices"] = selector.selected_indices

    def record_progress(epoch: int, history: TrainingHistory) -> None:
        """每轮记录进度；启用闭环选型时按闭环成功率决定 best 与早停。

        注意：当 `final_selection_samples > 0` 时进入**训练后排序选型**模式——
        本轮只保存周期快照，不再即时刷新 `best_closed_loop.pt`。
        原因：即时选型是贪心的，实测会把 best 锁死在早期较差 epoch
        （v17 锁在 epoch 4 的 25%），而事后排序能发现 epoch 24 的 35%。
        """
        if selector is not None:
            selection_config = config.closed_loop_selection
            assert selection_config is not None
            snapshot_every = int(selection_config.snapshot_every_epochs)
            if snapshot_every > 0 and (epoch + 1) % snapshot_every == 0:
                trainer._save_checkpoint(f"epoch{epoch + 1:04d}.pt", epoch, history)
            deferred_selection = int(selection_config.final_selection_samples) > 0
            should_evaluate = (
                (epoch + 1) % selection_config.every_epochs == 0
                or epoch + 1 == config.trainer.epochs
            )
            if should_evaluate:
                metrics = selector.evaluate(trainer.model)
                selection_state["history"].append({"epoch": epoch, **metrics})
                score = float(metrics["score"])
                best = selection_state["best_score"]
                if best is None or score > float(best):
                    selection_state["best_score"] = score
                    selection_state["best_epoch"] = epoch
                    selection_state["stale_epochs"] = 0
                    if not deferred_selection:
                        trainer._save_checkpoint("best_closed_loop.pt", epoch, history)
                else:
                    selection_state["stale_epochs"] += 1
                history.closed_loop_success_rate.append(score)
                history.closed_loop_collision_rate.append(
                    float(metrics["collision_rate"])
                )
                print(
                    f"  [closed-loop] epoch {epoch + 1} 成功 {score:.1%} "
                    f"碰撞 {metrics['collision_rate']:.1%} "
                    f"位置 {metrics['final_pos_err_mean']:.2f}m "
                    f"样本 {metrics['samples']} best {float(selection_state['best_score']):.1%}",
                    flush=True,
                )
                if (
                    not deferred_selection
                    and selection_config.patience > 0
                    and selection_state["stale_epochs"] >= selection_config.patience
                ):
                    # 标记即可：训练循环会在本轮结束后按标记停止，避免抛异常中断报告生成。
                    selection_state["stopped_early"] = True
                    history.closed_loop_stopped_early = True
                    print(
                        f"  [closed-loop] 连续 {selection_state['stale_epochs']} 次未提升，"
                        "按闭环指标早停",
                        flush=True,
                    )
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
    # 训练后排序选型：在全部候选快照上按闭环成功率排序，把最优权重的 model_state
    # 写入 best_closed_loop.pt。修复"训练中贪心即时锁定"导致最优权重不被部署的问题。
    closed_loop_checkpoint = config.output_dir / "best_closed_loop.pt"
    if (
        selector is not None
        and int(config.closed_loop_selection.final_selection_samples) > 0
    ):
        selection_config = config.closed_loop_selection
        print(
            f"  [final-selection] 在候选 checkpoint 上排序"
            f"（{selection_config.final_selection_samples} 样本）",
            flush=True,
        )
        selection_state["final_selection"] = select_best_checkpoint(
            config.output_dir,
            data_path=(selection_config.data or str(config.val_data)),
            samples=int(selection_config.final_selection_samples),
            device=config.trainer.device,
            max_steps=selection_config.max_steps,
            replan_every=selection_config.replan_every,
        )
        selected = selection_state["final_selection"]["best"]
        selection_state["best_score"] = selected["success_rate"]
        selection_state["best_epoch"] = selected["epoch"]
        print(
            f"  [final-selection] 选定 {selected['checkpoint']}（epoch {selected['epoch']}）"
            f" 成功 {selected['success_rate']:.1%} 碰撞 {selected['collision_rate']:.1%}",
            flush=True,
        )
    # 启用闭环选型且已产出结果时，部署用闭环最优权重而非 val_loss 最优。
    selected_by_closed_loop = False
    if selection_state["enabled"] and closed_loop_checkpoint.is_file():
        best_checkpoint = closed_loop_checkpoint
        selected_by_closed_loop = True
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
        "initialization": initialization,
        "data": {
            "train": str(config.train_data),
            "val": str(config.val_data),
            "train_samples": int(train_data["bevs"].shape[0]),
            "val_samples": int(val_data["bevs"].shape[0]),
        },
        "history": history.to_dict(),
        "metrics": metrics.to_dict(),
        "closed_loop_selection": {
            **{key: value for key, value in selection_state.items() if key != "history"},
            "selected_checkpoint": (
                str(closed_loop_checkpoint) if selected_by_closed_loop else str(config.output_dir / "best.pt")
            ),
            "selected_by_closed_loop": selected_by_closed_loop,
            "evaluations": selection_state["history"],
        },
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
