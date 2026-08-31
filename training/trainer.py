"""配置化训练、验证、early stopping 与原子 checkpoint。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

from model import loss_fn, variable_loss_fn


Batch = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TrainerConfig:
    """Trainer 核心配置。"""

    epochs: int = 30
    learning_rate: float = 1e-3
    patience: int = 5
    min_delta: float = 0.0
    device: str = "cpu"
    checkpoint_dir: str = "runs/training"
    gradient_clip_norm: float | None = None
    stop_loss_weight: float = 0.2
    seed: int = 0
    shuffle_train: bool = True
    balance_stop_loss: bool = True
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.2
    teacher_forcing_decay_epochs: int = 15
    early_stopping_start_epoch: int = 0
    stop_target_mode: str = "terminal"
    collision_loss_weight: float = 0.0
    safety_extra_margin_m: float = 0.1
    safety_sample_spacing_m: float = 0.5
    safety_max_swept_substeps: int = 16
    safety_out_of_bounds_weight: float = 1.0
    safety_loss_mode: str = "occupancy_max"
    safety_goal_exempt_radius_m: float = 0.0
    safety_goal_exempt_weight: float = 0.0
    balance_recovery_batches: bool = False
    endpoint_alignment_weight: float = 0.0
    endpoint_alignment_tail_points: int = 8

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs 必须为正")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须为有限正数")
        if self.patience <= 0:
            raise ValueError("patience 必须为正")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta 必须为有限非负数")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm 必须为正或 None")
        if not math.isfinite(self.stop_loss_weight) or self.stop_loss_weight < 0.0:
            raise ValueError("stop_loss_weight 必须为有限非负数")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed 必须为非负整数")
        if not isinstance(self.shuffle_train, bool):
            raise ValueError("shuffle_train 必须为布尔值")
        if not isinstance(self.balance_stop_loss, bool):
            raise ValueError("balance_stop_loss 必须为布尔值")
        for name, value in (
            ("teacher_forcing_start", self.teacher_forcing_start),
            ("teacher_forcing_end", self.teacher_forcing_end),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须为 [0,1] 内有限数")
        if self.teacher_forcing_start < self.teacher_forcing_end:
            raise ValueError("teacher_forcing_start 不得小于 teacher_forcing_end")
        if (
            isinstance(self.teacher_forcing_decay_epochs, bool)
            or not isinstance(self.teacher_forcing_decay_epochs, int)
            or self.teacher_forcing_decay_epochs <= 0
        ):
            raise ValueError("teacher_forcing_decay_epochs 必须为正整数")
        if (
            isinstance(self.early_stopping_start_epoch, bool)
            or not isinstance(self.early_stopping_start_epoch, int)
            or not 0 <= self.early_stopping_start_epoch < self.epochs
        ):
            raise ValueError("early_stopping_start_epoch 必须位于 [0, epochs) 内")
        if self.stop_target_mode not in {"terminal", "cumulative"}:
            raise ValueError("stop_target_mode 必须为 terminal 或 cumulative")
        for name, value in (
            ("collision_loss_weight", self.collision_loss_weight),
            ("safety_extra_margin_m", self.safety_extra_margin_m),
            ("safety_out_of_bounds_weight", self.safety_out_of_bounds_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须为有限非负数")
        if not math.isfinite(self.safety_sample_spacing_m) or self.safety_sample_spacing_m <= 0.0:
            raise ValueError("safety_sample_spacing_m 必须为有限正数")
        if (
            isinstance(self.safety_max_swept_substeps, bool)
            or not isinstance(self.safety_max_swept_substeps, int)
            or self.safety_max_swept_substeps <= 0
        ):
            raise ValueError("safety_max_swept_substeps 必须为正整数")
        if self.safety_loss_mode not in {"occupancy_max", "clearance_field"}:
            raise ValueError("safety_loss_mode 必须为 occupancy_max 或 clearance_field")
        for name, value in (
            ("safety_goal_exempt_radius_m", self.safety_goal_exempt_radius_m),
            ("safety_goal_exempt_weight", self.safety_goal_exempt_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须为有限非负数")
        if not 0.0 <= self.safety_goal_exempt_weight <= 1.0:
            raise ValueError("safety_goal_exempt_weight 必须位于 [0,1]")
        if not isinstance(self.balance_recovery_batches, bool):
            raise ValueError("balance_recovery_batches 必须为布尔值")
        if not math.isfinite(self.endpoint_alignment_weight) or self.endpoint_alignment_weight < 0.0:
            raise ValueError("endpoint_alignment_weight 必须为有限非负数")
        if (
            isinstance(self.endpoint_alignment_tail_points, bool)
            or not isinstance(self.endpoint_alignment_tail_points, int)
            or self.endpoint_alignment_tail_points <= 0
        ):
            raise ValueError("endpoint_alignment_tail_points 必须为正整数")

    def teacher_forcing_ratio(self, epoch: int) -> float:
        """线性退火并在终值处截断。"""
        if epoch >= self.teacher_forcing_decay_epochs:
            return self.teacher_forcing_end
        progress = max(epoch, 0) / self.teacher_forcing_decay_epochs
        return self.teacher_forcing_start + (
            self.teacher_forcing_end - self.teacher_forcing_start
        ) * progress


@dataclass
class TrainingHistory:
    """逐 epoch 训练证据。"""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_collision_loss: list[float] = field(default_factory=list)
    val_collision_loss: list[float] = field(default_factory=list)
    teacher_forcing_ratio: list[float] = field(default_factory=list)
    train_rollout_ade_m: list[float] = field(default_factory=list)
    train_rollout_fde_m: list[float] = field(default_factory=list)
    val_rollout_ade_m: list[float] = field(default_factory=list)
    val_rollout_fde_m: list[float] = field(default_factory=list)
    train_stop_found_rate: list[float | None] = field(default_factory=list)
    train_predicted_length_mae_points: list[float | None] = field(default_factory=list)
    val_stop_found_rate: list[float | None] = field(default_factory=list)
    val_predicted_length_mae_points: list[float | None] = field(default_factory=list)
    early_stopping_active: list[bool] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = math.inf
    stopped_early: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Trainer:
    """模型无关的训练/验证编排器。"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        *,
        model_name: str,
        model_config: dict | None = None,
        safety_loss: nn.Module | None = None,
        initialization: dict | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.model_name = model_name
        self.model_config = {} if model_config is None else dict(model_config)
        self.safety_loss = safety_loss
        self.initialization = None if initialization is None else dict(initialization)
        if config.collision_loss_weight > 0.0 and safety_loss is None:
            raise ValueError("collision_loss_weight > 0 时必须提供 safety_loss")
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.checkpoint_dir = Path(config.checkpoint_dir)

    def _move_batch(self, batch: Batch) -> Batch:
        return tuple(tensor.to(self.device) for tensor in batch)  # type: ignore[return-value]

    def _batch_loss(
        self,
        batch: Batch,
        *,
        training: bool,
        epoch: int,
        batch_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moved = self._move_batch(batch)
        bev, goal, state, target, mask = moved[:5]
        clearance_field = moved[5] if len(moved) >= 6 and moved[5].ndim == 4 else None
        forward_with_stop = getattr(self.model, "forward_with_stop", None)
        if callable(forward_with_stop):
            teacher_forcing_ratio = (
                self.config.teacher_forcing_ratio(epoch) if training else 0.0
            )
            sampling_generator = None
            if training and 0.0 < teacher_forcing_ratio < 1.0:
                sampling_generator = torch.Generator(device=self.device.type)
                sampling_generator.manual_seed(
                    self.config.seed + epoch * 1_000_003 + batch_index
                )
            prediction = forward_with_stop(
                bev,
                goal,
                state,
                teacher_points=target if training else None,
                teacher_forcing_ratio=teacher_forcing_ratio,
                sampling_generator=sampling_generator,
            )
            base_loss = variable_loss_fn(
                prediction.points,
                prediction.stop_logits,
                target,
                mask,
                stop_weight=self.config.stop_loss_weight,
                balance_stop=self.config.balance_stop_loss,
                stop_target_mode=self.config.stop_target_mode,
            )
            points = prediction.points
        else:
            points = self.model(bev, goal, state)
            base_loss = loss_fn(points, target, mask)
        if self.safety_loss is None or self.config.collision_loss_weight == 0.0:
            collision_loss = torch.zeros((), dtype=base_loss.dtype, device=base_loss.device)
            total = base_loss
        else:
            collision_loss = self.safety_loss(bev, points, mask, clearance_field, goal)
            total = base_loss + self.config.collision_loss_weight * collision_loss
        if self.config.endpoint_alignment_weight > 0.0:
            from model import endpoint_alignment_loss

            endpoint_loss = endpoint_alignment_loss(
                points,
                target,
                mask,
                tail_points=self.config.endpoint_alignment_tail_points,
            )
            total = total + self.config.endpoint_alignment_weight * endpoint_loss
        return total, collision_loss

    def _run_epoch(
        self, batches: tuple[Batch, ...], *, training: bool, epoch: int
    ) -> tuple[float, float]:
        if not batches:
            raise ValueError("训练和验证 batches 均不能为空")
        from .data import epoch_batches

        active_batches = epoch_batches(
            batches,
            shuffle=training and self.config.shuffle_train,
            seed=self.config.seed,
            epoch=epoch,
            balance_recovery=training and self.config.balance_recovery_batches,
        )
        self.model.train(training)
        total = 0.0
        collision_total = 0.0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_index, batch in enumerate(active_batches):
                if training:
                    self.optimizer.zero_grad()
                loss, collision_loss = self._batch_loss(
                    batch,
                    training=training,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                if training:
                    loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                    self.optimizer.step()
                total += float(loss.detach().cpu())
                collision_total += float(collision_loss.detach().cpu())
        return total / len(active_batches), collision_total / len(active_batches)

    @torch.no_grad()
    def _rollout_metrics(self, batches: tuple[Batch, ...]) -> dict[str, float | None]:
        """用推理时自由反馈计算逐点误差与变长停止质量。"""
        self.model.eval()
        distance_sum = 0.0
        point_count = 0
        final_distance_sum = 0.0
        sequence_count = 0
        stop_found = 0
        length_error_sum = 0.0
        variable_model = callable(getattr(self.model, "forward_with_stop", None))
        for batch in batches:
            bev, goal, state, target, mask = self._move_batch(batch)[:5]
            if variable_model:
                prediction = self.model.forward_with_stop(bev, goal, state)
                points = prediction.points
                probabilities = torch.sigmoid(prediction.stop_logits)
            else:
                points = self.model(bev, goal, state)
                probabilities = None
            lengths = mask.sum(dim=1).long()
            distances = torch.linalg.vector_norm(points[..., :2] - target[..., :2], dim=-1)
            distance_sum += float((distances * mask).sum().cpu())
            point_count += int(mask.sum().item())
            rows = torch.arange(lengths.shape[0], device=self.device)
            valid = lengths > 0
            if valid.any():
                final_distance_sum += float(
                    distances[rows[valid], lengths[valid] - 1].sum().cpu()
                )
                sequence_count += int(valid.sum().item())
            if probabilities is not None:
                reached = probabilities >= float(getattr(self.model, "stop_threshold", 0.5))
                has_stop = reached.any(dim=1)
                predicted_lengths = torch.where(
                    has_stop,
                    reached.to(torch.int64).argmax(dim=1) + 1,
                    torch.full_like(lengths, reached.shape[1]),
                )
                stop_found += int(has_stop.sum().item())
                length_error_sum += float(
                    (predicted_lengths - lengths).abs().sum().cpu()
                )
        if point_count == 0 or sequence_count == 0:
            raise ValueError("自由滚动评估至少需要一个有效轨迹点")
        return {
            "ade_m": distance_sum / point_count,
            "fde_m": final_distance_sum / sequence_count,
            "stop_found_rate": stop_found / sequence_count if variable_model else None,
            "predicted_length_mae_points": (
                length_error_sum / sequence_count if variable_model else None
            ),
        }

    def _checkpoint_payload(
        self, epoch: int, history: TrainingHistory
    ) -> dict:
        return {
            "schema_version": 1,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "trainer_config": asdict(self.config),
            "epoch": epoch,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "history": history.to_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "safety_loss_config": (
                None
                if self.safety_loss is None
                else getattr(self.safety_loss, "to_metadata")()
            ),
            "initialization": self.initialization,
        }

    def _save_checkpoint(
        self, name: str, epoch: int, history: TrainingHistory
    ) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / name
        temporary = path.with_name(f"{path.name}.tmp")
        torch.save(self._checkpoint_payload(epoch, history), temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path: str | Path) -> tuple[int, TrainingHistory]:
        """恢复兼容 checkpoint，返回下一 epoch 与历史。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema_version") != 1:
            raise RuntimeError("不支持的训练 checkpoint schema")
        if payload.get("resumable", True) is not True:
            raise RuntimeError("deployment checkpoint 仅供推理，不可恢复训练")
        if payload.get("model_name") != self.model_name:
            raise RuntimeError("checkpoint 模型变体与当前 Trainer 不一致")
        if payload.get("model_config", {}) != self.model_config:
            raise RuntimeError("checkpoint 模型配置与当前 Trainer 不一致")
        stored_trainer = payload.get("trainer_config", {})
        current_trainer = asdict(self.config)
        compatibility_fields = {
            "learning_rate",
            "patience",
            "min_delta",
            "gradient_clip_norm",
            "stop_loss_weight",
            "shuffle_train",
            "balance_stop_loss",
            "teacher_forcing_start",
            "teacher_forcing_end",
            "teacher_forcing_decay_epochs",
            "early_stopping_start_epoch",
            "stop_target_mode",
            "collision_loss_weight",
            "safety_extra_margin_m",
            "safety_sample_spacing_m",
            "safety_max_swept_substeps",
            "safety_out_of_bounds_weight",
            "safety_loss_mode",
            "safety_goal_exempt_radius_m",
            "safety_goal_exempt_weight",
            "balance_recovery_batches",
            "endpoint_alignment_weight",
            "endpoint_alignment_tail_points",
        }
        if any(
            stored_trainer.get(field) != current_trainer.get(field)
            for field in compatibility_fields
        ):
            raise RuntimeError("checkpoint 训练超参数与当前 Trainer 不一致")
        current_safety = (
            None
            if self.safety_loss is None
            else getattr(self.safety_loss, "to_metadata")()
        )
        if payload.get("safety_loss_config") != current_safety:
            raise RuntimeError("checkpoint 安全几何与当前 Trainer 不一致")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.initialization = payload.get("initialization")
        history = TrainingHistory(**payload["history"])
        return int(payload["epoch"]) + 1, history

    def fit(
        self,
        train_batches: Iterable[Batch],
        val_batches: Iterable[Batch],
        *,
        resume_from: str | Path | None = None,
        on_epoch_end: Callable[[int, TrainingHistory], None] | None = None,
    ) -> TrainingHistory:
        """训练并按验证损失 early stop；每 epoch 原子保存 last。"""
        train_data = tuple(train_batches)
        val_data = tuple(val_batches)
        start_epoch = 0
        history = TrainingHistory()
        if resume_from is not None:
            start_epoch, history = self.load_checkpoint(resume_from)
        monitoring_start = self.config.early_stopping_start_epoch
        if start_epoch <= monitoring_start:
            stale_epochs = 0
        elif history.best_epoch >= monitoring_start:
            stale_epochs = max(0, start_epoch - history.best_epoch - 1)
        else:
            stale_epochs = start_epoch - monitoring_start
        for epoch in range(start_epoch, self.config.epochs):
            train_loss, train_collision_loss = self._run_epoch(
                train_data, training=True, epoch=epoch
            )
            val_loss, val_collision_loss = self._run_epoch(
                val_data, training=False, epoch=epoch
            )
            train_rollout = self._rollout_metrics(train_data)
            val_rollout = self._rollout_metrics(val_data)
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.train_collision_loss.append(train_collision_loss)
            history.val_collision_loss.append(val_collision_loss)
            history.teacher_forcing_ratio.append(
                self.config.teacher_forcing_ratio(epoch)
            )
            history.train_rollout_ade_m.append(float(train_rollout["ade_m"]))
            history.train_rollout_fde_m.append(float(train_rollout["fde_m"]))
            history.val_rollout_ade_m.append(float(val_rollout["ade_m"]))
            history.val_rollout_fde_m.append(float(val_rollout["fde_m"]))
            history.train_stop_found_rate.append(train_rollout["stop_found_rate"])
            history.train_predicted_length_mae_points.append(
                train_rollout["predicted_length_mae_points"]
            )
            history.val_stop_found_rate.append(val_rollout["stop_found_rate"])
            history.val_predicted_length_mae_points.append(
                val_rollout["predicted_length_mae_points"]
            )
            monitoring_active = epoch >= monitoring_start
            history.early_stopping_active.append(monitoring_active)
            improved = val_loss < history.best_val_loss - self.config.min_delta
            if improved:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                if monitoring_active:
                    stale_epochs = 0
                self._save_checkpoint("best.pt", epoch, history)
            elif monitoring_active:
                stale_epochs += 1
            else:
                stale_epochs = 0
            self._save_checkpoint("last.pt", epoch, history)
            if on_epoch_end is not None:
                on_epoch_end(epoch, history)
            if monitoring_active and stale_epochs >= self.config.patience:
                history.stopped_early = True
                break
        return history
