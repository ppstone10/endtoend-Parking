"""配置化训练、验证、early stopping 与原子 checkpoint。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

from model import loss_fn, variable_loss_fn


Batch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


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


@dataclass
class TrainingHistory:
    """逐 epoch 训练证据。"""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
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
    ) -> None:
        self.model = model
        self.config = config
        self.model_name = model_name
        self.model_config = {} if model_config is None else dict(model_config)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.checkpoint_dir = Path(config.checkpoint_dir)

    def _move_batch(self, batch: Batch) -> Batch:
        return tuple(tensor.to(self.device) for tensor in batch)  # type: ignore[return-value]

    def _batch_loss(self, batch: Batch, *, training: bool) -> torch.Tensor:
        bev, goal, state, target, mask = self._move_batch(batch)
        forward_with_stop = getattr(self.model, "forward_with_stop", None)
        if callable(forward_with_stop):
            prediction = forward_with_stop(
                bev,
                goal,
                state,
                teacher_points=target if training else None,
            )
            return variable_loss_fn(
                prediction.points,
                prediction.stop_logits,
                target,
                mask,
                stop_weight=self.config.stop_loss_weight,
            )
        return loss_fn(self.model(bev, goal, state), target, mask)

    def _run_epoch(self, batches: tuple[Batch, ...], *, training: bool) -> float:
        if not batches:
            raise ValueError("训练和验证 batches 均不能为空")
        self.model.train(training)
        total = 0.0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in batches:
                if training:
                    self.optimizer.zero_grad()
                loss = self._batch_loss(batch, training=training)
                if training:
                    loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                    self.optimizer.step()
                total += float(loss.detach().cpu())
        return total / len(batches)

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
        }
        if any(
            stored_trainer.get(field) != current_trainer.get(field)
            for field in compatibility_fields
        ):
            raise RuntimeError("checkpoint 训练超参数与当前 Trainer 不一致")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
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
        stale_epochs = (
            max(0, start_epoch - history.best_epoch - 1)
            if history.best_epoch >= 0
            else 0
        )
        for epoch in range(start_epoch, self.config.epochs):
            train_loss = self._run_epoch(train_data, training=True)
            val_loss = self._run_epoch(val_data, training=False)
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            improved = val_loss < history.best_val_loss - self.config.min_delta
            if improved:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                stale_epochs = 0
                self._save_checkpoint("best.pt", epoch, history)
            else:
                stale_epochs += 1
            self._save_checkpoint("last.pt", epoch, history)
            if on_epoch_end is not None:
                on_epoch_end(epoch, history)
            if stale_epochs >= self.config.patience:
                history.stopped_early = True
                break
        return history
