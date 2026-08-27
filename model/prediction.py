"""模型变长轨迹输出契约。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectoryPrediction:
    """批量轨迹点与逐步终止 logits。"""

    points: torch.Tensor
    stop_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 3 or self.points.shape[-1] != 3:
            raise ValueError("points 必须是 (B,N,3)")
        if self.stop_logits is not None and self.stop_logits.shape != self.points.shape[:2]:
            raise ValueError("stop_logits 必须是 (B,N) 并与 points 对齐")
