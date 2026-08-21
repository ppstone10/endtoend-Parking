"""控制指令定义。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlCmd:
    """底盘控制指令，差分驱动模型。

    v 为线速度（米/秒），omega 为角速度（弧度/秒），对应差分驱动左右轮速控制。
    """

    v: float
    omega: float

    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.omega], dtype=np.float64)