"""可视化统一风格与保存工具。

图内标签使用英文（避免跨平台中文字体缺字导致 PDF 输出异常）；
全部图输出 PNG+PDF 双格式（论文插图直接取 PDF 矢量版）。
"""

from __future__ import annotations

import matplotlib as mpl

# 统一色表：轨迹类型 → 颜色/线型语义（论文全套图保持一致）。
COLORS = {
    "expert": "#2ca02c",   # 专家轨迹：绿
    "plan": "#ff7f0e",     # 网络/规划轨迹：橙
    "actual": "#1f77b4",   # 实际执行轨迹：蓝
    "obstacle": "#7f7f7f", # 障碍物：灰
    "spot": "#d62728",     # 目标车位：红
    "vehicle": "#17becf",  # 车辆矩形：青
    "bev_cmap": "viridis",
}

LINESTYLES = {
    "expert": "--",
    "plan": "-.",
    "actual": "-",
}

DEFAULT_DPI = 150


def setup_style() -> None:
    """应用统一 rcParams（白底、无多余刻度、合适字号）。"""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
            "savefig.dpi": DEFAULT_DPI,
        }
    )


def save_fig(fig, path: str) -> list[str]:
    """保存 PNG+PDF 双格式，返回实际写入的文件路径列表。

    path 不带扩展名（如 out/fig1）时生成 fig1.png 与 fig1.pdf；
    带扩展名则按扩展名只存一份。
    """
    import os

    root, _ = os.path.splitext(path)
    written = []
    fig.savefig(f"{root}.png")
    written.append(f"{root}.png")
    fig.savefig(f"{root}.pdf")
    written.append(f"{root}.pdf")
    return written
