"""V1 场景地图渲染：全部注册场景俯视图（论文平台图）。

用法：
    python scripts/render_scenes.py [--out out/scenes] [--show]

输出每场景 PNG+PDF（不带扩展名），命名与场景注册名一致。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.scenes import SCENE_REGISTRY, build_scene
from viz import render_world, setup_style


def render_all(out_dir: str) -> list[str]:
    setup_style()
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    # 3×3 网格总览。
    names = sorted(n for n in SCENE_REGISTRY)
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    for ax, name in zip(axes.flat, names):
        bundle = build_scene(name)
        render_world(ax, bundle.env, spots=bundle.spots)
        ax.set_title(f"{name}", fontsize=10)
        ax.tick_params(labelsize=7)
    fig.suptitle("Mine Parking Scene Library (S1-S9)", fontsize=14)
    fig.tight_layout()
    from viz.style import save_fig

    written += save_fig(fig, os.path.join(out_dir, "all_scenes"))
    plt.close(fig)

    # 单场景大图。
    for name in names:
        bundle = build_scene(name)
        fig, ax = plt.subplots(figsize=(8, 8))
        render_world(ax, bundle.env, spots=bundle.spots)
        ax.set_title(bundle.title_en or name)
        written += save_fig(fig, os.path.join(out_dir, name))
        plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="out/scenes")
    args = parser.parse_args()
    written = render_all(args.out)
    print(f"已渲染 {len(written)} 个文件到 {args.out}/")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
