"""YAML 驱动的 MineParkingNet 正式训练入口。"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from training.runner import run_training_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按安全 YAML 配置训练 MineParkingNet，并输出 checkpoint、曲线和开环报告"
    )
    parser.add_argument("--config", required=True, help="训练 YAML 路径")
    args = parser.parse_args()
    report = run_training_from_yaml(args.config)
    metrics = report["metrics"]
    print(
        f"完成 {report['model_name']}："
        f"ADE={metrics['ade_m']:.4f}m，FDE={metrics['fde_m']:.4f}m，"
        f"yaw MAE={np.degrees(metrics['yaw_mae_rad']):.3f}deg"
    )
    print(f"报告：{os.path.join(report['trainer_config']['checkpoint_dir'], 'report.json')}")


if __name__ == "__main__":
    main()
