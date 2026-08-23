# 无人矿卡端到端自动泊车系统

从纯 Python 仿真、CARLA 高保真仿真到履带式小车实物平台逐步迁移的无人矿卡低速自动泊车系统。三个阶段共用同一套输入输出接口：原始传感器数据（LiDAR 点云 / Camera 图像）经 Sensor2BEV 转为统一 BEV 表示，端到端网络输出未来轨迹点，MPC 控制器完成轨迹跟踪。

## 环境要求

- 本项目的开发、测试、运行、装包一律使用 conda 虚拟环境 `endtoend-parking`（Python 3.12，位于 `D:\conda\envs\endtoend-parking`），禁止使用本地环境 `C:\Python314`。
- numpy、PyTorch（CPU 版即可）
- 测试使用标准库 `unittest`，无需额外安装

## 安装

```bash
conda activate endtoend-parking
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 快速开始

```bash
# 激活环境（或直接用完整路径 D:\conda\envs\endtoend-parking\python.exe）
conda activate endtoend-parking

# 运行阶段一闭环演示：仿真环境 → LiDAR → BEV → 简单跟踪控制
python scripts/run_sim.py
```

```bash
# 运行全部单元测试
python -m unittest discover -s tests -v
```

```bash
# 生成小批量训练样本（默认 5 条，可传数量参数）
python scripts/generate_dataset.py 5
```

```bash
# 生成数据并训练 MineParkingNet（输出 data_training.npz 与 mineparkingnet.pt）
python scripts/train.py --samples 40 --epochs 30
```

```bash
# 闭环泊车演示（滚动闭环引擎）：专家轨迹+MPC 基线（M1 地基，成功率验收）
python scripts/run_closed_loop.py --source expert --samples 10

# 端到端主线：感知→BEV→网络→MPC 滚动闭环（需先训练模型与数据集）
python scripts/run_closed_loop.py --source network --data data_closed_loop.npz --samples 5
```

## 目录结构

```
interfaces/    统一接口定义（传感器帧、BEV、车辆状态、轨迹、控制指令）
sensor2bev/    Sensor2BEV 环境表示模块（LiDAR/Camera → BEV）
sim/           Python 仿真环境（二维矿区、车辆运动模型、模拟传感器）
model/         MineParkingNet 端到端轨迹生成网络（PyTorch）
controller/    MPC 轨迹跟踪控制器（CEM 滚动时域优化）
planner/       Hybrid A* 专家轨迹生成（差分驱动）
dataset/       训练样本生成（采样位姿对 + 专家轨迹 + 融合 BEV）
runtime/       滚动闭环引擎（轨迹源→MPC→车辆，终止判定与失败分类）
metrics/       回合指标定义与聚合（成功率/碰撞率/误差等）
scripts/       运行脚本
tests/         单元测试
docs/          任务日志与修改轨迹
spec/          行为规范
```

## 系统流程

```
LiDAR点云 / Camera图像
      ↓ 传感器适配器
      ↓ Sensor2BEV 环境表示模块
BEV + 车辆状态 + 目标泊车位姿
      ↓ MineParkingNet 端到端轨迹生成网络
未来N个局部轨迹点
      ↓ MPC 轨迹跟踪控制器
[v_cmd, omega_cmd]
      ↓ 平台执行器
Python车辆 / CARLA车辆 / 履带车底盘
```

详细设计见 `DESIGN.md`，模块与数据流见 `ARCHITECTURE.md`。

## 开发方法

本项目使用自适应 Agent 开发工作流（见 `AGENTS.md` 与 `.agents/`）。