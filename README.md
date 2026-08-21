# 无人矿卡端到端自动泊车系统

从纯 Python 仿真、CARLA 高保真仿真到履带式小车实物平台逐步迁移的无人矿卡低速自动泊车系统。三个阶段共用同一套输入输出接口：原始传感器数据（LiDAR 点云 / Camera 图像）经 Sensor2BEV 转为统一 BEV 表示，端到端网络输出未来轨迹点，MPC 控制器完成轨迹跟踪。

## 环境要求

- Python 3.10+（当前验证环境为 Python 3.14 / numpy 2.4.5）
- numpy（当前阶段唯一依赖）
- 测试使用标准库 `unittest`，无需额外安装

## 安装

```bash
pip install numpy
```

## 快速开始

```bash
# 运行阶段一闭环演示：仿真环境 → LiDAR → BEV → 简单跟踪控制
python scripts/run_sim.py
```

```bash
# 运行全部单元测试
python -m unittest discover -s tests -v
```

## 目录结构

```
interfaces/    统一接口定义（传感器帧、BEV、车辆状态、轨迹、控制指令）
sensor2bev/    Sensor2BEV 环境表示模块（LiDAR/Camera → BEV）
sim/           Python 仿真环境（二维矿区、车辆运动模型、模拟传感器）
model/         MineParkingNet 端到端轨迹生成网络（骨架）
controller/    MPC 轨迹跟踪控制器（骨架）
planner/       Hybrid A* 等专家轨迹生成（骨架）
dataset/       训练数据集生成与加载（骨架）
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