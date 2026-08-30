# 无人矿卡端到端自动泊车系统

从纯 Python 仿真、CARLA 高保真仿真到履带式钻机实物平台逐步迁移的低速自动泊车系统。当前理论车型是以两履带几何中心为旋转/轨迹中心的 6×3 m 居中矩形，允许停车原地转向；外廓、速度、角速度、安全余量和 Hybrid A* 搜索参数均由 JSON 配置。三个阶段共用同一套输入输出接口：原始传感器数据（LiDAR 点云 / Camera 图像）经 Sensor2BEV 转为统一 BEV 表示，端到端网络输出未来轨迹点，MPC 控制器完成轨迹跟踪。

## 环境要求

- 本项目的开发、测试、运行、装包一律使用 conda 虚拟环境 `endtoend-parking`（Python 3.12，位于 `D:\conda\envs\endtoend-parking`），禁止使用本地环境 `C:\Python314`。
- numpy、PyTorch（CPU 版即可）；配置化训练额外使用 PyYAML 与 matplotlib
- 测试使用标准库 `unittest`，无需额外安装

## 安装

```powershell
conda activate endtoend-parking
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-training.txt
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

```powershell
# 生成小批量训练样本（默认 5 条，可传数量参数）
python scripts/generate_dataset.py 5

# 预览 3000 条任务分层计划（不生成 BEV/轨迹）
python scripts/build_dataset.py --count 3000 --dry-run

# 旧 v4/v5 校准和未完成数据保留作旧几何证据；v5 全能力校准已完成 117/117 case，
# 完整证据保留在下列目录（v5 之后进入 v6 几何与对齐采样阶段）
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/calibrate_dataset.py --samples-per-cell 3 --max-retries 2 --task-budget-s 30 --probe-full-capability --vehicle-config configs/vehicles/tracked_drill_rig.json --output runs/dataset-calibration/tracked_pivot_v5_full

# 校准中断后使用完全相同的命令和输出目录恢复；已完成 case 会按 identity 校验后跳过
# report.json、cells.csv 和 run_state.json 可用于一次性判断成功率、超时与剩余量

# tracked_pivot_v7：继承 v6 场景复原，并让普通垂直/斜列车位的目标航向随前进/倒车入位翻转；
# 结构化入口检查整段车辆外廓走廊，相邻占用只选择入口未被堵住的目标，T4 先规划起点参考目标。
# 408 例全准入复核（34 单元×12、关闭重采）为 408/408；v6 检查点不得续入 v7。
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/build_dataset.py --count 3000 --seed 20260824 --vehicle-config configs/vehicles/tracked_drill_rig.json --output data/task_dataset/tracked_pivot_v7_3000 --batch-size 5 --max-retries 10

# 构建中断后使用完全相同的参数重启；已完成检查点会复核后跳过
# 新验收图以“前方 x、车体左方 y”为局部坐标，正 Left 显示在画面左侧；原地旋转标注 body LEFT/RIGHT 与真实有符号旋转弧
foreach ($split in 'train', 'val', 'test') {
  & 'D:\conda\envs\endtoend-parking\python.exe' scripts/inspect_dataset.py "data/task_dataset/tracked_pivot_v7_3000/$split.npz" --output "data/task_dataset/tracked_pivot_v7_3000/inspection/$split" --require-maneuver-consistency --require-trajectory-feasibility
}

# 旧全量数据已清理；仅保留 7 条 tracked_pivot_v3 真实样本用于历史可视化参考，不得用于训练或当作全量备份
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/inspect_dataset.py data/task_dataset/visual_reference_archive_v3/reference_samples.npz --output runs/visual-reference-v3-review
```

```powershell
# 生成数据并训练 MineParkingNet（输出 data_training.npz 与 mineparkingnet.pt）
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train.py --samples 40 --epochs 30

# v7 定型训练流程 smoke（全量数据、1 epoch，输出到 runs/training/v7-flow-v3/net-v1-smoke）
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train_model.py --config configs/training/net-v1-smoke.yaml

# smoke 通过后正式训练 net-v1；每 epoch 更新 loss、自由滚动 ADE/FDE、停止率和长度误差
# 中断恢复时只可使用同一训练语义生成的 checkpoint：
# resume_from: ../../runs/training/v7-flow-v3/net-v1/last.pt
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train_model.py --config configs/training/net-v1.yaml

# 下一阶段：从当前策略闭环访问状态采集专家恢复标签；按源任务原子检查点，可直接重复原命令续建
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/build_recovery_dataset.py --data data/task_dataset/tracked_pivot_v7_3000/train.npz --model runs/training/v7-flow-v3/net-v1/deployment.pt --output data/task_dataset/tracked_pivot_v8_recovery --samples 240

# 使用“原专家 + 闭环恢复”合并集训练；启用完整车体连续扫掠碰撞损失，输出写入新目录
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/train_model.py --config configs/training/net-v1-safe.yaml

# 中断恢复优先使用 last.pt；部署和开环分析使用自动校准后的 deployment.pt
# 完整参数含义、验收门槛和人工单变量调优方法见 docs/training_guide.md

# 用同一验证集比较一个或多个 Trainer checkpoint，输出 report.json 与 PNG/PDF 对比图
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/eval_openloop.py --data data/task_dataset/tracked_pivot_v7_3000/val.npz --checkpoint v0=runs/training/v7/net-v0/best.pt --checkpoint v1=runs/training/v7/net-v1/best.pt --checkpoint v2=runs/training/v7/net-v2/best.pt --output runs/openloop-eval/v7

# 对单个 checkpoint 按场景/任务/方向/噪声/相邻占用分组，并输出最差预测—专家叠加图
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/analyze_predictions.py --data data/task_dataset/tracked_pivot_v7_3000/val.npz --checkpoint runs/training/v7/net-v1/best.pt --output runs/openloop-analysis/v7/net-v1/val
```

```bash
# 闭环泊车演示（滚动闭环引擎）：专家轨迹+MPC 基线（M1 地基，成功率验收）
python scripts/run_closed_loop.py --source expert --samples 10

# 端到端主线：按 schema v2 元数据恢复原场景/占用/噪声/车位阈值，deployment→MPC 滚动闭环
# 小规模分层检查；320 点模型默认每 10 个控制周期重规划，报告含整体/分组/逐回合指标
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/val.npz --model runs/training/v7-flow-v3/net-v1/deployment.pt --samples 34 --output runs/closed-loop/v7-flow-v3/net-v1/val-stratified-34-k10/report.json

# 全量评测使用 --samples 0；T5 当前只按静态场景闭环，不代表动态障碍验证
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/test.npz --model runs/training/v7-flow-v3/net-v1/deployment.pt --samples 0 --output runs/closed-loop/v7-flow-v3/net-v1/test-full-k10/report.json

# 安全训练完成后分别保留纯网络和安全门禁两种口径；后者审查完整矩形扫掠并在不安全时专家重规划
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/val.npz --model runs/training/v8-safety-v1/net-v1/deployment.pt --samples 34 --safety-mode none --output runs/closed-loop/v8-safety-v1/net-v1/val-pure/report.json
python scripts/run_closed_loop.py --source network --data data/task_dataset/tracked_pivot_v7_3000/val.npz --model runs/training/v8-safety-v1/net-v1/deployment.pt --samples 34 --safety-mode expert_fallback --output runs/closed-loop/v8-safety-v1/net-v1/val-shield/report.json
```

## 目录结构

```
interfaces/    统一接口定义（传感器帧、BEV、车辆状态、轨迹、控制指令）
sensor2bev/    Sensor2BEV 环境表示模块（LiDAR/Camera → BEV）
sim/           Python 仿真环境（二维矿区、车辆运动模型、模拟传感器）
model/         MineParkingNet v0/v1/v2、变长轨迹输出与模型注册表（PyTorch）
training/      安全 YAML 配置、数据准备、完整车体扫掠损失、训练/验证、early stopping 与原子 checkpoint
controller/    MPC 轨迹跟踪控制器（CEM 滚动时域优化）
planner/       履带 Hybrid A* 专家轨迹（前后弧线、原地转向、RS/履带解析终连、完整扫掠碰撞）
dataset/       全能力校准、Task 分层/划分/重采、闭环恢复重标注、专家轨迹与融合 BEV、schema v2 统计/验收图
configs/       可编辑理论钻机与实验 JSON 配置
runtime/       滚动闭环引擎（轨迹源→安全门禁/专家回退→MPC→车辆，终止判定与失败分类）
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

默认钻机配置位于 `configs/vehicles/tracked_drill_rig.json`。它是理论等比模型，不代表某台实车标定；后续可直接修改外廓、底盘执行上限、规划速度/角速度、碰撞余量、搜索分辨率、原地旋转开关与代价。新正式数据会记录完整配置和模型版本，参数变化后必须重新生成或重新审计，旧数据不会自动视为兼容。

## 开发方法

本项目使用自适应 Agent 开发工作流（见 `AGENTS.md` 与 `.agents/`）。
