# 改造实施计划（对照 docs/REQUIREMENTS.md v2）

> 本文档是 M1–M4 改造的实施对照基准：每个工作包完成后在状态列打勾并附验证证据。
> 方案确立于 2026-08-23，由代码级差距分析产出。

## 模块级差距总览

| 现有模块 | 现状 | 缺口（对应 FR） | 改造动作 |
|---|---|---|---|
| `sim/environment.py` | 轴对齐矩形、0.05m 步进 raycast | FR-SIM-01 多边形、悬崖语义 | 重构：障碍抽象 + 解析求交 |
| `sim/sensor_sim.py` | 无噪声 | FR-SIM-03 | 扩展：注入噪声模型 |
| `sim/vehicle_model.py` | 固定 4×2 隐含 | FR-SIM-02 | 扩展：VehicleConfig 注入 |
| （无） | — | FR-SIM-04/05 场景库、FR-TASK-01 任务层 | 新建：spots/scenes/tasks |
| `planner/hybrid_astar.py` | 无 RS 扩展、无平滑 | FR-PLAN-01/02 | 扩展 + 新建平滑/速度剖面 |
| `dataset/generator.py` | 随机位姿对、无元数据 | FR-DATA-01/02/03 | 重构：任务驱动分层 + schema v2 |
| `model/network.py` | 仅 v0 | FR-NET-01..04 | 扩展：注册表 + v1/v2 |
| `controller/mpc.py` | 数值梯度下降（慢且弱） | FR-CTRL-01/02 | 重写：CEM 求解器 |
| `scripts/run_closed_loop.py` | 网络未进闭环、开环执行 | FR-LOOP-01/02/03 | 重写：滚动闭环引擎 |
| （无） | — | FR-METRIC、FR-VIZ、FR-SELECT | 新建：metrics/、viz/、runtime/、experiments/ |

## 改造总原则

1. 契约不动：`interfaces/` 六类签名零变更；模块单向依赖不变量保持。
2. 三个破坏性重构点集中在 M1/M2（MPC 求解器、环境障碍、闭环脚本），之后全部是增量扩展。
3. 每个工作包出口：新单测 + 既有测试保持绿色（回归门禁）。
4. 每个里程碑落一份 Spec（M1: LOOP/CTRL、M2: SCENE/TASK、M3: NET、M4: EXP）。

## M1 地基修复（约 1 周）

- [x] P1.1 MPC 重写（FR-CTRL-01/02）：CEM 求解器替换数值梯度下降；终态项强惩罚；dt 对齐校验；终点渐进减速。出口：S 形倒车跟踪、扰动恢复单测通过。（2026-08-23 完成：8 项单测通过；另发现并修复进度锚定缺陷——参考窗口从车辆当前位置改为轨迹时间轴单调推进）
- [x] P1.2 滚动闭环引擎（FR-LOOP-01/02/03）：新建 runtime/（engine/sources/termination/recorder）；NetworkSource/ExpertSource；失败自动分类；inject_obstacle 待 M2 任务层接入。出口：专家+MPC 200 实例 ≥95%。（2026-08-23 完成：10/10 成功 100%，实测平均位置误差 0.21m/航向 8.1°/跟踪 RMS 0.04m；大规模 200 实例验收待 P1.4 车辆参数化后统一跑）
- [x] P1.3 指标模块（FR-METRIC-01）：metrics/evaluation.py 八项指标；experiments runner 骨架。（2026-08-23 完成：EpisodeResult+summarize 接入引擎；experiments/run_experiment.py 配置驱动 runner + ground_baseline 配置与结果落盘）
- [x] P1.4 可视化骨架 + 车辆参数化提前：matplotlib 安装；viz/world_render、traj_render 初版；sim/vehicle_config.py 统一 6×3m。（2026-08-23 完成：viz/ 统一风格+世界/轨迹渲染+回合总图（PNG+PDF）；VehicleConfig 预设注入规划器/MPC/模型/碰撞；**200 回合地基验收 98.5% ≥ 95%，M1 出口判据达成**）

**M1 里程碑（2026-08-23）达成**：专家轨迹+CEM-MPC 闭环 200 回合成功率 98.5%（位置 0.24±0.71m / 航向 7.7° / 跟踪 RMS 0.037m），全部指标自动统计，配置驱动批量实验与可视化骨架就绪。附带修复：规划器 C-space 膨胀裕度（碰撞率 11%→1.5%）、拼接段加密校验；已知限制（格点可达性空洞）移交 P2.7 Reeds-Shepp。

## M2 平台与数据（约 1.5~2 周）

- [x] P2.1 障碍体系重构（FR-SIM-01）：Obstacle ABC + Polygon/Circle/Rectangle；kind 语义（wall/berm/cliff/rock/vehicle/equipment）；解析线段求交 raycast；悬崖"碰撞但不挡射线"。出口：新旧 raycast 一致性回归。（2026-08-24 完成：`sim/obstacles.py` 三形状 + 三语义属性；`environment.py` 解析 raycast 与旧步进一致性 ≤0.05m；LiDAR 一帧 56→1.2ms、NetworkSource 推理 158→5.4ms 达 NFR-05；新增 SIM Spec 3 规范 ID，86 项测试全过）
- [x] P2.2 场景库（FR-SIM-04/05）：spots.py、scenes.py 注册表 S1–S9、scenes_validate.py、experiments/scenes/*.yaml。出口：V1 九场景图。（2026-08-24 完成：`sim/spots.py` ParkingSpot（容差/占用/编号）+ `sim/scenes.py` 九场景注册表（S1 驻地/S2 斜列/S3 维修/S4 排土场卸载/S5 破碎站/S6 装载/S7 加油/S8 称重/S9 综合矿场）+ `sim/scenes_validate.py` 自检（捕获并修复 3 处车位-禁区几何冲突）；V1 渲染 20 文件（3×3 总览+九单图，PNG+PDF）；新增 SCENE Spec 4 规范 ID，95 项测试全过；YAML 落盘暂缓——构造参数即配置，M4 实验矩阵按需补）
- [x] P2.3 车辆参数化（FR-SIM-02）：已在 M1 提前完成。（2026-08-24 状态复核：`VehicleConfig` 统一 6×3m 矿卡与 4×2m 回归预设，规划器/MPC/车辆模型/碰撞 kwargs 统一注入；既有 VehicleConfig 回归与全仓门禁持续通过）
- [x] P2.4 噪声模型（FR-SIM-03）：sim/noise.py 三档。（2026-08-24 完成：clean/low/high 与自定义 `NoiseProfile`；LiDAR 距离高斯噪声、随机丢点、逐帧量程抖动；Camera 像素高斯噪声、目标漏检/误检；传感器私有 RNG 与 seed 复现，默认 clean 逐元素兼容；14 项噪声测试、受影响链路 53 项与全仓 138 项通过）
- [x] P2.5 BEV 参数化（FR-SIM-06）：extent 随场景配置；npz schema v2（bev_meta + task_meta）。（2026-08-24 完成：新增 frozen `BEVConfig`，默认 40×40m@0.25m；S1–S8 使用默认配置，S9 使用 80×80m@0.5m，均保持 160×160 栅格；LiDAR/Camera 共享配置并由管道提前校验；schema v2 以无 pickle 的 Unicode JSON 持久化 BEV/逐样本任务元数据，保留无版本 v1 读取兼容；全仓 149 项与末次定向 38 项通过）
- [x] P2.6 任务层（FR-TASK-01）：Task + TaskSampler 矩阵采样。（2026-08-24 完成：`Task`/`TaskGoal`/五类任务与正交难度元数据；`SeedSequence` 按根 seed×场景×类型×样本索引稳定派生；45 单元能力矩阵显式区分 40 个可采样单元与 5 个默认几何不支持单元，严格/非严格模式不静默降级；T4 保留 3–6 候选不预选，T5 输出进度触发障碍载荷；1000 任务压测零采样失败，T1–T5 代表规划集成与全仓 124 项测试通过）
- [x] P2.7 专家系统增强（FR-PLAN-01/02）：reeds_shepp.py（48 词曲表）；smoothing.py；profile.py 梯形速度剖面。（2026-08-24 完成：12 基本公式×4 对称的 48 词族、米制弧长采样与全候选排序；Hybrid A* 车身尺度解析邻域+每 5 节点门控+逐位姿碰撞安全接入，自动回归覆盖 S3/S5 直线及转向入位；三次 Hermite 捷径保留换向点并校验碰撞/曲率；梯形剖面约束加减速、换向停车和倒车降速。111 项全仓测试通过；1000 组随机 SE(2) 候选失败 0；200 回合闭环成功率 99.0%、碰撞率 1.0%、规划失败重采 26 次、耗时 139s，较 M1 基线 98.5%/1.5%/43 次/183s 改善）
- [x] P2.8 数据管线重构（FR-DATA-01/02/03）：generate() 接收 Task 列表；splits.py；build_dataset/inspect_dataset 脚本。出口：3000+ 条分层集 + 统计图。（2026-08-24 完成：Task→专家轨迹/融合 BEV 全链路接入，T4 记录首个可规划候选策略；纯几何能力上叠加专家可生成能力，按场景×类型×可表达难度分配并在同单元重采；稳定 8:1:1 划分且 S9 仅进入 test；正式生成 schema v2 train/val/test=2400/300/300，3000 个 task ID 跨 split 全部唯一，输出 manifest、三份统计 JSON 与 12 张 BEV+专家轨迹抽检图；全仓 169 项测试通过）

## M3 网络与训练（约 1.5 周）

- [ ] P3.1 模型注册表与变体（FR-NET-01..04）：registry.py；v1 变长（GRU+终止符）；v2 主模型（U-Net+交叉注意力）。
- [ ] P3.2 训练体系（FR-NET-05）：training/trainer.py YAML 配置、early stopping、checkpoint、曲线图。
- [ ] P3.3 开环评估：eval_openloop.py + V3 对比图。出口：v1/v2 val ADE/FDE 优于 v0。

## M4 实验矩阵（约 1.5~2 周）

- [ ] P4.1 多车位选优（FR-SELECT-01..04）：spot_scorer.py 五项评分器（oracle 标签）；SelectAndParkSource（R1）；no-goal 变体（R2）。
- [ ] P4.2 基线接入（FR-BASE-01..03）：StraightLine/Blind/PureMPC sources；≥3 seed。
- [ ] P4.3 批量 runner 完整版（FR-METRIC-02..06）：全网格+断点续跑；analyze.py LaTeX/Markdown 表+t 检验。
- [ ] P4.4 可视化完成版（FR-VIZ-01..05）：animation.py（V2 双视图 GIF）；bev_render.py（V4）；experiment_plots.py（V5）；style.py。

## 目录结构演进

```
改造前                          改造后（新增 ★ 重构 ✚）
interfaces/  (不动)             sim/  ✚obstacles.py ★vehicle_config.py ★noise.py
sim/         ✚environment.py    ★spots.py ★scenes.py ★scenes_validate.py ★tasks.py
sensor2bev/  (不动)             planner/ ★reeds_shepp.py ★smoothing.py ★profile.py ✚hybrid_astar.py
planner/     ✚hybrid_astar.py   dataset/ ✚generator.py ★splits.py
dataset/     ✚generator.py      model/  ★registry.py ★v1_autoregressive.py ★v2_unet_attn.py
model/       (v0 保留)          controller/ ✚mpc.py（求解器重写，接口不变）
controller/  ✚mpc.py            ★runtime/（engine/sources/termination/recorder）
scripts/     ✚run_closed_loop.py ★training/（trainer/configs）
                                  ★metrics/ ★viz/ ★experiments/（scenes/configs/run/analyze）
```

## 关键路径

```
P1.1 MPC ──→ P1.2 引擎 ──→ P1.3 指标 ─┐
P1.4 车辆参数化（提前入 M1）───────────┤
                                      ↓
P2.1 障碍体系 ──→ P2.2 场景 ──→ P2.6 任务 ──→ P2.8 数据 ──→ M3 网络 ──→ M4 实验
        └────→ P2.7 专家增强 ──────────↑        ↑ viz/ 各阶段滚动补齐
P2.4 噪声、P2.5 BEV 参数化（独立，任意时段插入）
```

## 风险与对策

| 风险 | 对策 |
|---|---|
| CEM 调参不达 95% | 终态权重/精英比网格搜索脚本化；备选 iLQR（接口预留） |
| RS 曲线实现错误 | 48 词曲线逐类单测（已知长度解析解对照） |
| 解析 raycast 改变点云分布 | 重构前先落"旧步进 vs 新解析"一致性回归测试 |
| 160×160 BEV 使 CNN CPU 推理超 10ms | 分辨率降档作为消融项；LazyLinear 自动适配 |
| 变长解码训练不稳 | 退路：固定 horizon=60+mask（需求已含） |
| npz schema v2 破坏旧数据 | 保留 v1 读取兼容 + 版本字段 |
