# 架构

## 组件与所有权

| 模块 | 目录 | 职责 |
|---|---|---|
| 统一接口 | `interfaces/` | 定义三阶段共用的数据契约：传感器帧、BEV 空间配置/张量、车辆状态、目标位姿、轨迹、控制指令；坐标统一为车辆中心局部坐标系 |
| Sensor2BEV | `sensor2bev/` | 将 LiDAR 点云/Camera 图像按共享 `BEVConfig` 转换为统一 BEV 表示：`lidar_bev.py`（ROI→降采样→地面滤除→栅格投影）、`camera_bev.py`（IPM 单应反投影目标区域）、`fusion.py`（通道级后融合） |
| Python 仿真与任务 | `sim/` | 二维矿区泊车环境、携带场景级 `BEVConfig` 的 S1–S9 场景与 T1–T5 任务层；`noise.py` 提供传感器噪声；`vehicle_config.py` 严格加载 `configs/vehicles/`，是履带钻机外廓、执行上限和规划搜索参数的统一来源 |
| 端到端网络 | `model/` | 注册表按 `net-v0/v1/v2` 构造模型：v0 为固定 horizon CNN+MLP，v1 为 CNN+GRU 变长解码和终止 logits，v2 增加 U-Net 跳连空间编码与 goal/state 交叉注意力；损失支持掩码轨迹 MSE 与终止 BCE |
| 训练体系 | `training/` | 安全 YAML 配置解析并相对配置文件定位数据/输出；`Trainer` 拥有确定性 shuffle、计划采样、累计停止监督、完整车体连续扫掠损失、课程感知 early stopping 和逐 epoch 自由滚动/碰撞诊断；runner 从 schema v2 解析车辆/BEV 安全几何，并在 best 上用 val 校准停止阈值 |
| 轨迹控制器 | `controller/` | MPC 轨迹跟踪：CEM 交叉熵求解 + 差分驱动模型预测，输出 `[v_cmd, omega_cmd]` |
| 专家轨迹 | `planner/` | Hybrid A* 生成履带低速运动学可行轨迹（前后差速弧线 + 左右原地旋转 + 48 词族 Reeds–Shepp/履带解析候选）；`collision.py` 拥有完整矩形与连续扫掠碰撞，`smoothing.py`/`profile.py` 提供可选平滑以及含原地旋转耗时的速度剖面 |
| 数据管线 | `dataset/` | `calibration.py` 直接枚举全部专家能力单元；Task 驱动生成经机动与可行性双门禁保存 schema v2；`recovery.py` 从学习器闭环偏离状态生成重新审计的专家恢复样本，碰撞时完整回溯最近的规划安全余量状态；构建脚本按源任务原子续建、从失败检查点派生困难补采并去重合并既有恢复集和原训练集 |
| 闭环运行时 | `runtime/` | `engine.py` 执行轨迹源→MPC→车辆并以完整矩形连续扫掠判碰撞；`sources.py` 提供 Expert/Network、当前状态专家重规划和安全门禁组合；`safety.py` 定义场景无关的轨迹审查接口与干预统计 |
| 实验指标 | `metrics/` | `EpisodeResult` 与闭环聚合；开环层在目标有效前缀上统计 ADE/FDE/环绕航向 MAE，并拒绝预测 horizon 不足的比较；预测诊断层保留逐样本误差与终止长度，按场景、任务、方向、噪声和相邻占用聚合 |
| 可视化 | `viz/`、`dataset/inspection.py` | 统一风格（`style.py` 色表/PNG+PDF 双格式）、世界俯视渲染、轨迹三线叠加、单回合总图与分组开环图；专家验收图和预测叠加图使用“前方 x、车体左方 y”右手局部系，将正 Left 显示在画面左侧，并把连续零位移航向变化汇总为从旋转前航向出发的有符号旋转弧 |
| 批量实验 | `experiments/` | 配置驱动专家 runner；`closed_loop_evaluation.py` 从 schema v2 NPZ/manifest 确定性复原任务场景并加载 Trainer deployment，输出网络闭环整体、分组与逐回合 JSON |
| 运行脚本 | `scripts/` | 阶段演示与数据流串联 |

## 数据流

```
ParkingEnvironment（真值）→ SimulatedLiDAR/Camera + NoiseProfile → LiDARFrame/CameraFrame
SceneBundle.bev_config → LiDAR2BEV/Camera2BEV → SensorBEVPipeline 配置一致性门
LiDARFrame/CameraFrame → sensor2bev(BEVConfig) → BEVTensor
  LiDARFrame → LiDAR2BEV → [occupancy, height, density]
  CameraFrame → Camera2BEV → [target]
  BEVFusion 拼接两路并追加 [vehicle] → 统一 BEVTensor
VehicleConfig 尺寸/margin → TaskSampler → 车辆相对 S3/S4/S7/S9 SceneBundle
SceneBundle + (TaskType, difficulty axes, seed, sample index) → TaskSampler → Task
  Task = scene + start + single/candidate goals + stable metadata + optional T5 event
  结构化车位按请求机动确定目标航向与入口侧，并只从连续 footprint 自由的轴线走廊采样起点
VehicleConfig JSON → HybridAStarPlanner/MPC/车辆模型/碰撞/inspection
VehicleState + GoalPose → HybridAStarPlanner（前后弧线 + 原地旋转 + RS/履带解析候选）→ Trajectory（中心位姿专家轨迹）
  Trajectory → 碰撞安全三次捷径（可选）→ Trajectory
  Trajectory → 梯形速度剖面（可选）→ VelocityProfile
Task[] → DatasetSplits（S9→test；其余按场景×类型分层 train/val）
专家能力矩阵 → 等额 CalibrationCase[]（不经过 split 配额）
  → 独立 worker 硬预算 → case 终态原子 JSON → 可恢复 report.json/cells.csv
Task → task 组件工厂 → HybridAStarPlanner + SensorBEVPipeline → TrainingSample
  轨迹方向距离审计：请求方向占比默认 ≥50%；不一致候选拒绝，失败在原场景×类型×难度单元重采
  轨迹可行性审计：起终位姿、线/角速度、横向残差、完整矩形扫掠、模型版本全部通过
  T4 按稳定候选顺序解析首个可规划且机动一致目标；显式关闭门禁时保留旧策略
  TrainingSample[] → NPZ schema v2（数组 + bev_meta + 逐样本车辆模型/机动/可行性证据）；无版本 NPZ 按 v1 读取
  固定批次 → 失败前原子 retry 游标 → 独立机动与运动学复算 + 碰撞证据/模型版本核对 → 原子成功检查点
  同身份检查点（seed/计划/车辆模型/重试参数/批大小）+ 未完成批次游标 → 不重放已排除 task ID 地补齐合并 → 正式 NPZ/manifest
训练 YAML → SafeLoader/严格 schema → 注册表模型 + train/val NPZ → `Trainer`
  → 样本级确定性 shuffle + 累计停止 BCE + teacher-forcing 线性退火
  → schema v2 车辆/BEV 几何 → 预测轨迹完整矩形与连续扫掠 occupancy/越界损失
  → 每 epoch train/val 自由滚动 ADE/FDE/停止诊断 + 课程预热后 early stopping + 原子 history/best/last
  → best checkpoint 的 val 停止阈值扫描 → 只读 deployment checkpoint + 校准 JSON
  → report.json + training_curve PNG/PDF
同一 val NPZ + 一个或多个 Trainer checkpoint → `eval_openloop.py`
  → ADE/FDE/航向 MAE report.json + openloop_comparison PNG/PDF
  NPZ → 长度/方向/可行性/分层统计 + occupancy/target/中心点/中间车身验收图
BEVTensor + VehicleState + GoalPose → 模型注册表（net-v0/v1/v2）→ Trajectory
  （训练：全局专家轨迹/目标 → 起始局部系 → Trainer → 掩码轨迹/终止损失 → 原子 checkpoint）
Trajectory + VehicleState → MPCController → ControlCmd[v, omega] → 平台执行器
ClosedLoopEngine：TrajectorySource(Expert/Network) → MPC → 车辆模型滚动循环
  → 可选 SafetyShieldSource（完整扫掠审查 → 当前状态专家回退）
  → 终止（到达双阈值/完整矩形连续扫掠碰撞/超时/振荡）→ EpisodeResult（含干预统计）
schema v2 NPZ + manifest + deployment checkpoint → 闭环评测编排
  → 复原 scene/occupancy/noise/BEV/selected goal → NetworkSource（目标通道随回合更新）→ 分组 JSON
  → 学习器闭环状态（固定步长偏离；碰撞时完整回溯最近的规划安全余量状态）→ 专家重规划与重新审计
  → 上轮碰撞/超时且零恢复检查点 → 通用困难任务补采 → recovery 去重合并 + 原 train 合并
```

- 障碍物碰撞与点云语义分离：`is_free`/`has_collision` 只检查 forbidden 障碍与地图边界；`raycast` 只与 emits_points 障碍求交（悬崖禁止进入但不挡射线，地面标线可通行）。
- Camera→BEV 与模拟相机共用同一套单应几何（`sim/camera_model.py` 与 `sensor2bev/camera_bev.py` 推导一致），保证渲染与反投影互逆。
- MPC 进度锚定轨迹时间轴（单调推进），参考轨迹 dt 与控制周期不一致时按时间插值对齐；求解器为 CEM（纯 numpy）。

## 统一接口（interfaces/）

- 传感器帧：`LiDARFrame`（N×4 `[x,y,z,intensity]` 点云）、`CameraFrame`（图像与标定参数）。
- 环境表示：`BEVConfig`（分辨率、前后左右范围与确定栅格形状）和 `BEVTensor`（`[C,H,W]`，通道语义：障碍物占据、高度、点云密度、目标区域、车辆轮廓）。
- 运动状态：`VehicleState`（`[x,y,yaw,v,omega]`）。
- 目标位姿：`GoalPose`（`[x_goal,y_goal,yaw_goal]`）。
- 轨迹：`Trajectory`（未来 N 个局部轨迹点 `[x_i,y_i,yaw_i]`）。
- 控制指令：`ControlCmd`（`[v_cmd, omega_cmd]`）。

## 不变量

- 网络输入输出契约在三个阶段保持不变；迁移只替换传感器来源和底盘执行接口。
- 坐标统一使用车辆中心局部坐标系。
- 专家验收图的数值局部系为前方 `x`、车体左方 `y`、正 `yaw` 为车体左转；画面水平轴反向显示数值 `y`，使数值正 Left 在画面上仍为左，不得直接按屏幕右正方向绘制。
- `interfaces/` 不依赖任何仿真、网络或硬件实现，反向依赖。
- `sim/`、`sensor2bev/`、`model/`、`controller/` 依赖 `interfaces/` 与各自必要的 `numpy`/`torch` 数值运行时，模块间不互相耦合；`training/` 依赖 `model/` 与 PyTorch，不反向进入模型或数据层。
- `runtime/` 依赖 `interfaces/`、`metrics/` 与注入的轨迹源/MPC/车辆模型（依赖注入，不直接 import sim/model）；`metrics/` 无内部依赖。
- 闭环执行统一经 `runtime/ClosedLoopEngine`，轨迹源策略可替换（Expert/Network/后续基线），指标口径唯一。
- 数据集网络闭环由 `experiments/closed_loop_evaluation.py` 编排，必须在回合开始前核对 manifest 车辆模型、任务确定性身份、selected goal、BEV 配置与 checkpoint 通道/dt；任何漂移都拒绝运行，不得回退到通用环境。T5 元数据当前不触发动态障碍注入，报告显式标为静态闭环。
- 默认理论车型由 `configs/vehicles/tracked_drill_rig.json` 定义为以两履带几何中心居中的 6×3 m 矩形；`VehicleConfig` 将外廓、执行上限、规划速度/角速度、安全余量、搜索分辨率与解析接管范围统一注入规划器/MPC/车辆模型/碰撞/inspection。`tracked_pivot_v5` 的解析接管范围为 T3 距离上限 30m。
- S4/S9 卸载区由 `TaskSampler` 将同一 `VehicleConfig` 的长、宽和 `collision_margin` 注入场景：双向主路至少 3.5 倍车宽、卸载位中心距至少 3 倍车宽，车尾到挡墙的物理净空为 `collision_margin + 0.3m`；`geometry_profile` 进入任务元数据和计划指纹，旧几何检查点不得续入。
- 批量实验统一经 `experiments/run_experiment.py`（JSON 配置驱动，结果落盘 `experiments/results/`），可视化统一经 `viz/`（PNG+PDF 双格式输出）。
- 低速泊车采用理想履带差速运动学，控制量为线速度 v 与角速度 omega；允许 `v=0, omega≠0`，原地旋转中心固定为两履带几何中心。履带滑移、沉陷、质量和惯量不属于当前理论模型。
- 平移弧线的 Reeds–Shepp 半径由 `|plan_v / plan_max_omega|` 推导，但它不代表履带全部可达集合；Hybrid A* 另有原地旋转基元和履带解析候选。默认/前进请求下倒车或反方向平移、以及原地旋转相对常规前进/小转向按 2:1 时间代价计；显式倒车监督任务仍以倒车为请求方向。原地旋转对矩形最远角点扫掠按配置分辨率加密。
- `_pose_free` 检查完整定向矩形与圆/矩形/多边形障碍相交，不再只验四角；任一运动基元、解析候选和数据可行性审计都复用相同外廓与 `collision_margin`。
- 平滑与速度剖面位于 `planner/` 内且为可选后处理，不改变三阶段共用的 `interfaces/Trajectory(points, dt)` 契约。
- `sim/tasks.py` 不依赖规划器：9×5 能力矩阵显式保留不支持单元，支持单元只保证任务几何契约；规划失败的重采样由后续数据/实验编排层负责。
- `TaskSampler` 将 `perpendicular_bay`/`diagonal_bay` 的基准航向解释为前进入位航向，倒车任务翻转 180° 使车头朝入口；`_sample_axial_start` 对普通结构化车位和紧 bay（maintenance_bay/crusher_slot/fuel_bay）从与请求机动一致的入口侧采样，并用 0.1m 步长确认目标到起点的完整车辆外廓走廊连续无碰撞。相邻占用会先过滤堵住入口走廊的目标；T4 仍携带 3–6 个候选，但把用于构造起点的参考目标放在首位，避免先在不匹配候选上耗尽规划预算。
- `dataset/build.py::expert_maneuvers` 拥有专家可生成能力：不改变任务几何矩阵，根据与车辆模型身份绑定的完整校准证据排除持续不可达/不稳定监督单元并限制单向可达单元；正式构建与后续校准共享该入口，偶发规划失败仍保持原单元难度重采。对齐采样后 S3 全单元与 S5/T1/T2/T5、S9/T2/T5 限制为倒车，S8/T2/T3/T5 限制为前进。
- `dataset/calibration.py` 的 case 计划不复用 8:1:1 数据集配额：默认枚举当前专家准入单元，新版本重校准可显式探测全部几何支持单元及双向机动。身份绑定 seed、车辆模型、探测模式、case 计划、重试和预算；只有原子终态检查点算完成，中断恢复跳过完成项并重做当前未落盘 case。
- `model/registry.py` 是模型名称到实现的唯一构造入口；`MineParkingNet` 保持 `net-v0` 兼容，v1/v2 的终止 logits 不改变闭环最终消费的 `Trajectory` 契约。
- `training/Trainer` 的 checkpoint 必须核对模型名称、模型配置和影响恢复语义的训练超参数，包括 shuffle、停止平衡和 teacher-forcing 调度；允许改变总 epochs、设备和输出目录，但不得静默加载不兼容状态。
- 启用碰撞损失时，train/val 必须为车辆模型与 BEV 几何一致的 schema v2；完整车体尺寸、数据碰撞余量和 occupancy 通道只从元数据解析。安全几何、额外余量、采样间距、扫掠上限和损失权重属于 checkpoint 恢复语义。
- 闭环恢复样本只从仍无碰撞的学习器访问状态生成，位置与航向偏离都参与选择，并优先保留碰撞前最后安全状态；新轨迹必须重新计算机动方向与可行性证据，不得复制原起点审计。恢复检查点绑定输入计划、权重摘要、车辆和选择参数。
- 运行时安全门禁与纯网络源正交组合：主轨迹不通过完整矩形/连续扫掠检查时才从当前状态调用专家回退；报告必须区分 `none` 与 `expert_fallback` 并记录干预率。闭环实际碰撞使用无安全余量的完整外廓，门禁使用车辆配置安全余量。
- `best.pt`/`last.pt` 保留训练恢复语义；`deployment.pt` 写入只在 val 上选择的停止阈值并显式标记不可恢复训练，下游推理不得退回未经校准的默认阈值，也不得使用 test 选择阈值。
- 相邻占用能力按场景×任务类型枚举实际可表达的 0/1/2 等级；T4 在占用后仍必须保留至少 3 个空闲候选位。
- 任务随机流由根 seed、场景稳定序号、任务类型稳定序号和样本索引派生；T4 不预选目标，T5 只描述触发与载荷且不在采样时修改环境。
- 传感器噪声只改变观测帧，不改变环境真值；默认 `clean` 与原输出兼容，非干净 profile 使用各传感器私有 seed/RNG，不读写 NumPy 全局随机状态。
- BEV 以车辆为中心生成；默认 40×40m@0.25m，场景可覆盖范围/分辨率但同一融合管道两路必须共用配置；S9 用 80×80m@0.5m，并与默认配置保持 160×160 栅格。
- 数据集 schema v2 将 `BEVTensor.to_metadata()` 与逐样本任务元数据编码为 Unicode JSON 数组，加载始终禁用 pickle；同一归档不允许混合 BEV 元数据或轨迹 `dt`，无版本旧归档识别为 v1。
- 默认数据构建按 8:1:1 输出 train/val/test，S9 全部且仅进入 test；构建失败只在同一场景×任务类型×难度重采，替代任务避开全计划、已用与 retry 状态排除的 task ID；未完成批次每次失败前原子持久化下一重采游标，成功 NPZ 与批次报告都落盘后才清除；未完成 split 保持 `.partial.npz`，三个 split 完成后才写 manifest。
- `dataset/maneuver.py` 是轨迹实际方向距离的唯一判定所有者；Task 请求方向默认必须占总行驶距离至少 50%，允许短距离反向调整。Task 生成先门禁候选，构建脚本再独立审计 partial；任一不一致、无效或缺失声明样本都不得晋升为正式数据集。
- `dataset/feasibility.py` 是归档运动学与模型版本审计所有者；生成期用规划器环境复核完整扫掠，partial 晋升前从数组独立复算并核对碰撞证据。旧 NPZ 或配置版本不匹配样本不得通过严格门禁。

## 阶段迁移边界

- 阶段一→二：替换传感器模拟为 CARLA 传感器，其余模块不动。
- 阶段二→三：替换 CARLA 车辆为履带底盘（新增 CAN 执行适配），其余模块不动。
