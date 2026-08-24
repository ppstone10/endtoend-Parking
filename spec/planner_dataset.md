# 专家轨迹与训练数据 Spec

## 元数据

- Spec ID 前缀：`EXPTRAJ`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-08-24

## 目标

- 从起始状态到目标泊车位姿生成差分驱动可行的专家轨迹，作为 MineParkingNet 训练标签。
- 使用 Reeds–Shepp 解析扩展闭合离散运动基元的格点可达性空洞，并提供经碰撞校验的轨迹平滑与倒车降速剖面。
- 批量生成训练样本（融合 BEV + 目标位姿 + 状态 + 专家轨迹），供阶段四网络训练使用。

## 非目标

- 不做轮胎侧偏、载荷转移等高阶动力学优化；速度剖面只约束纵向加减速、限速与换向停车。
- 不实现网络训练、MPC 闭环（后续阶段）。
- 不在本模块实现网络训练或 P4.1 多车位五项代价 oracle；P2.8 只为 T4 解析一个可规划监督目标并记录策略。

## 边界与约束

- 规划与数据均在 `ParkingEnvironment` 世界坐标系中进行；BEV 为车辆中心局部系。
- 差分驱动运动学：`[v, omega]`，规划参考速度 `plan_v`，最大角速度 `max_omega`。
- 碰撞检测采用车辆矩形（长×宽）四角全部位于自由空间；`collision_margin > 0` 时矩形各向外膨胀该裕度（C-space 膨胀），保证轨迹与障碍保持至少 margin 净空，吸收闭环跟踪误差。
- 近目标扩展必须是满足最小转弯半径的 Reeds–Shepp 曲线，按 `motion_resolution` 加密逐点校验车身矩形和搜索边界；禁止用位姿线性插值作为解析直连。
- 搜索需在起点-目标包围盒外加边距的范围内进行，并设探索节点上限防止发散。
- 平滑输入必须是已通过碰撞检查的专家轨迹；平滑不跨越前进/倒车换向点，三次捷径候选只有在轨迹更短、全采样位姿无碰撞且曲率不超限时才可接受。
- 速度剖面不修改几何轨迹；起终点与换向点速度为 0，倒车速度上限不高于前进上限。

## 数据与接口

- `ReedsSheppPath`：保存 48 词族中某个可行候选的转向类型、带符号米制段长、最小转弯半径和总长。段长正/负分别表示前进/倒车。
- 平滑结果：平滑入口仍接收/返回 `Trajectory`，不新增数据类，不改变三阶段统一轨迹接口。
- `VelocityProfile`：与轨迹等长的带符号速度数组和单调时间数组；位于 `planner/` 内，在任务/数据层决定落盘 schema 前不扩展 `interfaces/Trajectory`。
- `HybridAStarPlanner.plan` 的输入、返回类型和已有异常类型保持兼容。
- NPZ schema v2：保留 v1 的 `bevs/goals/states/trajs/masks/dt` 数组，新增标量 `schema_version=2`、单份 `bev_meta` 与逐样本 `task_meta`。元数据以 Unicode JSON 数组保存，加载后解码为字典/字典列表。
- Task 驱动入口：`DatasetGenerator.generate` 同时接受旧式整数数量或 `Task` 可迭代对象；跨场景生成由调用方注入 task→(planner, SensorBEVPipeline) 工厂。
- 划分结果：`DatasetSplits` 拥有互斥的 train/val/test Task；默认 test 场景为 S9，任何该场景样本不得进入 train/val。

## 行为与验收

### `EXPTRAJ-PLAN-001`：专家轨迹生成

- 前置：起始 `VehicleState` 与目标 `GoalPose` 均无碰撞。
- 行为：`HybridAStarPlanner.plan` 在状态离散空间搜索差分驱动可行路径。
- 结果：返回 `Trajectory`，终点与目标位姿偏差 ≤ 0.6m；轨迹各点均位于自由空间。
- 异常与恢复：起始/目标碰撞抛 `ValueError`；搜索发散或超出上限抛 `RuntimeError`，由调用方重试新位姿。

### `EXPTRAJ-DATA-001`：训练样本生成

- 前置：环境、规划器、传感器→BEV 管道已就绪。
- 行为：`DatasetGenerator.generate` 随机采样间距 [3, 12]m 的无碰撞位姿对，规划专家轨迹并采集融合 BEV。
- 结果：返回 `TrainingSample` 列表，每条含 5 通道融合 BEV、全局目标位姿、起始状态、专家轨迹。
- 异常与恢复：规划失败样本自动跳过并重试；重试达上限仍未凑够数量时抛 `RuntimeError`。

### `EXPTRAJ-DATA-002`：NPZ schema v2 自描述落盘

- 前置：样本列表非空；归档内所有 BEV 的分辨率、范围、通道、形状一致，所有专家轨迹 `dt` 一致；可选 `task_meta` 可 JSON 序列化。
- 行为：保存原有训练数组并写入版本 2；`bev_meta` 保存 `resolution/extent/channels/shape`，`task_meta` 与样本逐项对齐，缺省任务元数据写为空字典。
- 结果：调用方加载后可直接读取解码后的 `bev_meta` 与 `task_meta`，无需硬编码 BEV 几何和通道；元数据 JSON 使用排序键与紧凑分隔符，且不依赖 pickle。
- 异常与恢复：空样本、混合 BEV 元数据、混合 `dt` 或不可 JSON 序列化元数据抛出 `ValueError`，不生成语义含混的数据集。

### `EXPTRAJ-DATA-003`：schema v1 读取兼容

- 前置：旧归档没有 `schema_version` 字段，且原有数组字段可由 NumPy 安全加载。
- 行为：始终以 `allow_pickle=False` 加载；无版本归档识别为 v1，新格式只接受已知版本 2。
- 结果：v1 原数组字段保持可用，并附加 `schema_version=1`、`bev_meta=None`、`task_meta=None`；现有训练和闭环读取方无需迁移即可继续消费数组。
- 异常与恢复：未知显式版本或 v2 元数据缺失/JSON 损坏时抛出 `ValueError`，禁止猜测字段语义。

### `EXPTRAJ-DATA-004`：Task 驱动专家样本生成

- 前置：Task 合法；组件工厂返回与 Task 场景和 BEV 配置一致的规划器/传感器管道。
- 行为：单目标任务直接规划；T4 按稳定候选顺序选择首个可规划目标。采集 Task 起点 BEV，并把 `Task.to_metadata()`、实际噪声 profile、选中目标及解析策略写入样本元数据。
- 结果：每个成功 Task 恰生成一条 `TrainingSample`，状态等于 Task 起点、目标等于已解析目标、轨迹通过专家规划器；输入顺序与输出顺序一致。
- 异常与恢复：所有目标不可规划或组件配置不匹配时抛出携带 task ID 的 `TaskGenerationError`；构建层可据此在同一场景×任务类型单元重采。旧 `generate(int)` 行为保持兼容。
- T4 边界：`first_plannable_candidate` 不是“最优车位”标签；P4.1 可通过 goal selector 注入五项代价 oracle。

### `EXPTRAJ-DATA-005`：稳定分层与整场景泛化集

- 前置：Task ID 唯一，划分比例合法且测试保留场景存在。
- 行为：默认把 S9 全部放入 test；其余任务按场景×任务类型分层、用 seed 可复现地选择 val，剩余进入 train。
- 结果：三集合互斥且并集等于输入；S9 不进入 train/val；同一输入与 seed 返回相同 Task ID 集合。构建计划按总量预留 10% S9，使目标比例为 8:1:1。
- 异常与恢复：重复 Task ID、空输入、非法比例、测试场景无样本或非测试任务不足时抛出 `ValueError`。

### `EXPTRAJ-DATA-006`：配额构建与失败重采

- 前置：总量为正，能力矩阵含可支持单元，单元重试上限为正。
- 行为：在任务几何能力矩阵上叠加当前专家可生成能力，排除持续不可达单元并约束单向可达单元；按其余场景×任务类型单元分配配额，循环 noise/可表达的 maneuver 与相邻占用难度；规划失败只在原单元增加 sample index 重采，并累计失败原因。
- 结果：成功时各 split 达到计划数量，写出 schema v2 NPZ 与 JSON manifest；`--dry-run` 只输出计划，不生成 BEV/轨迹文件。
- 异常与恢复：某单元超过重试上限时终止并报告单元、成功数和失败原因；替代任务必须避开所有原计划及已用 task ID，禁止跨 split 泄漏；已完成文件不被伪装为完整数据集。

### `EXPTRAJ-DATA-007`：数据集统计与叠加抽检

- 前置：可读取的 NPZ 数据集；叠加图要求 v2 `bev_meta`。
- 行为：按 mask 统计专家轨迹长度，以轨迹切向相对车头投影统计倒车距离占比，并汇总场景/任务/噪声数量；抽样图把 occupancy/target BEV 与转到起点局部系的专家轨迹叠加。
- 结果：CLI 输出 JSON 统计，可选保存统计文件与 PNG 抽检图。
- 异常与恢复：v1 可输出不依赖任务元数据的基础统计；缺少 BEV 元数据时拒绝绘制并给出明确错误。

### `EXPTRAJ-DATA-008`：可解释的专家轨迹验收图

- 前置：schema v2 样本同时包含 `bev_meta`、起始状态、目标位姿、有效专家轨迹及与样本对齐的可选任务元数据。
- 行为：在起点局部坐标系中以统一矿卡配置绘制 6×3m 起点/目标车身和朝向；专家轨迹按实际位移相对车头投影区分前进与倒车，显示行驶方向和换向点；信息区展示任务类型、场景、难度、路径长度、倒车占比以及轨迹终点相对目标的位置/航向误差。
- 结果：单图可直接辨认“从何种位姿出发、如何行驶、以何种位姿到达以及到位误差”；CLI 默认按任务类型优先选择代表样本，在样本数允许时覆盖全部现有任务类型。
- 异常与恢复：索引越界、目标字段缺失、空轨迹或元数据与样本数量不一致时给出明确错误；旧 v1 数据仍可执行基础统计，但不伪造验收图所需语义。

### `EXPTRAJ-RS-001`：Reeds–Shepp 解析扩展

- 前置：起终位姿自由，最小转弯半径和采样步长为正数。
- 行为：从 12 个基本公式的原型/时间反演/镜像/双对称共 48 词族产生候选，按长度排序；Hybrid A* 在目标邻域对候选依次加密碰撞检查，接入第一条可行曲线。
- 结果：解析轨迹精确到达目标位姿，相邻采样距离不大于采样步长（浮点容差除外），且全位姿通过车身碰撞检查。
- 异常与恢复：单个解析候选冲突时尝试下一候选；无候选可行时继续 Hybrid A* 离散搜索。

### `EXPTRAJ-SMOOTH-001`：碰撞安全的三次捷径平滑

- 行为：在同一行驶方向的轨迹区间内用三次 Hermite 捷径替换局部折线，每个候选按固定步长采样检查车身碰撞和曲率。
- 结果：返回轨迹保持起点、终点和换向边界，路径不长于输入，全轨迹无碰撞。
- 异常与恢复：所有捷径不合格时返回原轨迹副本，不产出未校验轨迹。

### `EXPTRAJ-PROFILE-001`：梯形速度剖面

- 行为：沿几何轨迹作前向加速和反向减速约束传播，在倒车段应用独立的较低限速，在换向点强制停车。
- 结果：速度符号与行驶方向一致，速度、加速、减速不超配置限制，时间轴单调且起终速度为 0。
- 异常与恢复：非正限制、点数不足或显式方向长度不匹配抛 `ValueError`。

## 兼容、迁移与回退

- 无数据迁移；`Trajectory(points, dt)` 契约不变，现有 planner/dataset/runtime 调用无需改造。
- 解析扩展可由规划器构造参数禁用，回退到离散 Hybrid A* 路径；平滑与速度剖面为独立调用，不影响旧调用。
- 新保存文件使用 schema v2；无版本 v1 文件继续读取，旧数组键和含义不变。需要回退写出格式时可由旧版本生成器重建，v2 不覆盖输入归档。

## 安全、隐私与运行限制

- 安全边界由同一 `_pose_free` 车身矩形与 `collision_margin` 统一执行；解析轨迹和平滑轨迹不得绕过它。
- 本功能不处理个人数据、授权或外部通信。
- NPZ 元数据仅保存 Unicode 数组，加载显式禁用 pickle；数据规模仍由可信生成流程控制，调用方应避免加载来源不明的超大归档。
- 48 词族只在解析邻域计算；候选排序后遇到首条碰撞自由曲线即停止，避免全搜索中无界调用。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `EXPTRAJ-PLAN-001` | 终点偏差≤0.6m、轨迹点自由 | `tests/test_planner.py` | `planner/hybrid_astar.py::HybridAStarPlanner` | planner 定向 10 项通过；全仓 111 项通过 | ✅ |
| `EXPTRAJ-DATA-001` | 样本含 5 通道 BEV 且位姿自由 | `tests/test_dataset.py` | `dataset/generator.py::DatasetGenerator` | unittest 2 项通过 | ✅ |
| `EXPTRAJ-DATA-002` | v2 版本、BEV/任务元数据往返与一致性拒绝 | `tests/test_dataset.py::TestDatasetGenerator` | `dataset/generator.py::DatasetGenerator.save/load`; `dataset/generator.py::TrainingSample.task_meta` | 定向 unittest 通过 | ✅ |
| `EXPTRAJ-DATA-003` | 无版本 v1 安全加载 | `tests/test_dataset.py::TestDatasetGenerator.test_v1_archive_still_loads` | `dataset/generator.py::DatasetGenerator.load` | 定向 unittest 通过 | ✅ |
| `EXPTRAJ-DATA-004` | 单目标/T4 Task→样本、元数据与失败定位 | `tests/test_dataset.py::TestTaskDrivenDataset` | `dataset/generator.py::DatasetGenerator`; `dataset/pipeline.py::SensorBEVPipeline.set_target_goals` | 全仓 169 项通过；真实 S1/T1 目标通道与 3000 条 Task 生产通过 | ✅ |
| `EXPTRAJ-DATA-005` | 8:1:1 目标、S9 隔离、seed 复现与无重叠 | `tests/test_splits.py` | `dataset/splits.py::split_tasks`; `DatasetSplits` | 正式归档为 2400/300/300；跨 split 重叠 0、唯一 task ID 3000、test 仅 S9 | ✅ |
| `EXPTRAJ-DATA-006` | 配额、dry-run、同单元重采与 manifest | `tests/test_dataset_build.py`; `scripts/build_dataset.py` | `dataset/build.py::build_task_plan/generate_with_retries/expert_maneuvers`; `sim/tasks.py::TaskSampler.adjacent_occupancy_levels` | 3000 条正式生产完成；train/val/test 重采 296/150/66 次；manifest 与全局 ID 保留验证通过 | ✅ |
| `EXPTRAJ-DATA-007` | 长度/倒车/分布统计与 BEV 叠加图 | `tests/test_dataset_inspection.py`; `scripts/inspect_dataset.py` | `dataset/inspection.py::summarize_dataset/render_sample_overlay` | 三份统计 JSON 与 12 张 PNG 写出；全仓 169 项通过 | ✅ |
| `EXPTRAJ-DATA-008` | 起终矿卡位姿、行驶方向、换向与到位误差可视；代表样本优先覆盖任务类型 | `tests/test_dataset_inspection.py`; `scripts/inspect_dataset.py`; 抽检 PNG 人工查看 | `dataset/inspection.py::render_sample_overlay/select_representative_indices` | 5 项定向测试与全仓 172 项通过；三 split 共 15 张增强图写出，S9/T1–T5 人工图审通过 | ✅ |
| `EXPTRAJ-MARGIN-001` | 膨胀裕度语义与净空保持 | `tests/test_planner.py`（margin 两项测试） | `HybridAStarPlanner._pose_free`/`_splice_valid` | unittest 通过；200 回合地基基线碰撞率 11%→1.5% | ✅ |
| `EXPTRAJ-RS-001` | 48 词族、精确到达、S3/S5 紧凑场景可规划 | `tests/test_reeds_shepp.py`; `tests/test_planner.py` | `planner/reeds_shepp.py::reeds_shepp_paths`; `HybridAStarPlanner._analytic_connection` | 3 项几何 + 3 项解析接入回归通过；1000 组随机 SE(2) 失败 0；200 回合 99.0% 成功/1.0% 碰撞 | ✅ |
| `EXPTRAJ-SMOOTH-001` | 轨迹不变长、不跨换向、所有捷径全位姿无碰撞 | `tests/test_smoothing.py` | `planner/smoothing.py::smooth_trajectory` | 3 项定向回归通过 | ✅ |
| `EXPTRAJ-PROFILE-001` | 限速/加减速/换向停车/单调时间轴 | `tests/test_profile.py` | `planner/profile.py::trapezoidal_velocity_profile` | 4 项定向回归通过 | ✅ |

## 待人工确认

- 无。
