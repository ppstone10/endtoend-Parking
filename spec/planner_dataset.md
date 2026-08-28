# 专家轨迹与训练数据 Spec

## 元数据

- Spec ID 前缀：`EXPTRAJ`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-08-27

## 目标

- 从起始状态到目标泊车位姿生成履带钻机低速运动学可行的专家轨迹，作为 MineParkingNet 训练标签。
- 使用 Reeds–Shepp 解析扩展闭合离散运动基元的格点可达性空洞，并提供经碰撞校验的轨迹平滑与倒车降速剖面。
- 批量生成训练样本（融合 BEV + 目标位姿 + 状态 + 专家轨迹），供阶段四网络训练使用。

## 非目标

- 不做履带—地面滑移、土壤沉陷、质量、惯量或载荷转移等高阶动力学优化；速度剖面只约束纵向加减速、限速与换向停车。
- 不实现网络训练、MPC 闭环（后续阶段）。
- 不在本模块实现网络训练或 P4.1 多车位五项代价 oracle；P2.8 只为 T4 解析一个可规划监督目标并记录策略。

## 边界与约束

- 规划与数据均在 `ParkingEnvironment` 世界坐标系中进行；BEV 为车辆中心局部系。
- 履带低速运动学：`[v, omega]`；平移时允许前进、倒车及差速弧线，完全停车时允许 `v=0, omega≠0` 的左右原地旋转，旋转中心是两履带几何中心，也是轨迹 `(x,y)` 与矩形外廓中心。
- 理论车型默认采用居中 6×3 m 矩形；外廓、底盘控制上限、规划速度/角速度、安全余量、搜索分辨率、原地旋转开关与代价必须来自同一车辆配置，不得在规划、控制、碰撞和可视化中各自硬编码。
- 碰撞检测采用完整车辆矩形而非只检查中心或四角；`collision_margin > 0` 时矩形各向外膨胀该裕度。原地旋转及相邻轨迹位姿之间按矩形最远角点的扫掠距离加密，禁止只检查旋转起止姿态。
- Reeds–Shepp 仅作为满足平移差速弧线约束的候选；近目标还必须允许“原地旋转—直线平移—原地旋转”的履带解析候选。所有候选按统一时间/惩罚成本比较，并逐姿态校验完整外廓和搜索边界；禁止用任意位姿线性插值作为终点直连。
- 搜索需在起点-目标包围盒外加边距的范围内进行，并设探索节点上限防止发散。
- 平滑输入必须是已通过碰撞检查的专家轨迹；平滑不跨越前进/倒车换向点，三次捷径候选只有在轨迹更短、全采样位姿无碰撞且曲率不超限时才可接受。
- 速度剖面不修改几何轨迹；起终点与换向点速度为 0，倒车速度上限不高于前进上限。

## 数据与接口

- `ReedsSheppPath`：保存 48 词族中某个可行候选的转向类型、带符号米制段长、最小转弯半径和总长。段长正/负分别表示前进/倒车。
- 平滑结果：平滑入口仍接收/返回 `Trajectory`，不新增数据类，不改变三阶段统一轨迹接口。
- `VelocityProfile`：与轨迹等长的带符号速度数组和单调时间数组；位于 `planner/` 内，在任务/数据层决定落盘 schema 前不扩展 `interfaces/Trajectory`。
- `HybridAStarPlanner.plan` 保持原有起终位姿调用兼容，并可选接收任务请求方向；返回类型和已有异常类型保持兼容。
- `VehicleConfig`：统一保存理论外廓、执行上限和规划搜索参数；默认从可编辑 JSON 配置加载并可序列化为稳定模型元数据。新数据必须记录模型名称、版本和关键数值，用于阻止旧模型轨迹混入新训练集。
- NPZ schema v2：保留 v1 的 `bevs/goals/states/trajs/masks/dt` 数组，新增标量 `schema_version=2`、单份 `bev_meta` 与逐样本 `task_meta`。元数据以 Unicode JSON 数组保存，加载后解码为字典/字典列表。
- Task 驱动入口：`DatasetGenerator.generate` 同时接受旧式整数数量或 `Task` 可迭代对象；跨场景生成由调用方注入 task→(planner, SensorBEVPipeline) 工厂。
- 划分结果：`DatasetSplits` 拥有互斥的 train/val/test Task；默认 test 场景为 S9，任何该场景样本不得进入 train/val。

## 行为与验收

### `EXPTRAJ-TRACK-001`：可配置履带钻机模型

- 前置：配置包含有限正数的长、宽、速度/角速度上限、规划分辨率、单次规划墙钟上限、至少为 1 的反方向代价系数、非负安全余量和可选的非负解析接管范围；规划速度/角速度不超过底盘执行上限。
- 行为：默认从 `configs/vehicles/tracked_drill_rig.json` 加载居中 6×3 m 理论车型；同一配置向 Hybrid A*、差分运动模型、MPC、碰撞和 inspection 提供参数，并输出稳定模型元数据。影响轨迹标签的搜索限制和方向代价变化必须更新模型版本。
- 结果：只修改配置即可改变外廓、控制上限与 Hybrid A* 搜索参数，无需修改算法源码；默认配置明确标识为理论模型而非实车标定。
- 异常与恢复：缺字段、未知字段、非法数值或规划上限高于执行上限时拒绝加载；可显式传入其他配置文件回退或适配实车。

### `EXPTRAJ-TRACK-002`：履带原地旋转与连续扫掠

- 前置：起终位姿的完整膨胀矩形均无碰撞，原地旋转已在配置中启用。
- 行为：Hybrid A* 在前进/倒车弧线之外扩展左右原地旋转一个航向栅格；原地旋转段保持 `(x,y)` 不变，角速度不超过配置上限，并以较高旋转代价抑制无必要搓地。近目标同时比较履带解析候选与 Reeds–Shepp 候选。
- 结果：同一中心不同航向的目标可仅靠原地旋转精确到达；轨迹包含连续中心位姿序列，所有平移和旋转段的完整矩形扫掠均无碰撞。
- 异常与恢复：任一中间姿态碰撞即拒绝该基元/候选并继续搜索；关闭原地旋转时恢复原有平移运动集，不产生零位移转向标签。

### `EXPTRAJ-PLAN-001`：专家轨迹生成

- 前置：起始 `VehicleState` 与目标 `GoalPose` 均无碰撞。
- 行为：`HybridAStarPlanner.plan` 在状态离散空间搜索履带低速运动学可行路径；默认或前进请求下，倒车/反方向平移相对前进/请求方向按 `2:1` 时间代价计；明确的倒车监督任务仍以倒车为请求方向，从而保留所需能力样本，但不鼓励额外换向。原地旋转相对同持续时间的直行或小范围弧线转向同样按 `2:1` 计，正常平移弧线不附加旋转惩罚。`tracked_pivot_v5` 在不超过 T3 上限 30m 的目标距离内先尝试加密碰撞检查的解析连接，失败后仍回退离散 Hybrid A*。
- 结果：返回 `Trajectory`，终点与目标位姿偏差 ≤ 0.6m；轨迹各点均位于自由空间。
- 异常与恢复：起始/目标碰撞抛 `ValueError`；搜索发散、节点数超限或超过配置的单次规划墙钟上限时抛 `RuntimeError`，由调用方在原任务单元重采。

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
- 行为：单目标任务直接规划；未启用机动门禁时 T4 按稳定候选顺序选择首个可规划目标，默认启用门禁时选择首个可规划且机动一致的目标。采集 Task 起点 BEV，并把 `Task.to_metadata()`、实际噪声 profile、选中目标及解析策略写入样本元数据。
- 结果：每个成功 Task 恰生成一条 `TrainingSample`，状态等于 Task 起点、目标等于已解析目标、轨迹通过专家规划器；输入顺序与输出顺序一致。
- 异常与恢复：所有目标不可规划或组件配置不匹配时抛出携带 task ID 的 `TaskGenerationError`；构建层可据此在同一场景×任务类型单元重采。旧 `generate(int)` 行为保持兼容。
- T4 边界：`first_plannable_candidate` / `first_consistent_plannable_candidate` 都不是“最优车位”标签；P4.1 可通过 goal selector 注入五项代价 oracle。

### `EXPTRAJ-DATA-005`：稳定分层与整场景泛化集

- 前置：Task ID 唯一，划分比例合法且测试保留场景存在。
- 行为：默认把 S9 全部放入 test；其余任务按场景×任务类型分层、用 seed 可复现地选择 val，剩余进入 train。
- 结果：三集合互斥且并集等于输入；S9 不进入 train/val；同一输入与 seed 返回相同 Task ID 集合。构建计划按总量预留 10% S9，使目标比例为 8:1:1。
- 异常与恢复：重复 Task ID、空输入、非法比例、测试场景无样本或非测试任务不足时抛出 `ValueError`。

### `EXPTRAJ-DATA-006`：配额构建与失败重采

- 前置：总量为正，能力矩阵含可支持单元，单元重试上限为正。
- 行为：在任务几何能力矩阵上叠加当前专家可生成能力，排除经完整校准证明持续不可达的单元，并对只有单向成功证据的单元只生成该机动方向；按其余场景×任务类型单元分配配额，循环 noise/准入 maneuver 与相邻占用难度；规划失败只在原单元增加 sample index 重采，并累计失败原因。正式构建按固定批量输出完成数、耗时、失败数和预计剩余量，并把每个已通过双门禁的批次原子写入独立检查点。
- 结果：成功时各 split 达到计划数量，合并检查点写出 schema v2 NPZ 与 JSON manifest；`--dry-run` 只输出计划，不生成 BEV/轨迹文件。相同 seed、车辆模型版本和任务计划可跳过已验证检查点继续构建。
- 异常与恢复：单次规划超时作为稳定失败原因参与重采；某单元超过重试上限时终止并报告单元、成功数和失败原因；替代任务必须避开所有原计划及已用 task ID，禁止跨 split 泄漏；配置、计划或检查点审计不匹配时拒绝恢复，已完成文件不被伪装为完整数据集。

### `EXPTRAJ-DATA-015`：未完成批次的持久失败游标

- 前置：正式构建已通过身份校验，当前固定批次尚未形成同时存在的 NPZ 和批次报告。
- 行为：每个专家生成失败在继续重采前，原子写入与 split、批次索引和原任务 ID 列表绑定的 retry JSON；状态保存各场景×任务类型的下一 sample index、累计失败计数/原因和不得再试的排除 ID。
- 结果：相同身份重启时，重采从已持久化游标继续，不再尝试已排除 ID；批次成功报告合并跨进程失败证据，只有 NPZ 和报告都已原子落盘后才删除 retry JSON。旧失败无持久证据时，可保守排除未完成批次的原任务并保持同难度重采，不伪造具体失败结论。
- 异常与恢复：retry JSON 损坏、schema 未知或批次身份不符时明确拒绝恢复，不猜测游标；达到本次命令的重试上限仍保持受控终止，但错误明确指示使用原命令续建。
- 数据与状态：retry JSON 是未完成批次的临时构建状态，不进入正式 NPZ/manifest；完成批次 JSON 仍是该批次的最终审计证据。
- 兼容、迁移与回退：新状态文件不改已有 identity schema 与成功检查点；无 retry JSON 的旧目录按空状态开始。回退代码前可保留 retry JSON 作为证据，旧代码会忽略它但不会破坏已完成批次。
- 安全与隐私：只记录确定性仿真任务标识与失败分类，不含个人数据或外部通信。

### `EXPTRAJ-CAL-001`：全专家能力单元校准

- 前置：每单元样本数为正，车辆配置合法，任务几何能力与专家可生成能力可读取。
- 行为：校准计划直接枚举全部“几何支持且专家可生成”的场景×任务类型单元，不经过 train/val/test 比例或 `val_count`；每单元按稳定 seed 生成相同数量的 case，并循环可表达的机动方向、噪声和相邻占用难度。
- 结果：默认按当前专家准入构造校准计划；显式全能力探测时忽略历史准入结论，覆盖全部 39 个几何支持单元并轮换前进/倒车，用于新车辆/场景/规划版本重新决策准入。每个 case 都产生成功、失败或预算超时的终态记录。
- 异常：单个 case 失败不得终止其他单元校准；任务采样本身失败也必须成为可汇总记录。

### `EXPTRAJ-CAL-002`：硬预算与中断恢复

- 前置：单 case 总墙钟预算为有限正数，输出目录可写。
- 行为：每个 case 在可回收的独立进程中执行，预算覆盖原任务和同单元重试；超时后终止该 worker 并记录稳定 `task_budget_exceeded`。case 终态先原子写入独立检查点，再更新汇总和运行状态。
- 恢复：身份绑定 schema、seed、车辆模型、完整 case 计划、每单元样本数、重试和预算；身份一致时跳过已完成 case，运行中断时只重做尚无终态检查点的当前 case；身份不一致时拒绝复用目录。
- 结果：`run_state.json` 区分 running/interrupted/completed/failed；任何时刻的 `report.json` 都明确完成数、剩余数和是否 partial，不把中断结果伪装为完整校准。

### `EXPTRAJ-CAL-003`：能力报告

- 行为：按 case 和场景×任务类型聚合计划数、成功数、失败数、预算超时数、重试失败次数、成功率、总耗时和平均耗时；JSON 保存完整结构，CSV 提供单元级表格。
- 结果：报告足以判断哪些单元可进入正式生成、哪些需要隔离或调优；报告不自动修改专家能力矩阵。

### `EXPTRAJ-DATA-007`：数据集统计与叠加抽检

- 前置：可读取的 NPZ 数据集；叠加图要求 v2 `bev_meta`。
- 行为：按 mask 统计专家轨迹长度，以轨迹切向相对车头投影统计倒车距离占比，并汇总场景/任务/噪声数量；抽样图把 occupancy/target BEV 与转到起点局部系的专家轨迹叠加。
- 结果：CLI 输出 JSON 统计，可选保存统计文件与 PNG 抽检图。
- 异常与恢复：v1 可输出不依赖任务元数据的基础统计；缺少 BEV 元数据时拒绝绘制并给出明确错误。

### `EXPTRAJ-DATA-008`：可解释的专家轨迹验收图

- 前置：schema v2 样本同时包含 `bev_meta`、起始状态、目标位姿、有效专家轨迹及与样本对齐的可选任务元数据。
- 行为：在起点局部坐标系中按样本记录的履带钻机配置绘制起点/目标车身和朝向；专家轨迹按实际位移相对车头投影区分前进与倒车，显示行驶方向和换向点；信息区展示任务类型、场景、难度、路径长度、倒车占比以及轨迹终点相对目标的位置/航向误差。
- 结果：单图可直接辨认“从何种位姿出发、如何行驶、以何种位姿到达以及到位误差”；CLI 默认按任务类型优先选择代表样本，在样本数允许时覆盖全部现有任务类型。
- 异常与恢复：索引越界、目标字段缺失、空轨迹或元数据与样本数量不一致时给出明确错误；旧 v1 数据仍可执行基础统计，但不伪造验收图所需语义。

### `EXPTRAJ-DATA-009`：任务机动与专家轨迹一致性审计

- 数据定义：沿每段起点航向投影实际位移，投影为负的段计入倒车距离，其余非零段计入前进距离；任务声明方向的距离占总路径距离比例称为“请求方向占比”。默认最低请求方向占比为 0.5，即请求方向必须至少承担一半实际行驶距离，允许短距离反向调整与换向。
- 行为：对具有 `difficulty.maneuver` 的样本逐项计算前进/倒车距离、请求方向占比、换向次数与一致性；数据集统计汇总已审计、一致、不一致和缺失声明数量，并按请求方向及场景×任务类型汇总不一致项。
- 结果：审计输出不依赖已有 `task_meta.dataset` 中的派生值，可从轨迹和任务声明独立复算；新生成样本把相同审计结果写入 `task_meta.dataset.maneuver_audit`，便于追溯判定阈值与实际比例。
- 异常与恢复：轨迹形状非法、少于两个点、含非有限值或总行驶距离为零时拒绝判定；v1 或缺少 `maneuver` 的样本计为未审计，不猜测标签。

### `EXPTRAJ-DATA-010`：机动一致性数据门禁

- 前置：Task 驱动生成已得到候选目标的专家轨迹；生成器默认启用门禁，最低请求方向占比默认 0.5。
- 行为：候选轨迹未达到请求方向占比时不得生成 `TrainingSample`；T4 继续尝试下一候选目标，其他任务或所有 T4 候选均不一致时抛出携带 task ID、请求方向、实际前进/倒车占比的 `TaskGenerationError`，并以稳定原因码进入同场景×任务类型×难度重采。inspection CLI 的严格审计开关在归档存在任一不一致、无效轨迹或缺失机动声明样本时返回失败。
- 结果：默认新构建归档的已声明样本一致率为 100%；门禁可由生成器显式关闭以审计/兼容旧流程，但仍写入 `maneuver_audit.consistent=false`，不得伪装为通过。
- 兼容、迁移与回退：NPZ 仍为 schema v2，原数组键不变，任务元数据只增加派生审计字段；既有 NPZ 不自动改写，先以 inspection 审计，需满足新门禁时由原 seed/计划重建。显式 `enforce_maneuver_consistency=False` 是回退路径。

### `EXPTRAJ-DATA-011`：轨迹运动学与碰撞可行性审计门禁

- 数据定义：平移段按相邻中心位姿、固定 `dt` 复算线速度、角速度和中点航向横向残差；中心位移近零且航向变化非零的段为原地旋转段。生成期使用规划器的完整外廓检查复核每个轨迹位姿和相邻位姿扫掠。
- 行为：Task 候选只有在数值有限、起终点存在、线/角速度不过限、平移横向残差在容差内、原地旋转中心不漂移、全部扫掠无碰撞且模型元数据与当前配置一致时才可成为专家标签；partial 晋升前从 NPZ 独立复算运动学并核对生成期碰撞证据和模型版本。
- 结果：新正式数据集的已声明样本可行率为 100%，统计输出平移/原地旋转段数、最大速度、最大角速度、最大横向残差以及模型不匹配/缺失碰撞证据数量。
- 异常、迁移与回退：失败候选以稳定原因进入原任务单元重采；旧 NPZ 不原地改写，因缺少新模型和碰撞证据而不能通过严格门禁。门禁可显式关闭用于调查，但不得把 FAIL 元数据伪装为通过。

### `EXPTRAJ-DATA-012`：中心轨迹与中间车身姿态验收图

- 坐标定义：轨迹局部系为右手系“前方 `x`、车体左方 `y`、正 `yaw` 表示相对当前车头左转”。验收图必须让正 `y` 实际显示在画面左侧，禁止把“左为正”的数值直接映射成“屏幕向右为正”而产生横向镜像。
- 行为：验收图除起终外廓外，沿轨迹显示离散车体中心点及航向；按距离、原地旋转段、换向点和曲率变化选择中间位姿，绘制半透明 6×3 m 车身包络。连续零位移航向变化汇总为原地旋转事件，在固定中心使用独立颜色，并从旋转前航向沿有符号累计角绘制方向箭头；标签使用车体语义 `LEFT/RIGHT + 角度绝对值`。信息区同时显示事件数、累计绝对旋转角和单次最大角。
- 结果：即使中心轨迹很短或多个航向点重合，单图仍可按当前车头判断旋转发生位置、车体左/右转、累计角度、旋转后的车头/车尾方向、中心是否固定和中间车身是否侵入障碍，并看到运动学/碰撞可行性 PASS/FAIL。
- 异常与恢复：缺少模型元数据时使用当前理论配置绘图但明确标为未确认模型；不得把缺少碰撞审计的旧样本显示为可行性 PASS。

### `EXPTRAJ-DATA-013`：旧专家数据可视化参考归档与清理

- 目标与非目标：在重建当前专家数据前，从旧归档保留足以解释可视化行为和历史模型差异的少量真实样本；参考归档不是训练集、校准报告或全量数据备份，不用于恢复被删除的旧 train/val/test。
- 数据与状态：归档至少包含一个安全加载的 schema v2 NPZ、逐样本来源路径/索引/任务 ID/场景/任务类型/机动/选择理由、来源与归档文件 SHA-256、车辆模型版本、归档前后文件数与字节数，以及使用当前 inspection 生成的 summary 和 PNG。样本覆盖前进、倒车、不同任务类型和大角度原地旋转；重复类别可合并但不得只保留合成样本。
- 操作接口：清理只能作用于显式解析且位于项目 `data/task_dataset` 下的旧专家数据目标；归档目录必须位于同一根目录并从删除集合排除。归档 NPZ 可由 `DatasetGenerator.load` 在禁用 pickle 的条件下重读，PNG 必须在删除前成功生成。
- 异常与恢复：任一来源不可读、归档写入失败、哈希/样本数不匹配、PNG 未生成或目标路径越界时，停止删除并保留原数据。归档验证成功后才允许删除其余旧数据；删除完成后再次核对根目录只含保留归档和明确允许的新数据。
- 兼容与迁移：旧模型版本和可行性状态按原元数据如实保存，不伪装成当前 v4 正式数据。后续 v4 数据必须写入新目录；需要旧全量数据时按原 seed、配置与历史版本代码重建，不能将参考归档扩充或复制为训练集。
- 安全与隐私：仅处理项目生成的仿真数组和图片，不含个人数据或外部上传；删除前记录精确清单、大小和授权边界。删除是不可逆操作，归档只能恢复保留样本，不能恢复其余旧数据。

### `EXPTRAJ-DATA-014`：`tracked_pivot_v4` 校准后专家准入

- 前置与证据：决策来源必须是与当前车辆模型身份一致、全部 case 有终态且非 partial 的校准报告。当前基线为 `tracked_pivot_v4`、seed `20260824`、每单元 3 case、每 case 最多 3 次尝试的 105/105 完整报告。
- 单元准入：`S7_fuel_station/T1`、`S9_mine_complex/T3` 因为 0/3 成功不进入正式计划；`S5_crusher/T2` 因为仅 1/3 成功、唯一成功耗时贴近 8 秒上限且同 task 复现超时，同样暂不准入。`S5_crusher/T1/T5` 只准入倒车；`S9_mine_complex/T1/T5` 只准入前进。既有 `S3_maintenance` 全任务倒车限制和 `S8_weigh_station/T5` 前进限制保持不变；其他准入单元仍循环可表达方向。
- 可观察结果：3000 条 dry-run 不得出现上述两个排除单元或受限单元的反方向任务；总量仍为 train/val/test `2400/300/300`，test 仍只含 S9。已准入但偶发失败的单元继续在原单元和原难度重采，不跨单元隐藏失败。
- 迁移与恢复：专家准入变更会改变任务计划身份，不得恢复或混用变更前的正式构建检查点。当规划器、车辆模型版本或任务几何改变时，必须重新完成全准入校准才能放宽或改写这些规则。
- 回退、安全与隐私：回退只允许恢复到上一个有完整校准证据的准入矩阵，不能通过提高重试次数伪装不可达单元。本变更仅影响仿真任务采样，不涉及个人数据、外部通信或实车安全承诺。

### `EXPTRAJ-DATA-016`：`tracked_pivot_v5` 完整校准后专家准入

- 前置与证据：只接受模型身份为 `tracked_pivot_v5`、`probe_full_capability=true`、117/117 case 全部终态且非 partial 的报告；方向稳定性按每个 case 的最终状态和内部重采证据共同判断，单次偶发成功不能单独放宽准入。
- 单元准入：S3/T3 仅前进；S8/T2、T3、T5 仅前进；S3 其余任务与 S5/T1、T5 仅倒车；S9/T5 仅前进。S5/T2、S7/T1/T3/T5、S9/T1/T3 不进入正式计划；其他几何支持单元保留前进/倒车轮换。
- 可观察结果：3000 条 dry-run 仍为 train/val/test `2400/300/300`，普通场景覆盖 30 个准入单元，S9 test 覆盖 T2/T4/T5 三个单元；受限单元不得出现相反方向，排除单元不得出现。S4/T1–T5 保留双向监督。
- 异常与恢复：报告身份、完整性或 case 计划不匹配时拒绝据此改写准入；已准入任务的偶发失败仍在原场景×类型×难度内重采并持久化游标，不能跨单元替代。
- 迁移与回退：v5 准入改变任务计划指纹，只能写入新的 `tracked_pivot_v5_3000`；回退必须同时恢复 v4 配置、v4 准入和 v4 数据目录，禁止混合检查点或样本。
- 安全与隐私：本准入仅声明当前仿真专家生成稳定性，不声明真实车辆可达性；不涉及个人数据、外部通信或实车控制授权。

### `EXPTRAJ-DATA-017`：紧 bay 轴线对齐起点采样与可行驶几何准入

- 前置与证据：S3/T3 在 v5 正式构建反复耗尽重试；诊断显示远距离（15–30m）起点若与目标 bay 横向偏移 ≥4m 或带随机大转角，Hybrid A* 的 Reeds-Shepp/履带解析候选会在紧 bay 口全部碰撞、离散搜索难以收敛（25s/10 万节点仍失败）；起点与 bay 轴线平行、偏移 ≤1.5m 时倒车 8/8 秒级成功。S6/T3 根因是西南岩石挡住接近走廊（去岩石后 12/12 秒过）；S7/T2 根因是平行靠岛位未计入 `collision_margin`（实际净空仅 0.1m）；S9/T2 根因是破碎槽朝向与 S5/规格不一致（前进入槽）。
- 行为：`TaskSampler._sample_axial_start` 对紧 bay 车位（`maintenance_bay`/`crusher_slot`/`fuel_bay`）的 T2/T3/T4/T5 起点沿 bay 轴线对齐采样：yaw 恒等于车位目标 yaw，按机动方向从轴线鼻前侧（倒车）或鼻后侧（前进）`d ∈ 距离档` 米外接近，横向偏移受限；机动方向与 bay 几何不一致时回退常规采样。S3 作业道按 3.5 倍车长加深、spawn 与外墙保持不小于原地旋转扫掠半径；S7 靠岛净空计为 `0.3m + collision_margin`；S9 破碎槽统一倒车入槽。
- 准入结果：S3 全部单元恢复倒车监督（含 T3）；S5/T2 从排除恢复为仅倒车；S9/T2、T5 仅倒车；S8/T2/T3/T5 仅前进。3000 条 dry-run 保持 `2400/300/300`、S9 test 隔离（T2/T4/T5 各 100）。
- 可观察结果：S3/T2–T5、S5/T2、S6/T3、S7/T2、S9/T2 在 8s 预算下成功率 ≥5/6（多数 6/6、毫秒级）；100 条端到端构建烟测仅 3 次失败重采。
- 异常与恢复：对齐采样找不到自由起点时回退常规采样，失败仍按原单元重采；准入方向必须与 bay 几何一致，否则该单元退回旧失败模式。
- 迁移与回退：几何与采样变更改变任务计划指纹，正式数据只写新目录（如 `tracked_pivot_v6_3000`）；旧 v5 检查点、校准证据与归档保持不变。回退需成套恢复旧代码、旧准入与旧目录。
- 安全与隐私：本约束仅声明仿真专家生成稳定性；不涉及个人数据、外部通信或实车控制授权。

### `EXPTRAJ-RS-001`：Reeds–Shepp 解析扩展

- 前置：起终位姿自由，最小转弯半径和采样步长为正数。
- 行为：从 12 个基本公式的原型/时间反演/镜像/双对称共 48 词族产生候选，按长度排序；Hybrid A* 在目标邻域对候选依次加密碰撞检查，接入第一条可行曲线。
- 结果：解析轨迹精确到达目标位姿，相邻采样距离不大于采样步长（浮点容差除外），且全位姿通过车身碰撞检查。
- 异常与恢复：单个解析候选冲突时尝试下一候选；无候选可行时继续 Hybrid A* 离散搜索。

### `EXPTRAJ-SMOOTH-001`：碰撞安全的三次捷径平滑

- 行为：在同一行驶方向的轨迹区间内用三次 Hermite 捷径替换局部折线，每个候选按固定步长采样检查车身碰撞和曲率。
- 结果：返回轨迹保持起点、终点和换向边界，路径不长于输入，全轨迹无碰撞。
- 异常与恢复：所有捷径不合格时返回原轨迹副本，不产出未校验轨迹。

### `EXPTRAJ-PROFILE-001`：梯形速度与原地旋转时间剖面

- 行为：沿几何轨迹作前向加速和反向减速约束传播，在倒车段应用独立的较低限速，在换向点和原地旋转两端强制线速度为 0；零位移航向变化按最大角速度计算持续时间。
- 结果：速度符号与行驶方向一致，速度、加速、减速不超配置限制，原地旋转不产生中心平移且具有非零物理持续时间，时间轴单调且起终速度为 0。
- 异常与恢复：非正限制、点数不足或显式方向长度不匹配抛 `ValueError`。

## 兼容、迁移与回退

- `Trajectory(points, dt)` 与 NPZ schema v2 数组键保持不变；新任务元数据增加车辆模型与可行性审计。旧归档不迁移，严格门禁拒绝缺少新证据的样本，需按原计划重建。
- 解析扩展可由规划器构造参数禁用，回退到离散 Hybrid A* 路径；平滑与速度剖面为独立调用，不影响旧调用。
- 新保存文件使用 schema v2；无版本 v1 文件继续读取，旧数组键和含义不变。需要回退写出格式时可由旧版本生成器重建，v2 不覆盖输入归档。
- 旧专家数据清理只保留显式参考归档；它不改变 NPZ schema，也不提供完整训练集恢复。后续正式数据始终使用新模型版本和新输出目录。

## 安全、隐私与运行限制

- 安全边界由同一 `_pose_free` 车身矩形与 `collision_margin` 统一执行；解析轨迹和平滑轨迹不得绕过它。
- 默认配置是理论等比车型；高阶履带滑移和真实制动能力必须由后续实车标定验证，不能仅凭几何/运动学 PASS 宣称实车安全。
- 本功能不处理个人数据、授权或外部通信。
- NPZ 元数据仅保存 Unicode 数组，加载显式禁用 pickle；数据规模仍由可信生成流程控制，调用方应避免加载来源不明的超大归档。
- 清理操作不得使用未解析变量、通配符或工作区根目录作为递归删除目标；归档验证失败时禁止继续删除。
- 48 词族只在解析邻域计算；候选排序后遇到首条碰撞自由曲线即停止，避免全搜索中无界调用。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `EXPTRAJ-TRACK-001` | JSON 配置统一驱动外廓、控制、方向代价和时间上限，非法配置拒绝 | `tests/test_viz.py::TestVehicleConfig`; `tests/test_planner.py` | 待补 | 待验证 | ⏳ |
| `EXPTRAJ-TRACK-002` | 同中心异航向可规划；旋转中间姿态碰撞时拒绝 | `tests/test_planner.py` | `planner/hybrid_astar.py::HybridAStarPlanner._expand/_tracked_direct_candidates`; `planner/collision.py::RectangleFootprintCollisionChecker` | 纯原地旋转、端点自由但扫掠碰撞、车内障碍三项回归通过；S3/S5 回归通过 | ✅ |
| `EXPTRAJ-PLAN-001` | 默认倒车/反方向与原地旋转相对常规前进/小转向按 2:1 计；显式倒车任务仍保留；单次规划有墙钟上限 | `tests/test_planner.py`; 正式计划长尾基准 | `planner/hybrid_astar.py::HybridAStarPlanner._translation_cost_factor/_expand/_directional_translation_cost/_tracked_direct_candidates`; `configs/vehicles/tracked_drill_rig.json` | 2:1 定向回归、模型 v4 注入和全仓 225 项通过；长耗时 v4 校准/正式生成按约定留给用户执行 | ✅ |
| `EXPTRAJ-DATA-001` | 样本含 5 通道 BEV 且位姿自由 | `tests/test_dataset.py` | `dataset/generator.py::DatasetGenerator` | unittest 2 项通过 | ✅ |
| `EXPTRAJ-DATA-002` | v2 版本、BEV/任务元数据往返与一致性拒绝 | `tests/test_dataset.py::TestDatasetGenerator` | `dataset/generator.py::DatasetGenerator.save/load`; `dataset/generator.py::TrainingSample.task_meta` | 定向 unittest 通过 | ✅ |
| `EXPTRAJ-DATA-003` | 无版本 v1 安全加载 | `tests/test_dataset.py::TestDatasetGenerator.test_v1_archive_still_loads` | `dataset/generator.py::DatasetGenerator.load` | 定向 unittest 通过 | ✅ |
| `EXPTRAJ-DATA-004` | 单目标/T4 Task→样本、元数据与失败定位 | `tests/test_dataset.py::TestTaskDrivenDataset` | `dataset/generator.py::DatasetGenerator`; `dataset/pipeline.py::SensorBEVPipeline.set_target_goals` | 全仓 169 项通过；真实 S1/T1 目标通道与 3000 条 Task 生产通过 | ✅ |
| `EXPTRAJ-DATA-005` | 8:1:1 目标、S9 隔离、seed 复现与无重叠 | `tests/test_splits.py` | `dataset/splits.py::split_tasks`; `DatasetSplits` | 正式归档为 2400/300/300；跨 split 重叠 0、唯一 task ID 3000、test 仅 S9 | ✅ |
| `EXPTRAJ-DATA-006` | 配额、方向感知重采、批量进度、可恢复检查点与 manifest | `tests/test_dataset_build.py`; `scripts/build_dataset.py`; 中断恢复烟测 | `dataset/build.py::build_task_plan/expert_maneuvers/generate_with_retries`; `scripts/build_dataset.py` | 准入配额和原单元重采回归通过；批次原子检查点的正式长任务验收等待 v5 构建 | ⚠️ |
| `EXPTRAJ-DATA-007` | 长度/倒车/分布统计与 BEV 叠加图 | `tests/test_dataset_inspection.py`; `scripts/inspect_dataset.py` | `dataset/inspection.py::summarize_dataset/render_sample_overlay` | 三份统计 JSON 与 12 张 PNG 写出；全仓 169 项通过 | ✅ |
| `EXPTRAJ-DATA-008` | 起终矿卡位姿、行驶方向、换向与到位误差可视；代表样本优先覆盖任务类型 | `tests/test_dataset_inspection.py`; `scripts/inspect_dataset.py`; 抽检 PNG 人工查看 | `dataset/inspection.py::render_sample_overlay/select_representative_indices` | 5 项定向测试与全仓 172 项通过；三 split 共 15 张增强图写出，S9/T1–T5 人工图审通过 | ✅ |
| `EXPTRAJ-DATA-009` | 方向距离、请求占比、换向与分层不一致统计 | `tests/test_maneuver_audit.py`; `tests/test_dataset_inspection.py` | `dataset/maneuver.py::audit_maneuver_consistency/summarize_maneuver_consistency`; `dataset/inspection.py::summarize_dataset` | 定向 11 项与全仓 182 项通过；既有 3000 条审计出 708 条不一致、0 无效、0 缺失 | ✅ |
| `EXPTRAJ-DATA-010` | 生成前拒绝不一致轨迹、T4 候选继续、稳定重采原因与严格 CLI | `tests/test_dataset.py`; `tests/test_dataset_build.py`; `scripts/inspect_dataset.py`; `scripts/build_dataset.py` | `dataset/generator.py::DatasetGenerator._resolve_goal`; `dataset/build.py::generate_with_retries`; `dataset/maneuver.py::require_maneuver_consistency` | 10 条真实构建烟测经生成/partial 双门禁后全 split 100%；旧 test 严格检查按预期失败；Spec 检查 PASS | ✅ |
| `EXPTRAJ-DATA-011` | 运动学独立复算、生成期扫掠碰撞、模型版本与 partial 严格门禁 | `tests/test_trajectory_feasibility.py`; `tests/test_dataset.py`; `tests/test_dataset_build.py`; `scripts/build_dataset.py` | `dataset/feasibility.py`; `DatasetGenerator._audit_feasibility`; `require_trajectory_feasibility` | 定向审计/归档门禁通过；真实 10 条 train/val/test 构建可行率 100%；全仓 191 项通过 | ✅ |
| `EXPTRAJ-DATA-012` | 正 `Left` 显示在画面左侧；旋转箭头从旋转前航向沿真实有符号角绘制；车头/车尾与 body LEFT/RIGHT 一致 | `tests/test_dataset_inspection.py`; `scripts/inspect_dataset.py`; 抽检 PNG 人工查看 | `dataset/inspection.py::_orient_left_positive_axis/_pivot_arrow_points/_draw_pivot_event` | 镜像坐标和“右转 180° 后车体左转 3°”屏幕方向回归通过；真实样本 PNG 人工图审通过；全仓 228 项通过 | ✅ |
| `EXPTRAJ-DATA-013` | 真实代表样本归档含来源、哈希、版本、统计和 PNG；验证成功后仅删除边界内冗余旧数据 | `tests/test_archive_visual_references.py`; 归档 manifest；`DatasetGenerator.load`; 删除前后清单；人工 PNG 检查 | `scripts/archive_visual_references.py::create_visual_reference_archive`; `data/task_dataset/visual_reference_archive_v3` | 7 条 schema v2 真实样本覆盖 T1–T5/前进/倒车，7 张 PNG 和全部 SHA-256 复核通过；912 文件/27,855,463 字节→12 文件/1,758,656 字节；全仓 228 项通过 | ✅ |
| `EXPTRAJ-DATA-014` | 零成功/不稳定单元不进入计划；受限单元只生成校准成功方向；3000 条仍为 2400/300/300 | `tests/test_dataset_build.py`; `tests/test_dataset_calibration.py`; `scripts/build_dataset.py --dry-run`; 定向真实烟测 | `dataset/build.py::_EXPERT_UNREACHABLE_CELLS/_EXPERT_MANEUVER_OVERRIDES/expert_maneuvers`; `dataset/calibration.py::build_calibration_cases` | 32 单元计划无 3 个排除单元和受限反方向；4 个保留弱组合真实轨迹双门禁 PASS；17 项定向与全仓 228 项通过 | ✅ |
| `EXPTRAJ-DATA-015` | 未完成批次持久化失败游标，原命令续建不重放已排除 ID | `tests/test_dataset_build.py::TestBatchRetryState`; 现有 v4 第 294 批恢复探针 | `dataset/build.py::generate_with_retries`; `scripts/build_dataset.py::_load_retry_state_or_default/_record_retry_failure/_build_split_in_batches` | 两次进程轮次的失败 ID 集互斥；retry 身份错配拒绝；现有 1465 条检查点未改，第 294 批首个恢复任务为 sample index 124；全仓 232 项与 Spec 检查通过 | ✅ |
| `EXPTRAJ-DATA-016` | v5 报告驱动排除/方向限制，3000 条保持 split 且不含不稳定单元 | `tests/test_dataset_build.py`; v5 `report.json/cells.csv`; `scripts/build_dataset.py --dry-run` | `dataset/build.py::expert_maneuvers` | 117/117 报告完整；30 普通+3 S9 单元；2400/300/300；4 个关键方向真实双门禁通过；全仓 237 项通过 | ✅ |
| `EXPTRAJ-DATA-017` | 紧 bay 对齐起点采样、S3/S7/S9 可行驶几何、准入方向与几何一致 | `tests/test_tasks.py::TestTaskPlannerIntegration.test_axial_bay_long_distance_start_is_aligned_and_plannable`; `tests/test_scenes.py::TestMineScaleContract.test_s3_maintenance_bays_are_vehicle_relative/test_s7_fuel_bay_keeps_margin_aware_island_clearance`; `scripts/build_dataset.py --dry-run` | `sim/tasks.py::TaskSampler._sample_axial_start`; `sim/scenes.py::s3_maintenance/s7_fuel_station/s9_mine_complex`; `dataset/build.py::_EXPERT_UNREACHABLE_CELLS/_EXPERT_MANEUVER_OVERRIDES` | S3/T3 由反复超时变为 6/6（0.016s）；S5/T2、S6/T3、S7/T2、S9/T2 八秒扫描 ≥5/6；100 条端到端烟测 3 次重采；全仓 239 项通过 | ✅ |
| `EXPTRAJ-CAL-001` | 全部专家能力单元等额校准且失败不中断 | `tests/test_dataset_calibration.py` | `dataset/calibration.py::build_calibration_cases/run_calibration` | v4 准入计划 96 case 保持；v5 全能力模式覆盖 39 单元/117 case 并轮换双向；全仓 237 项通过 | ✅ |
| `EXPTRAJ-CAL-002` | case 硬预算、原子状态、身份校验与中断续跑 | `tests/test_dataset_calibration.py`; `scripts/calibrate_dataset.py` | `dataset/calibration.py::run_case_with_budget/run_calibration` | 中断后跳过已完成项、身份拒绝、0.01s 硬超时及真实 S1 worker 成功路径通过 | ✅ |
| `EXPTRAJ-CAL-003` | case 与单元级 JSON/CSV 能力报告 | `tests/test_dataset_calibration.py` | `dataset/calibration.py::_aggregate_report/_write_csv_atomic` | partial/completed、单元完成率/成功率与 CSV 回归通过 | ✅ |
| `EXPTRAJ-MARGIN-001` | 膨胀裕度语义与净空保持 | `tests/test_planner.py`（margin 两项测试） | `HybridAStarPlanner._pose_free`/`_splice_valid` | unittest 通过；200 回合地基基线碰撞率 11%→1.5% | ✅ |
| `EXPTRAJ-RS-001` | 48 词族、精确到达、S3/S5 紧凑场景可规划 | `tests/test_reeds_shepp.py`; `tests/test_planner.py` | `planner/reeds_shepp.py::reeds_shepp_paths`; `HybridAStarPlanner._analytic_connection` | 3 项几何 + 3 项解析接入回归通过；1000 组随机 SE(2) 失败 0；200 回合 99.0% 成功/1.0% 碰撞 | ✅ |
| `EXPTRAJ-SMOOTH-001` | 轨迹不变长、不跨换向、所有捷径全位姿无碰撞 | `tests/test_smoothing.py` | `planner/smoothing.py::smooth_trajectory` | 3 项定向回归通过 | ✅ |
| `EXPTRAJ-PROFILE-001` | 限速/加减速/换向及原地旋转停车/角速度耗时/单调时间轴 | `tests/test_profile.py` | `planner/profile.py::trapezoidal_velocity_profile` | 5 项定向回归与全仓回归通过 | ✅ |

## 待人工确认

- 无。
