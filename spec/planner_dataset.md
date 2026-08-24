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
- 数据集不落盘格式约定，阶段四按需扩展。

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

## 安全、隐私与运行限制

- 安全边界由同一 `_pose_free` 车身矩形与 `collision_margin` 统一执行；解析轨迹和平滑轨迹不得绕过它。
- 本功能不处理个人数据、授权或外部通信。
- 48 词族只在解析邻域计算；候选排序后遇到首条碰撞自由曲线即停止，避免全搜索中无界调用。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `EXPTRAJ-PLAN-001` | 终点偏差≤0.6m、轨迹点自由 | `tests/test_planner.py` | `planner/hybrid_astar.py::HybridAStarPlanner` | planner 定向 10 项通过；全仓 111 项通过 | ✅ |
| `EXPTRAJ-DATA-001` | 样本含 5 通道 BEV 且位姿自由 | `tests/test_dataset.py` | `dataset/generator.py::DatasetGenerator` | unittest 2 项通过 | ✅ |
| `EXPTRAJ-MARGIN-001` | 膨胀裕度语义与净空保持 | `tests/test_planner.py`（margin 两项测试） | `HybridAStarPlanner._pose_free`/`_splice_valid` | unittest 通过；200 回合地基基线碰撞率 11%→1.5% | ✅ |
| `EXPTRAJ-RS-001` | 48 词族、精确到达、S3/S5 紧凑场景可规划 | `tests/test_reeds_shepp.py`; `tests/test_planner.py` | `planner/reeds_shepp.py::reeds_shepp_paths`; `HybridAStarPlanner._analytic_connection` | 3 项几何 + 3 项解析接入回归通过；1000 组随机 SE(2) 失败 0；200 回合 99.0% 成功/1.0% 碰撞 | ✅ |
| `EXPTRAJ-SMOOTH-001` | 轨迹不变长、不跨换向、所有捷径全位姿无碰撞 | `tests/test_smoothing.py` | `planner/smoothing.py::smooth_trajectory` | 3 项定向回归通过 | ✅ |
| `EXPTRAJ-PROFILE-001` | 限速/加减速/换向停车/单调时间轴 | `tests/test_profile.py` | `planner/profile.py::trapezoidal_velocity_profile` | 4 项定向回归通过 | ✅ |

## 待人工确认

- 无。
