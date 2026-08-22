# 专家轨迹与训练数据 Spec

## 元数据

- Spec ID 前缀：`EXPTRAJ`
- 强度：轻量
- 状态：已采纳
- 最后更新：2026-08-21

## 目标

- 从起始状态到目标泊车位姿生成差分驱动可行的专家轨迹，作为 MineParkingNet 训练标签。
- 批量生成训练样本（融合 BEV + 目标位姿 + 状态 + 专家轨迹），供阶段四网络训练使用。

## 非目标

- 不做动力学约束与速度剖面优化，只做运动学可行轨迹。
- 不实现网络训练、MPC 闭环（后续阶段）。
- 数据集不落盘格式约定，阶段四按需扩展。

## 边界与约束

- 规划与数据均在 `ParkingEnvironment` 世界坐标系中进行；BEV 为车辆中心局部系。
- 差分驱动运动学：`[v, omega]`，规划参考速度 `plan_v`，最大角速度 `max_omega`。
- 碰撞检测采用车辆矩形（长×宽）四角全部位于自由空间。
- 搜索需在起点-目标包围盒外加边距的范围内进行，并设探索节点上限防止发散。

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

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `EXPTRAJ-PLAN-001` | 终点偏差≤0.6m、轨迹点自由 | `tests/test_planner.py` | `planner/hybrid_astar.py::HybridAStarPlanner` | unittest 4 项通过 | ✅ |
| `EXPTRAJ-DATA-001` | 样本含 5 通道 BEV 且位姿自由 | `tests/test_dataset.py` | `dataset/generator.py::DatasetGenerator` | unittest 2 项通过 | ✅ |

## 待人工确认

- 无。