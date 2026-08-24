# 任务层 Spec

## 元数据

- Spec ID 前缀：`TASK`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-08-24

## 目标

- 在 S1–S9 场景之上提供 T1–T5 任务模型与可复现采样，作为规划评测和后续数据生成的统一输入。
- 让距离、机动方向、相邻占用、通道宽度和噪声等级以正交、稳定的元数据字段进入实验记录。
- 显式表达现有场景几何对任务类型的支持能力，不生成违反距离或候选车位定义的“伪任务”。

## 非目标

- 不实现候选车位评分器（`FR-SELECT-01`）或替 T4 选择最终目标。
- 不执行 T5 动态事件，不修改闭环运行器；本层只给出可由后续运行器消费的触发与障碍载荷。
- 不落盘轨迹、传感器帧或任务数据集（P2.8）。

## 边界与约束

- `TaskSampler` 只依赖场景/几何层，不反向依赖规划器；规划可达性由规划器集成验收覆盖。
- T1 距离为 `[4, 8]m`，T2 为 `[8, 15]m`，T3 为 `[15, 30]m`；边界包含在对应区间内。
- T4 必须携带 3–6 个不同空闲车位候选且不预选目标。少于 3 个空闲车位的场景不支持 T4。
- 9×5 能力矩阵必须包含全部 45 个单元；不满足当前场景几何的单元标记为不支持并给出原因。非严格矩阵采样跳过这些单元，严格模式拒绝不完整矩阵。
- 相同根 seed、场景、任务类型和样本索引必须得到相同任务元数据；各矩阵单元的随机流由稳定整数索引派生，不依赖调用顺序或 Python 哈希。
- 相邻占用只在支持 `occupied_pattern` 的场景生效；请求不可表达的占用数量时显式拒绝。
- T5 事件以路径进度比例作为触发条件，载荷为新增圆形车辆障碍；任务层不直接改变 `ParkingEnvironment`。

## 行为与验收

### `TASK-MODEL-001`：稳定任务契约

- 前置：合法的场景、起点、单目标或候选目标、难度参数。
- 行为：构造 `Task`，并通过 `to_metadata()` 导出仅含 JSON 基础类型的稳定记录。
- 结果：T1/T2/T3/T5 恰有一个目标；T4 无预选目标且有 3–6 个候选；T5 恰有一个动态事件。非法组合在构造时抛出 `ValueError`。
- 验收：`tests/test_tasks.py::TestTaskModel` 通过。

### `TASK-SAMPLE-001`：可复现的分层采样

- 前置：根 seed、场景名、T1–T5 类型和样本索引。
- 行为：`TaskSampler.sample` 在场景起点区内采样车辆完整 footprint 无碰撞的起点，并满足任务距离层；可选机动方向、相邻占用和噪声等级写入难度元数据。
- 结果：相同输入得到相同元数据，不同样本索引派生不同 task seed/id；不满足定义时显式抛 `UnsupportedTaskError`。
- 验收：`tests/test_tasks.py::TestTaskSampling` 通过。

### `TASK-MATRIX-001`：完整能力矩阵与有效任务集合

- 前置：S1–S9 注册表和 T1–T5 类型表。
- 行为：`capability_matrix()` 返回 45 个稳定排序单元；`sample_matrix()` 对支持单元采样，严格模式要求所有请求单元均支持。
- 结果：每个返回任务满足自身类型不变量；不支持单元带可读原因，且不会被静默降级为越界距离或不足候选数的任务。
- 验收：`tests/test_tasks.py::TestTaskMatrix` 通过。

### `TASK-DYNAMIC-001`：T5 动态事件描述

- 前置：可采样的 T5 单目标任务。
- 行为：生成一次性、路径进度触发的新增车辆障碍事件，位置位于对应进度截面的自由点。
- 结果：事件触发比例在 `(0, 1)`，障碍半径为正，序列化载荷稳定；采样时环境保持不变。
- 验收：`tests/test_tasks.py::TestTaskDynamicEvent` 通过。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `TASK-MODEL-001` | 类型不变量与 JSON 元数据 | `tests/test_tasks.py::TestTaskModel` | `sim/tasks.py::Task/TaskGoal/TaskDifficulty` | unittest 通过 | ✅ |
| `TASK-SAMPLE-001` | 距离、footprint、seed 与难度轴 | `tests/test_tasks.py::TestTaskSampling` | `sim/tasks.py::TaskSampler.sample` | unittest 通过；1000 任务压测失败 0 | ✅ |
| `TASK-MATRIX-001` | 45 单元能力表、严格/非严格矩阵 | `tests/test_tasks.py::TestTaskMatrix` | `sim/tasks.py::TaskSampler.capability_matrix/sample_matrix` | unittest 通过；40 支持/5 不支持 | ✅ |
| `TASK-DYNAMIC-001` | T5 事件载荷与环境不变 | `tests/test_tasks.py::TestTaskDynamicEvent` | `sim/tasks.py::DynamicObstacleEvent` | unittest 通过 | ✅ |

## 待人工确认

- 当前 5 个“不支持单元”为 S5/S6/S7 的 T4，以及 S8 的 T1/T4，来源是默认起点距离或车位数量；若论文实验要求严格填满全部 45 个单元，需要后续扩展对应场景尺寸/车位配置，而不能由任务层放宽 T1/T4 定义。
- 代表性的 T1–T5 任务均通过 Hybrid A* 集成回归；复杂 S9 长距随机样本仍可能触发规划器搜索上限，P2.8 应保留按任务单元计数的规划失败重采样与失败率统计。
