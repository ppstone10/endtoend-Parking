# 滚动闭环引擎 Spec

## 元数据

- Spec ID 前缀：`LOOP`
- 强度：轻量
- 状态：已采纳
- 最后更新：2026-08-23

## 目标

- 统一执行"轨迹源 → MPC → 车辆"滚动闭环，产出完整回合指标（论文 8 项指标中单回合可度量部分）。
- 支持专家轨迹源（地基基线）与端到端网络源（感知→BEV→网络→MPC 主线）在同一口径下对比。

## 非目标

- 不做轨迹生成本身（由 sources 供给）与可视化回放（viz/ 负责）。
- 不做动态障碍注入（T5 属 M2 任务层，引擎预留 replan_every 接口）。

## 边界与约束

- 轨迹源输出全局坐标轨迹；引擎、MPC、车辆状态均为全局坐标。
- 到达判定为位置+航向双阈值（默认 0.3m / 10°，随目标车位容差配置）。
- 碰撞检测使用车辆矩形四角 + `env.is_free`。
- 失败分类优先级：碰撞 > 振荡 > 位姿超差 > 超时。

## 行为与验收

### `LOOP-ENGINE-001`：闭环回合执行

- 前置：起始 `VehicleState`、目标 `GoalPose`、轨迹源、MPC、车辆模型。
- 行为：每 replan_every 个控制周期向轨迹源取轨迹（K=1 逐周期），MPC 跟踪并推进车辆；到达/碰撞/超时/振荡即终止。
- 结果：返回 `EpisodeResult`（成功标志、失败分类、最终位置/航向误差、路径长度、泊车时间、跟踪 RMS、推理耗时）。
- 验收：`tests/test_runtime.py::TestClosedLoopEngine` 通过。

### `LOOP-TERM-001`：双阈值到达判定

- 前置：`TerminalChecker(tol_pos, tol_yaw)`。
- 行为：位置与航向同时低于阈值才判到达；航向差跨 ±pi 正确回绕。
- 结果：到达/未到达布尔判定。
- 验收：`tests/test_runtime.py::TestTerminalChecker` 通过。

### `LOOP-FAIL-001`：失败自动分类

- 前置：回合终止但未到达。
- 行为：碰撞 → "collision"；速度符号翻转远超参考轨迹方向切换数 → "oscillation"；接近但未达标 → "pose_error"；其余 → "timeout"。
- 结果：`failure` 字段为四类之一。
- 验收：`tests/test_runtime.py`（timeout/collision/oscillation 用例）通过。

### `LOOP-SRC-001`：轨迹源接口

- 前置：`begin(start, goal)` 后调用 `next_trajectory(state)`。
- 行为：ExpertSource 回合内规划一次复用；NetworkSource 每次调用重感知（capture_bev）并推理，局部轨迹转全局。
- 结果：返回 (全局坐标 Trajectory, 耗时 ms)。
- 验收：`tests/test_runtime.py::TestNetworkSourcePlumbing` 通过（局部→全局坐标变换正确性）。

### `LOOP-GROUND-001`：地基基线（M1 出口判据）

- 前置：专家轨迹源 + CEM-MPC（collision_margin=0.15），矿卡 6×3m，随机采样无碰撞位姿对（距离 3~12m）。
- 行为：`experiments/run_experiment.py --config experiments/configs/ground_baseline.json` 批量执行 200 回合。
- 结果：成功率 ≥ 95%。
- 验收：2026-08-23 实测 197/200 成功（98.5%），碰撞率 1.5%，位置误差 0.24±0.71m，航向误差 7.7°，跟踪 RMS 0.037m，耗时 183s。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `LOOP-ENGINE-001` | 回合执行与指标 | `tests/test_runtime.py::TestClosedLoopEngine` | `runtime/engine.py::ClosedLoopEngine` | unittest 通过 | ✅ |
| `LOOP-TERM-001` | 双阈值判定 | `tests/test_runtime.py::TestTerminalChecker` | `runtime/termination.py::TerminalChecker` | unittest 通过 | ✅ |
| `LOOP-FAIL-001` | 失败分类 | `tests/test_runtime.py` 各失败用例 | `runtime/termination.py::classify_oscillation`、`runtime/engine.py` | unittest 通过 | ✅ |
| `LOOP-SRC-001` | 轨迹源接口 | `tests/test_runtime.py::TestNetworkSourcePlumbing` | `runtime/sources.py` | unittest 通过 | ✅ |
| `LOOP-GROUND-001` | 专家+MPC ≥95% | `experiments/run_experiment.py` ground_baseline | 引擎全链路 | 197/200 成功（98.5%） | ✅ |

## 待人工确认

- 无。
