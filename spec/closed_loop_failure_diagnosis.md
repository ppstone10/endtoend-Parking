# 闭环失败回放诊断 Spec

## 元数据

- 前缀：`LOOP-DIAG`；强度：轻量；状态：已采纳；日期：2026-10-18。

## 目标

为自主闭环执行计划 P0/P1 提供可重放、时间对齐的失败证据。

## 非目标

修改网络、MPC、碰撞/成功判据；以整条参考的远端穿障直接证明当前碰撞原因。

## 边界与约束

沿用现有评测编排、numpy/JSON 落盘及完整车体检查器；观察挂接不改变传感器 RNG 或控制器推进，只支持无干预 E5。

## 行为与验收

### `LOOP-DIAG-001`：真实初态和实际控制参考

记录真实初态及每个控制转移的前后状态、命令、MPC 实际使用的参考窗口。第一步前态必须是传入引擎的初态，不能使用 recorder 的首个推进后状态代替。保留起始和最后一次推理的 BEV、各次重规划轨迹以及控制周期。

### `LOOP-DIAG-002`：候选机制与证据边界

分别检查初态、MPC 实际参考窗口、碰撞时生效的完整网络参考及实际转移。输出首次不安全参考窗口、首次碰撞、碰撞前最后具有规划余量的状态；记录为候选机制，不把远端参考穿障当作当前碰撞的充分因果证据。无碰撞和零步记录必须能处理。

### `LOOP-DIAG-003`：基线身份与非干预

从指定基线报告提取失败任务，按可选场景筛选；校验报告的数据、模型和控制参数，按原协议执行无干预 E5。每条输出独立 JSON/无 pickle NPZ，批次输出完整索引、复原失败、基线成功/失败/步数匹配情况。诊断挂接必须在异常时恢复，且不额外推进传感器 RNG 或控制器状态。

## 追溯

| ID | 验收入口 | 实现 | 实际验证 | 状态 |
|---|---|---|---|---|
| LOOP-DIAG-001 | tests.test_failure_diagnosis.TestFailureDiagnosis.test_first_transition_uses_true_start；真实 S8 回放 | scripts/diagnose_closed_loop_failures.py::analyze_record、run_diagnosis | 39 项聚合测试通过；95 条 NPZ 初态/窗口与报告校验通过 | ✅ |
| LOOP-DIAG-002 | test_far_reference_does_not_mean_control_window_unsafe、test_empty_record | analyze_record | 单测通过；95 条真实失败输出候选机制，不宣称因果已闭合 | ✅ |
| LOOP-DIAG-003 | test_observation_hooks_restore_after_evaluation_error、test_failure_selection_keeps_all_failure_types_and_scene_filter | run_diagnosis、select_failure_indices | 95/95 回放与基线结果、步数及位置误差匹配；95 条无 pickle 归档可读 | ✅ |
