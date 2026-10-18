# MPC 求解器反事实 Spec

## 元数据

- 前缀：`MPC-PROBE`；强度：轻量；状态：实验；日期：2026-10-18。

## 目标

隔离有限采样求解对无干预闭环的影响。

## 非目标

不修改部署控制器、网络、轨迹参考、碰撞和成功阈值；不使用障碍代价、安全过滤或专家回退。不将实验性控制器成绩当作原协议成绩。

## 边界与约束

使用现有矩阵评测与固定基线参数，实验候选位于脚本，不对生产控制器增加配置。单进程调用内替换工厂，禁止多线程并发调用同一进程的矩阵评测。所有运行在指定 conda 环境，输出目录不得覆盖。

## 行为与验收

### `MPC-PROBE-001`：单变量求解器

保留现有 MPC 动力学、进度、目标函数与计算参数。incumbent 仅增加当前均值/历史最优候选并按同目标选择输出；correlated 仅将白噪声改为相邻步相关系数0.8的平稳高斯噪声，仍返回精英均值。支持 baseline 原实现。输出限幅且同 seed 可复现；incumbent 不得返回比输入热启动更高代价的序列。

### `MPC-PROBE-002`：隔离与证据

实验仅在一次评测调用内替换 MPC 工厂，成功或异常均恢复。读取指定已完成无干预 E5 报告的相同数据、模型、seed、步数与重规划间隔；索引必须唯一且属于基线。记录模型和数据哈希、候选算法及固定参数、配对结果和非主验收声明，拒绝覆盖已有输出。

### `MPC-PROBE-003`：逐级验证

先执行固定已知回归或预先声明子集，对照任务身份与基线结果；仅可信改善后运行完整集。控制器变化必须显式标记，不能宣称纯网络学习能力提高。无收益时不替换部署。

## 追溯

| ID | 验收入口 | 实现 | 实际验证 | 状态 |
|---|---|---|---|---|
| MPC-PROBE-001 | tests.test_mpc_solver_probe.TestSolverCandidates | scripts/probe_mpc_solver.py::IncumbentMPC、CorrelatedMPC | 零代价热启动保留、代价非增、确定性/限幅和噪声尺度相关性测试通过 | ✅ |
| MPC-PROBE-002 | test_factory_restored_on_exception、test_changed_control_period_is_rejected_before_output；实际回归 | isolated_controller、validate_probe_indices、run_probe | 工厂与引擎异常恢复/错误索引/控制周期测试通过，baseline复现69/69且所有候选身份一致 | ✅ |
| MPC-PROBE-003 | p3-*-regression*/paired.json；test_experimental_controller_report_is_rejected | run_probe；scripts/compare_selection_candidates.py::compare_reports | 两个候选无可信收益，未扩大/采用；固定协议选型拒绝反事实测试通过 | ✅ |
