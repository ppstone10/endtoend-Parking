# MPC 轨迹跟踪控制器 Spec

## 元数据

- Spec ID 前缀：`CTRL`
- 强度：轻量
- 状态：已采纳
- 最后更新：2026-08-23

## 目标

- 输入未来参考轨迹与当前车辆状态，输出差分驱动控制指令 `[v, omega]`。
- 通过滚动时域优化实现轨迹跟踪，使车辆沿参考轨迹行驶并以正确位置与航向停在轨迹终点。

## 非目标

- 不做障碍物避障或安全约束（轨迹层已由规划/网络保证；碰撞检测由闭环引擎负责）。
- 不优化全局路径，只做局部跟踪。

## 边界与约束

- 差分驱动运动学模型：`x' += v*cos(yaw)*dt`、`y' += v*sin(yaw)*dt`、`yaw' += omega*dt`。
- 控制量按 `max_v`/`max_omega` 限幅；纯 numpy 实现（无 scipy 依赖）。
- 参考轨迹 dt 与控制周期不一致时按时间插值对齐（yaw 用 sin/cos 分量插值防跨 ±pi 跳变）。
- 进度锚定为轨迹时间轴而非车辆当前位置（位置锚定会使代价面在进度方向平坦，导致无恢复力与极限环振荡）。

## 行为与验收

### `CTRL-MPC-001`：控制指令输出

- 前置：参考 `Trajectory` 与当前 `VehicleState`。
- 行为：CEM（交叉熵方法）在预测时域内搜索控制序列（population 采样 → 批量 rollout → 精英拟合），上周期解平移一步热启动；返回序列首项。
- 结果：返回 `ControlCmd`（v, omega），均在限幅内。
- 验收：`tests/test_mpc.py::TestMPCController` 通过。

### `CTRL-TRACK-001`：闭环跟踪收敛

- 前置：直线参考轨迹（0→5m）。
- 行为：MPC 与差分模型闭环推进。
- 结果：车辆到达 x ≥ 4m，侧向偏差 < 0.3m。
- 验收：`tests/test_mpc.py::TestMPCController.test_closed_loop_straight_tracking` 通过。

### `CTRL-TURN-001`：转向轨迹响应

- 前置：朝正 y 方向弯曲的参考轨迹。
- 行为：MPC 输出非零 omega。
- 结果：omega 朝轨迹弯曲方向。
- 验收：`tests/test_mpc.py::TestMPCController.test_turning_trajectory` 通过。

### `CTRL-REV-001`：倒车轨迹跟踪

- 前置：S 形倒车参考轨迹（航向左右摆动，含方向反转）。
- 行为：MPC 闭环推进至轨迹终点。
- 结果：终点位置误差 < 0.3m，航向误差 < 15°。
- 验收：`tests/test_mpc.py::TestMPCController.test_closed_loop_reverse_s_tracking` 通过。

### `CTRL-RECOVER-001`：扰动恢复

- 前置：直线参考轨迹，初始横向偏差 0.5m。
- 行为：MPC 闭环推进。
- 结果：横向偏差收敛 < 0.3m 且继续前进至终点附近。
- 验收：`tests/test_mpc.py::TestMPCController.test_disturbance_recovery` 通过。

### `CTRL-TERM-001`：终态位姿对齐

- 前置：单点参考轨迹（终点含航向）。
- 行为：MPC 闭环推进，终态强惩罚（位置+航向）驱动收敛。
- 结果：位置误差 < 0.3m 且航向误差 < 20°（原地转向收敛）。
- 验收：`tests/test_mpc.py::TestMPCController.test_terminal_pose_alignment` 通过。

### `CTRL-DT-001`：dt 对齐

- 前置：轨迹 dt=0.2，控制周期 dt=0.1。
- 行为：参考轨迹按时间插值重采样后跟踪。
- 结果：直线跟踪收敛不受 dt 失配影响。
- 验收：`tests/test_mpc.py::TestMPCController.test_dt_mismatch_alignment` 通过。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `CTRL-MPC-001` | 指令在限幅内 | `tests/test_mpc.py::TestMPCController` | `controller/mpc.py::MPCController` | unittest 8 项通过 | ✅ |
| `CTRL-TRACK-001` | 直线收敛 | `tests/test_mpc.py` 闭环测试 | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |
| `CTRL-TURN-001` | 转向响应 | `tests/test_mpc.py` 转向测试 | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |
| `CTRL-REV-001` | S 形倒车收敛 | `tests/test_mpc.py` 倒车测试 | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |
| `CTRL-RECOVER-001` | 扰动恢复 | `tests/test_mpc.py` 扰动测试 | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |
| `CTRL-TERM-001` | 终态位姿对齐 | `tests/test_mpc.py` 终态测试 | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |
| `CTRL-DT-001` | dt 对齐跟踪 | `tests/test_mpc.py` dt 失配测试 | `controller/mpc.py::_align_dt` | unittest 通过 | ✅ |

## 待人工确认

- 无。
