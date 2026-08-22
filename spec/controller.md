# MPC 轨迹跟踪控制器 Spec

## 元数据

- Spec ID 前缀：`CTRL`
- 强度：轻量
- 状态：已采纳
- 最后更新：2026-08-21

## 目标

- 输入未来参考轨迹与当前车辆状态，输出差分驱动控制指令 `[v, omega]`。
- 通过滚动时域优化实现轨迹跟踪，使车辆沿参考轨迹行驶并逼近终点。

## 非目标

- 不做障碍物避让或安全约束（轨迹层已由规划/网络保证）。
- 不优化全局路径，只做局部跟踪。
- 不保证大幅转向（>180°）的快速收敛，收敛速度受车辆动力学限制。

## 边界与约束

- 差分驱动运动学模型：`x' += v*cos(yaw)*dt`、`y' += v*sin(yaw)*dt`、`yaw' += omega*dt`。
- 控制量按 `max_v`/`max_omega` 限幅；参考点从距离当前状态最近的轨迹点起取预测时域。
- 纯 numpy 实现，数值梯度下降求解，无 scipy 依赖。

## 行为与验收

### `CTRL-MPC-001`：控制指令输出

- 前置：参考 `Trajectory` 与当前 `VehicleState`。
- 行为：取最近参考点后的预测时域，优化控制序列，返回序列首项。
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

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| `CTRL-MPC-001` | 指令在限幅内 | `tests/test_mpc.py::TestMPCController` | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |
| `CTRL-TRACK-001` | 直线收敛 | `tests/test_mpc.py` 闭环测试 | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |
| `CTRL-TURN-001` | 转向响应 | `tests/test_mpc.py` 转向测试 | `controller/mpc.py::MPCController` | unittest 通过 | ✅ |

## 待人工确认

- 无。