# 轮式MPC与独立闭环执行 Spec

## 元数据

- Spec ID 前缀：`WHEEL-EXEC`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-11-04

## 目标

- 以第1环节单轨模型为唯一运动预测/执行约定，提供显式轮式参考、CEM跟踪与短闭环验证。
- 通过当前环节门槛后进入专家规划；不隐式修改旧履带入口。

## 非目标

- 不直接读取旧NPZ、加载旧checkpoint或宣称生产数据/网络/恢复采集已轮式兼容。
- 本环节不保证任意参考可执行、全局收敛、实车实时性或碰撞安全；车体几何检查与专家规划另行接入。

## 边界与约束

- 世界系车体中心参考为只读float64 N×3数组，时间步有限正数，绑定64位SHA256车型身份；禁止接受旧Trajectory充当轮式参考。
- MPC请求(v,steering_angle)，完整状态含实际转角；所有候选rollout调用同一KinematicBicycleModel.step，不复制差速或单轨公式。
- 返回最佳已评估候选，不能因均值未经验证而直接发出。种子、配置与reset可复现。
- 暂以独立轮式运行时为权威入口；旧ClosedLoopEngine、旧评测、选型与恢复采集继续履带语义，轮式旧NPZ接入留在数据契约环节。

## 行为与验收

### `WHEEL-EXEC-001`：显式参考与拒绝错配

- 参考必须非空、有限、N×3、只读复制，并绑定车型身份和dt；模型身份/对象类型不一致必须在发出控制前报错，执行器与控制器dt必须一致。参考dt可重采样，角度需unwrap再插值，保持真实时间间隔。

### `WHEEL-EXEC-002`：预测与实际推进一致

- 同一状态、请求序列、dt下，MPC rollout逐步与独立模型step一致；速度/转向限制与换向不跨零由基础模型统一执行。
- 保存选中请求序列及完整预测状态；执行首步与预测首步一致。失败不提交新状态，不自动回退差速控制。

### `WHEEL-EXEC-003`：确定性跟踪

- CEM保留原思路，比较采样、均值、移位热启动、几何前馈和停车候选，沿用同一跟踪代价；只选择实际计算过的有限代价结果。
- 参考进度按控制时间推进，新参考重新初始化；reset同时恢复随机种子和控制上下文。短测覆盖直线、倒车、转弯以及定位偏差，达到既定位置/航向门槛。

### `WHEEL-EXEC-004`：独立闭环、失败与记录

- 独立入口保留真实转角，支持固定或周期重新查询轮式参考，按原位置/航向双门槛判位姿到达；不把它称为停稳。
- 结果含配置身份、实际状态/请求、预测一致误差、到达/超时/规划失败/安全停止分类及控制耗时。可注入转移检查器，在接受状态前拦截不合格转移。
- 初态已到达时零控制结束；规划失败明确记录；未知程序异常不能变成正常停车。没有几何检查器时报告未检查碰撞，而非零碰撞。

## 数据、接口与迁移

- 新WheeledTrajectory明确world_body_center语义；旧Trajectory和VehicleState维度相同不构成兼容依据。
- 模型由外部注入控制器/执行器；接口与controller不反向依赖sim实现。运行时组合负责核对身份与dt。
- 本环节无旧数据迁移/覆盖；停用显式轮式入口即可回退旧基线。未接入硬件或外部服务。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| WHEEL-EXEC-001 | 参考身份/类型/dt及重采样 | tests.test_wheeled_execution | interfaces.wheeled.WheeledTrajectory；WheeledMPCController.align_reference | 聚合88项通过，含类型/身份/时间与角度重采样 | ✅ |
| WHEEL-EXEC-002 | 完整预测/执行一致 | tests.test_wheeled_execution | WheeledMPCController.rollout；WheeledExecutor.propose | 完整序列逐步一致、实际首步误差0；新旧对象拒绝 | ✅ |
| WHEEL-EXEC-003 | 确定性与短跟踪 | tests.test_wheeled_execution；scripts/run_wheeled.py --mode tracking | controller/wheeled_mpc.py | reset确定性、偏差/直线/倒车/弯曲短测通过 | ✅ |
| WHEEL-EXEC-004 | 闭环失败、拦截与结果 | tests.test_wheeled_execution | runtime/wheeled_execution.py；experiments/wheeled_evaluation.py | 规划失败/异常/转移拦截/资源预算/原容差传递通过 | ✅ |

## 待人工确认

- 真实车型参数尚未提供，继续使用research_assumption配置；不是本环节解析/短仿真验证的阻塞，但实车结论仍不授予。
