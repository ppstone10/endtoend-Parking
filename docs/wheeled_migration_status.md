# 轮式改造分环节状态

本文件记录各环节交付状态和下一项依赖。稳定行为契约归属对应Spec，研究背景见[改造清单](reviews/2026-11-04-wheeled-migration-change-plan.md)。用户要求每项独立验证，出现问题先在当前项定位，不把失败带入后续环节。

| 环节 | 范围 | 状态 | 通过依据/下一依赖 |
|---|---|---|---|
| 1 | 轮式契约、配置、中心/后轴转换与运动模型 | 完整完成 | WHEEL-BASE-001～006；新25项+旧27项通过，任务T-132604 |
| 2 | 轮式MPC和执行/评测/选型/恢复采集入口 | 未开始 | 环节1通过；预测与实际推进必须一致 |
| 3 | 轮式专家规划、后处理及合法安全/终端回退 | 未开始 | 环节2通过；专家短闭环先于数据建设 |
| 4 | 旧标签轮式审核、场景可解性与必要重建 | 未开始 | 环节3通过；长任务另交付计划由用户运行 |
| 5 | 轮式网络基线与统一验证 | 未开始 | 环节4提供合法数据及身份；正式训练/批评仍交用户 |
| 6 | 履带受限适配与实物功能验证 | 未开始 | 软件主线通过；需要真实硬件参数与标定 |

## 当前工作边界

本轮完成环节1。旧履带默认配置、数据、模型和实验入口保留；尚无轮式闭环、训练收益、碰撞或真实车辆结论。研究配置不代表真实矿卡参数。下一环节不得直接套用旧差速MPC或原地终端转向。

## 环节1入口与验证

- 配置：`sim.wheeled_config.load_wheeled_vehicle_config('configs/vehicles/wheeled_research_v1.json')`；全部参数标记`research_assumption`，含完整模型/参考点/执行身份。
- 状态与指令：从`interfaces.wheeled`导入`WheeledVehicleState`和`WheeledControlCmd`；构造`sim.wheeled_vehicle.KinematicBicycleModel(config)`后调用`step(state,command,dt)`。这是显式基础API，尚不能放进旧ClosedLoopEngine。
- 最终新测试：指定conda的`python.exe -X utf8 -m unittest tests.test_wheeled_vehicle`，25项通过，0.405秒；此前与`tests.test_vehicle_model_acceptance tests.test_sim tests.test_interfaces`聚合51项通过（当时新24+旧27）。后续仅补新模型测试，未重复旧回归。
- 问题与定位：最初两项文件测试因系统临时目录写入/清理被Windows沙箱拒绝；改为模拟读取JSON载荷，真实示例文件仍由各配置测试直接读取。未改产品权限或模型门槛。另保留极小非零目标的数值回归，达到目标时直接返回目标，避免浮点相消误归零。

## 下一环节

环节2：建立轮式MPC预测和执行接入。必须显式携带转角状态、使用本模型的同一离散约定，并处理旧五维状态与新五维状态语义不同的问题；先核预测/执行单步一致，再做短闭环。未开始正式数据建设、训练或硬件适配。
