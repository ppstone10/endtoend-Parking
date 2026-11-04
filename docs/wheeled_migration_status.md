# 轮式改造分环节状态

本文件记录各环节交付状态和下一项依赖。稳定行为契约归属对应Spec，研究背景见[改造清单](reviews/2026-11-04-wheeled-migration-change-plan.md)。用户要求每项独立验证，出现问题先在当前项定位，不把失败带入后续环节。

| 环节 | 范围 | 状态 | 通过依据/下一依赖 |
|---|---|---|---|
| 1 | 轮式契约、配置、中心/后轴转换与运动模型 | 完整完成 | WHEEL-BASE-001～006；新25项+旧27项通过，任务T-132604 |
| 2 | 轮式MPC、独立执行与组合根 | 基础关口通过；旧数据入口适配待4/5 | WHEEL-EXEC-001～004；13项执行测试通过，旧NPZ选型/恢复/训练不自动兼容 |
| 3 | 轮式专家、离散时序与合法转移检查 | 有限RS专家基础关口通过；覆盖待校准 | WHEEL-PLAN-001～004；9项测试与转弯CLI短检通过，无全局搜索完备性结论 |
| 4 | 旧标签轮式审核、场景可解性与必要重建 | 静态审核完成；正式22任务校准待用户运行 | train2400/val300均不可直接复用；LT-20261104-45待运行，未重建训练集 |
| 5 | 轮式网络基线与统一验证 | 未开始 | 环节4提供合法数据及身份；正式训练/批评仍交用户 |
| 6 | 履带受限适配与实物功能验证 | 未开始 | 软件主线通过；需要真实硬件参数与标定 |

## 当前工作边界

已从环节1自动推进到环节4。独立轮式基础、CEM执行及有限RS专家短闭环可用；旧履带默认入口保持。当前停在正式教师校准与数据迁移关口，尚无新训练集、网络学习收益、主300或真实车辆结论。参数仍为研究假设，不能把有限RS候选失败当全局无解。

## 环节1入口与验证

- 配置：`sim.wheeled_config.load_wheeled_vehicle_config('configs/vehicles/wheeled_research_v1.json')`；全部参数标记`research_assumption`，含完整模型/参考点/执行身份。
- 状态与指令：从`interfaces.wheeled`导入`WheeledVehicleState`和`WheeledControlCmd`；构造`sim.wheeled_vehicle.KinematicBicycleModel(config)`后调用`step(state,command,dt)`。这是显式基础API，尚不能放进旧ClosedLoopEngine。
- 最终新测试：指定conda的`python.exe -X utf8 -m unittest tests.test_wheeled_vehicle`，25项通过，0.405秒；此前与`tests.test_vehicle_model_acceptance tests.test_sim tests.test_interfaces`聚合51项通过（当时新24+旧27）。后续仅补新模型测试，未重复旧回归。
- 问题与定位：最初两项文件测试因系统临时目录写入/清理被Windows沙箱拒绝；改为模拟读取JSON载荷，真实示例文件仍由各配置测试直接读取。未改产品权限或模型门槛。另保留极小非零目标的数值回归，达到目标时直接返回目标，避免浮点相消误归零。

## 环节2/3入口与验证

- 显式运行入口：`scripts/run_wheeled.py --mode tracking --case all`及`--mode expert --case arc`，均使用指定conda Python。前者三例不做几何检查；后者为GT辅助专家+轮式MPC，不能互换成绩口径。
- 组合根`experiments/wheeled_evaluation.py::build_wheeled_engine`提供完整轮式状态、模型身份/dt核对、原车位容差传递与资源预算。MPC rollout统一调用基础model.step；不向旧ClosedLoopEngine塞入转角数组。
- 专家基础用后轴RS候选，每段停车、转轮、离散加减速后由模型重放；所有真实转移按车体余量、弓高及采样保守膨胀审核。没有履带原地朝向接管，没有全局障碍搜索完备性或时间最优结论。
- 聚合验证88项通过（5.215秒）：轮式基础25、执行13、专家9、数据9及旧MPC/执行/RS/车辆32项。真实转弯短检177步位姿到达、位置0.273110m、航向0.113274rad，预测首步误差0，GT检查通过；不是停稳/主表结果。
- 问题定位：最初新转弯测试140步，小于参考自身187步；修正为项目原600步并加入参考长度断言后通过，未更改正式协议。源码未接旧差速控制或原地接管以救分。

## 环节4证据与下一步

- 当前轮式假设配置下，旧train2400/val300均有速度变化必要条件缺口；其中855/105有后轴侧向/曲率缺口、59/9含车体pivot。失败原因可重叠计数。几何必要条件通过也不能代替转角/碰撞/闭环认证。
- 完整只读报告：`runs/verification/wheeled-legacy-audit-20261104-v2.json`，行分母2400/300已核对。无v2同名文件是NumPy float32容差未转原生float导致的导出失败现场，不能使用；实现已修并加入序列化/量化回归。
- 校准prepared：`runs/prepared/wheeled-calibration-20261104-v1/config.json`，22任务/333源码输入SHA冻结。首个真实train#0短检与逐控制复核通过，94步到达、方向比1.0，未运行正式面板。
- [LT-20261104-45可运行计划](long_tasks/LT-20261104-45-wheeled-teacher-calibration.md)已交付，正式run/inspect由用户运行。反馈后先核验完整分母、任务身份、GT安全、原容差与请求机动，再决定进一步专家验证及必要重建；不自动进入网络训练。
- 旧NPZ评测、checkpoint选型、恢复数据与正式训练入口仍为履带契约。它们的轮式接入待新数据schema和身份建立后单独完成，不能称本轮已全面兼容。
