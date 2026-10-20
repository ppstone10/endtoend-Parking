# 固定候选闭环净收益试验

## 元数据

前缀：`CL-BENEFIT`；强度：轻量；范围：离线评测和人工长任务交接，不改训练/运行时/模型。

## 目标

以真实无干预E5成功/碰撞配对结果结束一次候选试验，不以loss或局部数学条件代替。只比较固定R0与既有R2 epoch2在原独立选型300条上的表现。

## 非目标

不重训、不扫候选/阈值、不修改旧失败结论、不自动进入主300或部署，不宣称这是未触碰最终测试集。

## 边界与约束

新增离线实验入口，复用E5 runner和既有冻结/人工执行模式；不修改公共训练或运行时契约。正式批量评测交用户，旧训练/数据/报告只读。

## 行为与验收

### CL-BENEFIT-001：固定身份与拒绝覆盖

prepare固定父/候选checkpoint、数据与manifest、来源训练报告/旧回归摘要及执行代码/环境SHA；校验两模型模型配置和停止阈值一致，候选为训练存档epoch2。训练来源与选型任务/恢复来源身份无交集，非S9，完整300条。运行必须显式提供config SHA；任何身份漂移拒绝，已有输出拒绝覆盖。

### CL-BENEFIT-002：等价完整闭环

正式每臂完整索引0～299，E5/sensor/network/mpc_vehicle/safety none，原MPC dt.1/horizon10，seed0、每10步重规划、上限600步。两报告除checkpoint外所有protocol、车辆与实验配置相同；逐任务task_id/scene/容差相同；没有复原失败、重复、缺失或非法/非有限结果。按逐回合计数复核overall，不因一臂失败缩分母。

### CL-BENEFIT-003：收益分类而非自动准入

始终报告两臂成功/碰撞/全部其他失败、配对转移、新增/丢失/保留成功，以及全场景计数和净退化列表。成功数严格增加且碰撞严格减少才有联合收益；固定+9成功/−9碰撞为有意义阶段收益线，弱于该线只报小收益。≥70%/≤10%为选型集门槛而非主协议达标。任何场景净成功降低或净碰撞增加均显式标记须审查；不要求逐样本全保留，也不静默掩盖回归。无收益则该候选结束，不追加参数搜索。

### CL-BENEFIT-004：短预检与有限交接

check-only不调用评测；smoke只取预先按S3/S4/S8各一条的固定任务、每臂最多20步，不得通过正式收益/目标门槛。正式仅600回合，逐臂保存、进度可见、异常写失败状态并停止；只可显式resume复用身份一致且SHA核验通过的完整臂，禁止自动重试/覆盖/续跑候选网格。

### CL-BENEFIT-005：人工决策

长任务计划列明命令/资源/产物/异常与恢复/验收。Agent仅prepare、check-only和短smoke，不代跑正式任务；完成后须核验全部产物才答复三类结论及分场景风险。即使选型通过，也不能自动部署、更新主成绩或运行下阶段。

## 追溯

| ID | 实现/验收 | 状态 |
|---|---|---|
| CL-BENEFIT-001 | scripts/validate_closed_loop_benefit.py::prepare/load_config；tests/test_closed_loop_benefit.py | ✅ 真实prepare/check-only、模型/训练来源/104源7输入冻结及覆盖/SHA拒绝测试通过 |
| CL-BENEFIT-002 | experiments/closed_loop_benefit.py::validate_report/summarize_pair；CLI::validate_frozen_report；results/independent-review.md | ✅ 正式两臂完整300条/全部源与报告SHA/配对协议核验通过；R0全部结果与步数精确复现存档、终点误差差0 |
| CL-BENEFIT-003 | 同上；ClosedLoopBenefitTests；LT14正式report与independent-review | ✅ 实际213/79对200/85，联合收益+13/−6；未达阶段双项线和目标，S4/S5回归，决定不晋升/不扩大/不追加旧配方；证据完整不代表目标通过 |
| CL-BENEFIT-004 | CLI::run；BenefitRunnerTests；preflight/report.json | ✅ check-only零执行、冒烟不晋升、异常记录/显式复用完整臂/拒绝篡改与重复测试通过，真实每臂3×20步smoke完成；真实中断恢复未触发 |
| CL-BENEFIT-005 | docs/long_tasks/LT-20261020-14-closed-loop-benefit.md；results/independent-review.md | ✅ 人工交接与正式结果复核闭环已完成，无自动下一阶段/训练/部署；本轮只读核验，主300成绩不变 |
