# 旧数据轮式审核与可解性校准 Spec

## 元数据

- Spec ID 前缀：`WHEEL-DATA`
- 强度：完整
- 状态：已采纳
- 最后更新：2026-11-04

## 目标

- 在不改写旧数据的前提下量化轮式必要条件缺口，冻结分层源任务，校准轮式专家及MPC是否能提供合格教师。
- 当前交付到可运行长任务计划，不在校准未通过前建设正式训练集。

## 非目标

- 静态必要条件通过不等于标签可直接复用；未知初始转角、停留段转角、碰撞与实际闭环仍需审核。
- 不触碰保留test，不混入恢复数据，不训练，不自动运行正式批校准或下一阶段。

## 边界与约束

- 只读取旧NPZ的轨迹、mask、状态、任务元数据和dt，避免加载BEV大数组；原始train/val分母与任务顺序保持。
- 几何按世界系中心转换到新后轴，以显式float32量化容差检查后轴侧向残差、曲率/速度、采样变化率、未停车换向与车体原地旋转。每种失败允许重叠计数。
- 无新配置身份的旧标签永不被静态报告自动授予轮式训练资格。

## 行为与验收

### `WHEEL-DATA-001`：旧标签必要条件与分母

- 检查schema v2、有限轨迹与连续前缀mask、元数据条数、正dt；每条返回原因与necessary_conditions_only/incompatible，不产生qualified=true。
- 真实轮式模型生成的直线/弧线应通过必要条件；履带pivot与过小半径应被拒绝，不能仅检测原地旋转。

### `WHEEL-DATA-002`：冻结训练/验证源校准面板

- 每个split按scene×requested maneuver取固定最早源索引；source task ID、源NPZ/manifest、轮式配置和运行代码哈希冻结，禁止改模型身份标签或替换失败任务。
- 原任务通过旧manifest严格复原，只复用几何与目标；初态v=omega=0且新旧尺寸/余量一致才允许明确设轮式初始转角0，否则拒绝。该初始化是新实验假设，不是从旧omega恢复转角。
- test、恢复标签和同ID跨split不得混入。prepare/precheck无正式回合；run在新目录输出完整状态/请求及逐任务结果。

### `WHEEL-DATA-003`：教师与执行联合关口

- 对每任务重新轮式规划，核终端、原请求机动距离占比至少50%、原目标容差，执行原600步上限，不放宽失败条件。
- 原生轮式MPC加GT转移审查，保存预测误差、真实状态/控制与运行耗时；初态/实际转移由inspect重新审核。所有面板任务必须规划/机动/位姿到达和审计通过才放行下一阶段，否则停留本环节。
- 规划候选失败只说明该算法与预算未找到解；实际安全停止或程序异常不算泊车成功。

### `WHEEL-DATA-004`：长任务及产物核验

- 正式run由用户运行，输出禁止覆盖，提供prepare/precheck/run/inspect与日志/退出记录。不存在缺失回合时才称complete；质量失败仍可形成完整负结果。
- 配置与逐例文件SHA绑定；inspect核分母、模型、逐控制重放与GT余量、终端及机动门槛，不依赖单个汇总数字；缺失/漂移拒绝，未知异常使作业显式失败。

## 数据、接口与迁移

- 输出是校准证据，不是新训练NPZ。原始文件只读；新模型身份与源任务身份分别保存，不能覆盖原metadata中的履带车型声明。
- 旧评测/训练/恢复入口尚未接入轮式数据，必须在校准与新数据schema完成后单独迁移。
- 无硬件操作或外部服务。中断保留现场；本版不自动续跑/重试或挑选剩余任务，重跑需新目录并依新计划处理。

## 追溯

| Spec ID | 验收 | 测试或人工入口 | 实现符号 | 实际验证 | 状态 |
|---|---|---|---|---|---|
| WHEEL-DATA-001 | 正反必要条件及原分母 | tests.test_wheeled_data；scripts/audit_wheeled_dataset.py | dataset/wheeled_audit.py | train2400/val300只读审核完成；float32量化及JSON回归通过 | ✅ |
| WHEEL-DATA-002 | 冻结源/面板与身份拒绝 | tests.test_wheeled_data；scripts/calibrate_wheeled_tasks.py --phase precheck | verification/wheeled_calibration.py::prepare_config/precheck | 22任务/333文件冻结通过；改分母/文件漂移拒绝 | ✅ |
| WHEEL-DATA-003 | 面板联合教师/执行结果 | 用户正式run，LT-20261104-45 | verification/wheeled_calibration.py::run_case | 真实首例短检通过，完整22正式任务未运行 | ⏳ |
| WHEEL-DATA-004 | 产物与逐控制独立审核 | tests.test_wheeled_data；用户正式inspect | scripts/calibrate_wheeled_tasks.py::inspect_formal；inspect_case | 状态/控制篡改、缺产物与负结果核验测试通过；正式inspect待运行 | ⚠️ |

## 待人工确认

- 正式校准待用户运行；真实矿卡参数仍未标定。现有RS有限候选覆盖是否足够生成新监督，是本次校准要回答的问题。
