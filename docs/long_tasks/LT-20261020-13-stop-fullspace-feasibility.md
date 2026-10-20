# LT-20261020-13：完整停止头空间受保护方向延长审计

状态：**已运行完成（2026-10-20 19:59–20:02，按第 1、2 步执行）**。无保护的 2 项/8 项范围得到严格共同下降见证；两项带逐样本保护的范围在 4096 次/120 s 内仍 `iteration_budget_exhausted`，未解出。不是训练，不生成模型，也不运行闭环。

独立复核已完成：[结果与边界](../../runs/diagnostics/stop-fullspace-extended-20261020-v1/results/agent-audit.md)。全部G/P/对偶量与两无保护见证重算通过；两受保护候选仍不通过。以下运行命令仅作为已完成任务记录，不要重复运行或增加预算。后续[固定有限步长检查](../../runs/diagnostics/stop-finite-step-20261020-v1/summary.md)也已完成：保留成立但无新增合格，不进入训练或闭环扩大。

## 目的与唯一主要变量

在完整129维线性停止头空间中，查找同时改善指定分项且保护原1144开环合格样本的方向。短检已找到无逐样本保护的8分项共同下降见证；严格/位姿兼容保护两项在256迭代内未解出，不能认定无解。

唯一主要变量是求解预算：256迭代/范围8秒 → **4096迭代/范围120秒**；保持同一代码、输入、损失分项、归一化、阈值和保护定义。不改训练权重、模型、标签、采样或部署。

## 固定身份和参数

- 工作目录：`D:\program\endtoend Parking`；环境：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12.13、torch2.13.0+cpu，CPU单线程。
- 入口：`scripts/audit_stop_fullspace.py`；核心：`experiments/stop_fullspace_feasibility.py`。
- 冻结配置：`runs/diagnostics/stop-fullspace-extended-20261020-v1/config.json`。
- config SHA256：`b645fa3f655e05fe598a53078b14402c3505993a98844cabcea1a81c41fc02df`。
- 输入：LT12固定config/report、R0父模型、2918 free缓存、原2400父样本记录；各路径/SHA和6份执行源摘要及运行时完整写在配置。缓存SHA `7ccb2b77f47cd21fe0bfab10d4d95d559a249062b2ef8d716e3a0e306b1bbba4`，父模型SHA `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`。
- H320、原阈值0.26000000000000006；位置0.3m/yaw10度仅用于固定训练初态保护诊断，不是新教师。
- 固定四范围：完整2918两分项；加入原GT索引合格1760/S4 430/S8 85的八分项；八分项+严格保护131446行；八分项+位姿兼容保护127712行。
- 固定数值门槛：单位方向全部保护残差≤1e−9、全部分项目标方向导数<−1e−8；小对偶范数只限制单位球共同改善裕量，不代表所有方向不可行。

## 已完成的短预检

40项聚合测试通过，另有1项冻结配置漂移/覆盖回归通过；短审计v2共15.619秒。完整池2项与分组8项共同下降见证均经独立NumPy重算；八分项最小单位方向下降裕量约0.003907，但该见证没有通过逐样本保护。两项保护状态为iteration_budget_exhausted，不是不可行。配置check-only通过。

## 用户运行命令（按顺序）

### 1. 固定目录与身份检查

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$taskPython = 'D:\conda\envs\endtoend-parking\python.exe'
$taskConfig = 'D:\program\endtoend Parking\runs\diagnostics\stop-fullspace-extended-20261020-v1\config.json'
$taskConfigHash = (Get-FileHash -LiteralPath $taskConfig -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskConfigHash -ne 'b645fa3f655e05fe598a53078b14402c3505993a98844cabcea1a81c41fc02df') { throw '冻结配置摘要不一致，请停止并反馈' }
& $taskPython -B scripts/audit_stop_fullspace.py --config $taskConfig --check-only
if ($LASTEXITCODE -ne 0) { throw '身份检查失败，请停止并反馈；不要绕过检查' }
```

### 2. 延长审计（仅一次）

```powershell
& $taskPython -B scripts/audit_stop_fullspace.py --config $taskConfig 2>&1 | Tee-Object -FilePath 'D:\program\endtoend Parking\runs\diagnostics\stop-fullspace-extended-20261020-v1\console.log' -Append
if ($LASTEXITCODE -ne 0) { throw '审计异常结束，请保留日志和产物并反馈' }
```

不要执行prepare、不要手改配置/源文件或并行启动重复任务；results目录一旦存在入口会拒绝覆盖。两个代码块须在同一PowerShell会话执行。

## 耗时与资源

依据15.619秒短检及迭代预算增加16倍，预计约2～5分钟；收敛速度、CPU/磁盘及系统负载存在不确定性。每范围硬时间预算120秒，四范围理论最多约8分钟，另加输入校验/构造矩阵/保存。超出预估而仍持续输出不等于失败，超过约10分钟且长时间无新输出可中断并反馈。

CPU单线程，无GPU；预计内存约2～4GB（500MB缓存、两份保护矩阵和临时float64张量），磁盘日志/证据通常数MB；不会保存模型checkpoint。系统内存不足、数值异常或身份漂移应停止，不自动重跑或放宽门槛。

## 输出、进度与完成标记

根目录：`runs/diagnostics/stop-fullspace-extended-20261020-v1/`。

- `config.json`：冻结输入/执行源/运行时/预算，禁止覆盖。
- `console.log`：每范围start/done、每64迭代的上界/保护残差。
- `results/status.json`：running/failed/completed；成功时含report SHA。
- `results/report.partial.json`：每范围完成后保留的证据，不可恢复运行，不是模型checkpoint。
- `results/report.json`：完整四范围、lambda/mu/活跃保护行/单位见证与指标。

正常结束应出现末行`completed ...report.json`、status.json为completed、report.status为completed_requires_design_review且四范围齐全。某范围仍为iteration_budget_exhausted或time_budget_exhausted也是审计正常完成，但数学问题仍未解决；不得改口为不可行或效果通过。

## 异常、中断和重跑

- 身份漂移：保留错误，交Agent核对，不改SHA、不关闭检查。
- 数值/内存异常：保留status、partial、console；不要跳过场景或删除保护行。
- Ctrl+C/系统中断可能留下running状态；这不表示仍在运行或已完成。报告终端退出信息，不根据status单独推断。
- 无自动恢复和重试；不从partial继续。重跑需Agent复核原因、准备新版本任务目录与冻结配置，禁止清空或覆盖现有results、旧短审计、LT12训练模型/缓存。

## 完成后反馈

请回复“完整停止头空间延长审计完成”，或反馈异常；保留`config.json`、`console.log`、`results/status.json`及完整/partial报告，无需手工摘取指标。若输出位置变化请说明。

## Agent后续验收

1. 核对任务配置、代码/输入/运行时、完成状态与四范围分母。
2. 对见证重新计算完整129维G与全部保护P，验证单位范数、全部分项目标方向及最坏保护残差，不只看平均数；独立重算对偶非负系数和单位裕量上界。
3. 若保护方向通过，再设计有界有限步长验证并检查原合格保留与新增合格；不能由方向导数直接宣称闭环改善或开正式训练。
4. 若仍预算耗尽，明确未解决，不无限自动加算；若小上界证据成立，仅报告它排除的单位改善裕量，区分严格共同下降与原非增条件。

主300仍为原68.3%成功/25.7%碰撞，≥70%/≤10%目标未达。本任务不替代几何、MPC执行与闭环联合验收。

## 运行结果（2026-10-20 19:59–20:02，按第 1、2 步执行）

- 第 1 步 check-only 通过；第 2 步一次运行完成，172.99 s，`results/report.json` 为 `completed_requires_design_review`，`results/status.json` 为 `completed`（`report_sha256=f08e3595…6ad35`，与文件实算一致）；`training_performed/model_exported/closed_loop_evaluated` 全 false。
- 身份：config `b645fa3f…`、LT12 config/report、R0、`cache-free.pt`（`7ccb2b77…`）、`parent/samples.json`（`83d000f7…`）全部匹配；预算 4096 次 / 120 s 每范围，阈值 0.26；分母 2918/1760/430/85、保护行 131446/127712。
- 结果：`aggregate_only` 与 `guarded_metrics` 均 **verified_common_descent**（单位见证，最小分项下降 0.104107 / 0.003907，后者与短检一致）；`strict_index_protection` 与 `goal_compatible_anchor_protection` 均 **iteration_budget_exhausted**（4096 次、84.5/81.8 s），无见证，候选最小下降 −0.000133 / −0.005057、保护残差 0.005322 / 0.006710，对偶界 0.006662 / 0.006942（活跃行 198/166）。
- 独立复核：两处见证 `‖w‖=1`、`G·w` 与报告斜率逐项一致（误差 ≤1.4e−17）、全部分项 < −1e−8；两个受保护范围的对偶系数全部非负，但保护残差未归零，0.00666/0.00694 不能当作已证书的严格上界。
- 结论：预算扩大 16 倍只确认了无保护范围严格可行；带逐样本保护的范围仍未解出，既非不可行也非通过。不追加算力、不改门槛或保护定义；不训练、不生成模型、不跑闭环。明细见 `runs/diagnostics/stop-fullspace-extended-20261020-v1/results/summary.md`。
