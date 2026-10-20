# LT-20261019-08：事件对齐与S8早期执行验证

## 状态、目的与边界

已运行完成（2026-10-20 09:37–09:38，Agent 代跑并已核验）：11/11 源完成，判定 `event_audit_only_no_training`；本页命令不再重复执行。不训练、不部署，不生成回灌标签。原R0、原实际E5、原MPC、几何/控制预算/阈值不变，不能重新启用已失败的候选求解器。

LT07有两条S4分叉晚于参考安全回退10步，且S8四条源在第一次重规划前已失余量。本计划分别检查事件时点与早期执行兼容性，不再用一个更新幅值或“零救回”统一归因。样本由LT07训练域诊断事后确定，是机制对照而非独立总体评测，不使用val/selection/S9案例或标签补挑。

唯一主要分叉变量是第一次返回哪条参考：正常网络、延迟一次（返回已有同一旧参考对象保持进度）、专家仅替换一次。三者都消费同次正常感知/网络推理；之后全部恢复正常网络重规划，绝非全程专家。相同车辆状态、MPC热启动/RNG和感知快照。专家失败保留planning_failure，不重试、不跳过来源。

## 固定来源、时点和预算

| 顺序 | train索引 | 分叉步 | 角色/事件 | 分支预算 |
|---:|---:|---:|---|---|
| 1 | 1298 | 210 | S4规划安全回退（比LT07前移10步） | 各390步上限 |
| 2 | 1455 | 180 | S4规划安全回退（前移10步） | 各420步上限 |
| 3 | 1559 | 310 | S4规划安全回退 | 各290步上限 |
| 4 | 1441 | 220 | S4接触安全回退 | 各380步上限 |
| 5 | 2364 | 10 | S8规划回退，连接两类机制 | 各590步上限 |
| 6 | 1334 | 60 | S4实际成功对照，但曾损失规划余量 | 各540步上限 |
| 7 | 1680 | 110 | S4成功对照 | 各490步上限 |
| 8 | 2348 | 0 | S8初始参考执行 | 仅10步 |
| 9 | 2330 | 0 | 同上 | 仅10步 |
| 10 | 2332 | 0 | 同上 | 仅10步 |
| 11 | 2317 | 0 | 同上 | 仅10步 |

前7源完整回放须复现LT07全部原状态/命令，然后从固定安全状态分叉，先正常完整后缀精确复现（1e-9）再做延迟/专家一次。回退类型与时点必须复现，缺状态/不匹配即停，不自动换点。预算600−事件步数，不放宽到达标准。

后4源仅复现LT07前10步，step0没有旧参考，只有正常/专家一次两分支；不得伪造延迟。10步不发生第二次网络重规划，明确`horizon_limited=true`、`full_task_evaluation=false`，无接触或无余量损失不等于任务到达/全程恢复。前7组是全程事件对照，后4组只是首段执行诊断，不能合并成总体成功率。

所有task_id及31个输入绝对路径/摘要逐项存于config，不因新结果调整来源。

## 冻结配置和环境

新根目录 `runs/diagnostics/reference-event-probe-20261019-v1/`，`config.json` SHA256：

`14b46ab57beb0ddc61da863572a6891da0a1181c20a32ec867a2a4d224a4659f`

冻结31输入：LT07原7输入、LT07配置与正式报告、所选11源各自report.json/trajectories.npz（22份）；另冻结91执行代码文件与运行环境。准备时逐项验证旧89源摘要不变，旧任务配置/报告不改写。新增核心改变代码全集，旧入口会拒绝旧身份重跑，不要改hash绕过。

关键身份：R0 SHA256 `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`；原2400任务train `8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f`；LT07 config `3474f667b06253515926546f4071087cf754c434cb505a61f6221d6e3175c211`。LT07任务NPZ须与该正式报告摘要相符，旧逐源报告须与总报告完全相同。

LT07正式report SHA256 `ca9733af9044ff3f426da3ea09605571ba640e8c30f5c138f7c0aa7fc5ce55f6`。审计额外计算已选序列的代价与几何用于记录，不增加采样、不参与控制选择，不把诊断开销当新求解器性能。

工作目录 `D:\program\endtoend Parking`；唯一Python `D:\conda\envs\endtoend-parking\python.exe`（3.12.13），NumPy2.5.2、torch2.13.0+cpu，CPU单线程，无新安装/GPU。原实际pipeline保持camera目标通道，不静默改checkpoint元数据；逐源初态BEV必须等于归档train。seed0+train索引、replan10、dt0.1/horizon10、父停止阈值约0.26、6×3m车辆、接触余量0/规划余量0.2m全部保持。

预计3～15分钟，专家首次规划耗时与后续网络反馈长度不定，不能按smoke线性承诺；建议4GB可用内存/1GB磁盘，非实测峰值。不要同时训练或修改执行源码。正式是多分支批量验证，由用户运行。

## 已完成的有限验证

43项聚合测试通过，覆盖一次专家后恢复网络、首轮仍消费感知RNG、无旧参考拒绝延迟、专家无解记账、私有solve观察非干预、正常完整后缀、预测首步与真实推进、事件时点/类型、源前缀与全程区别、固定来源/冻结身份/输出复用拒绝及短窗不算全程成功。实现前测试因探针模块尚缺失败，随后实现通过。

真实短预检5.875秒：#1298源220步，step210三分支各10步；#2348源与初态两个分支各10步。正常分支状态/命令误差均0，全部分支未到达、无接触。#1298三分支均无余量损失，但不是全程通过；#2348正常首损失step2/共4步，专家一次首损失step3/共2步，窗口/连接当时均规划安全，未得到严格安全执行。`decision=bounded_preflight_only`，`full_task_evaluated_sources=0`。不重跑smoke。

## PowerShell运行（同一窗口，按顺序）

### 1. 只读身份预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$eventPython = 'D:\conda\envs\endtoend-parking\python.exe'
$eventRoot = 'D:\program\endtoend Parking\runs\diagnostics\reference-event-probe-20261019-v1'
$eventConfig = Join-Path $eventRoot 'config.json'
$eventHash = '14b46ab57beb0ddc61da863572a6891da0a1181c20a32ec867a2a4d224a4659f'
if ((Get-FileHash -LiteralPath $eventConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $eventHash) { throw '配置摘要变化，停止反馈' }
& $eventPython -u -B scripts/probe_reference_events.py --config $eventConfig --config-sha256 $eventHash --check-only
if ($LASTEXITCODE -ne 0) { throw '输入/代码/环境校验失败，不继续' }
```

预期 `INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`，不要重新prepare或smoke。

### 2. 正式11源及适用分支

```powershell
if (Test-Path -LiteralPath (Join-Path $eventRoot 'results')) { throw '已有正式输出，禁止覆盖或重跑，先反馈' }
$eventLog = Join-Path $eventRoot 'console.log'
if (Test-Path -LiteralPath $eventLog) { throw '已有日志，禁止覆盖，先反馈' }
& $eventPython -u -B scripts/probe_reference_events.py --config $eventConfig --config-sha256 $eventHash 2>&1 | Tee-Object -FilePath $eventLog
if ($LASTEXITCODE -ne 0) { throw '失败/中断，保留产物，不自动重试' }
```

进度逐任务`COLLECT BEGIN`→`fresh BEGIN`→有旧参考时`delay_once BEGIN`→`expert_once BEGIN`→`COMPLETE`。先正常后缀通过才会开始反事实；step0没有delay_once是正常行为。正式完成以results/report.json `status=completed`、`smoke=false`、11条tasks为准，不以单个COMPLETE判断整批完成。

## 产物、停止、恢复与回传

根目录console.log；results/identity.json完整冻结身份；逐源task-NNNN/report.json保存旧源复现、各分支结果/规划失败/专家尝试与应用计数、全程/短窗边界、逐次参考更新；trajectories.npz保存原源与分支状态/命令、原网络参考、上下文；fresh/delay_once/expert_once-audit.json保存每控制步原MPC实际窗口、控制序列、rollout、代价、首步/窗口几何及同xy换yaw/参考xy换实际yaw的**几何**探针。所有NPZ和审计文件摘要保存在任务report中。

results/report.json汇总分开事件组与10步早期组，正式decision仍是`event_audit_only_no_training`。专家一次applied应≤1，规划失败允许attempts1/applied0，不能把失败当跳过；延迟组恰好一次，初态组必须没有延迟。

漂移、初态BEV不符、历史源或正常后缀不匹配、事件缺失立即停止；异常写status.json并保留已完成任务，Ctrl+C也保留，强杀可能无status.json。无自动resume/retry，不删除既有results、不改hash/参数/余量、不补挑源；反馈后另定新目录与剩余任务计划。

完成后回复“事件对齐与早期执行验证完成”；失败则附控制台末尾和status.json。Agent核验31输入/91源、11账本、7组全程正常后缀和4组10步前缀、所有实际推进/审计/计数；再判断早期事件处一次替换能否严格恢复、是否伤害两条S4成功对照，以及S8安全窗口仍执行越界的xy/yaw关系。无论结果都不自动训练、扩大或部署，也不以事后少量机制样本声称70%/10%达标。

## 运行结果（2026-10-20 09:37–09:38，Agent 代跑）

- 只读预检通过；正式运行 40.2 s，11/11 源落盘 report、trajectories 与三分支审计（含 SHA256），`status=completed`、`smoke=false`，无 `status.json`。
- 身份：11/11 起始 BEV 与训练归档一致；源前缀（如 #1298 225 步）与正常分支后缀（状态/命令误差 0.0）全部复现；4 条 `horizon_limited=true` 恰为 10 步早期组；`full_task_evaluation=false` 仅这 4 条，`for_training` 全 false。
- 全程事件组（7）：fresh 2 成功/5 碰撞/0 严格恢复；delay_once 同 2/5/0；**expert_once 1 成功/6 碰撞/1 严格恢复/丢失 2 条成功对照**。唯一严格恢复是 #1455（84 步 mv=0）；#1298/#1559/#1441/#2364 三分支全碰撞，而 #1334/#1680 两条 S4 成功对照被专家一次替换打坏（69 步 mv=6、145 步 mv=13）。
- 早期 10 步组（4）：fresh 规划安全 0/4（mv 4/2/4/3），expert_once 1/4（仅 #2330 mv=0），专家无规划失败、applied 均 ≤1。
- 结论：`decision=event_audit_only_no_training`。换一次参考不是可靠恢复手段，且会破坏既有成功对照；S8 早期越界只在个别状态被参考改善，3/4 仍越界。不训练、不扩大、不调参；明细见 `runs/diagnostics/reference-event-probe-20261019-v1/results/summary.md`。
