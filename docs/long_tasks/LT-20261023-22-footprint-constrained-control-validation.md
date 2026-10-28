# LT22：完整车体约束控制的同300正式验证

状态：**已执行完成**（用户要求运行同300条条件实验；568.7秒、退出码0、inspect通过）。243成功/0碰撞，但旧成功#168丢失，`candidate_qualified=false`；余量侵入与干预统计见文末「执行结果」。固定短检、普通MPC独立重算和正式配置precheck此前已通过。对应`T-191307`、`EXEC-UPDATE-001..006`。

适用范围：**仿真完整场景几何辅助的控制执行实验**。约束判定直接使用仿真几何检查器，不是由当前传感器BEV构造的占用自由空间。LT22用于检验这个条件下的控制收益、旧能力和失效分布；即使实验指标达标，`production_qualified`仍为false，不能当作完整感知闭环/纯网络或实车已修复。对应`EXEC-UPDATE-006`。

## 目的与唯一主要变量

验证提供完整场景几何约束后，控制输出选择能否在原网络参考下减少实际碰撞，并保持LT21原205个成功。行为变量是MPC最终控制序列增加几何可行性选择：原最终均值接触安全则原样使用；危险时按原跟踪代价检查最多16个同次原CEM样本，以及原移位热启动/零动作，只输出完整预测时域物理接触安全的序列。CEM采样、均值拟合、成本、车辆/时序/停止阈值均不改，无新采样、专家回退、几何参考过滤或训练。

原网络的危险参考仍须如实统计，本轮不声称修好了网络参考生成或S6收敛。参考过滤单独/组合两臂短检没有新增成功，正式不继续它们。

## 已完成的前置证据

- 修正版验证从LT21归档精确重建76,282步真实状态/标量类型，消除4条L1初态误报；质量35危险参考/42安全当步窗口下执行碰撞/18未到达与契约分开展示，原137源/609产物不变。
- R0与v40 epoch7权重逐元素相同；训练直接读取2918个存储BEV，其中2814个target不同于几何栅格，固定普通/恢复样本5例target均精确匹配image管道。仅生成声明校正副本，target从geometry改为image，权重、停止阈值、实际输入不变，4例预测逐位相同。原模型不覆盖。
- 最终固定四案例（0/123/163/292）四臂短检40.25秒：原版1成功/3碰撞；control **4成功/0碰撞**，原成功#0保持，新增#123/#163/#292到达；reference 1成功/2碰撞/1振荡；combined 1成功/0碰撞/3振荡。短检没有S6，不能宣称S6已修复或旧205全部保住。
- 普通MPC独立重采样/拟合，未调用新候选求解：四回合1181步的候选选择、成本、窗口、真实状态精确一致；所有选中时域与实际转移接触安全。110步干预（35安全原样本、65移位热启动、10零动作）。
- **规划余量仍有问题**：control合计377个0.2m余量侵入步（#123为105、#163为2、#292为270），原四例合计157。成功/物理无碰撞不等于严格无余量侵入；这不是合格离线教师或实车安全证明。正式必须单列余量与时间/停车退化。

## 输入、代码与固定协议

- 冻结配置：`runs/verification/footprint-control-formal-20261023/config.json`，143个实际执行源与14个输入逐文件SHA、运行环境、短检与独立审计资格均已登记；正式`arms=['control']`、索引0..299。代码以SHA为准，当前工作树未提交。
- 缓存基线：`runs/verification/component-v1-formal-20261023-v2/result/`。使用已验收的同300原始结果205/77/18及归档余量统计；短检原生臂已精确复现原模型/控制结果，不再花时间重跑完整基线。
- 数据：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，SHA `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`；manifest保持原文件。
- 模型副本：`runs/verification/input-contract-corrected-20261023/deployment-image-contract.pt`，SHA `e917fed2ee965de2b497f744a8cd0f398bf85787c87176ab90f12aafc0973009`；父模型`runs/validation/arch-ab/dep-epoch0007.pt`的SHA仍`d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`。只有声明对齐image，不改神经函数。
- net-v1/hidden128/max_horizon320、网络dt=.2、原stop_threshold=.26000000000000006；MPC dt=.1/H10、population80/iterations5、最多检查16个原样本、seed=index+0、每10控制步重规划、最多600步。
- 6×3m原车辆，max_v2/max_omega1；实际物理接触完整矩形连续扫掠margin=0，规划余量=.2另列。原车位位置/环绕航向双阈值严格小于，均不放宽。
- 前置短检：`runs/verification/footprint-candidate-pilot-20261023-v2/smoke/`；独立审计：`runs/verification/footprint-candidate-audit-20261023-v2/report.json`。正式准备门禁重算配对指标、核验短检全部产物和独立审计，不只相信pilot_benefit字段。

## PowerShell命令（按顺序执行）

所有Python使用指定conda环境。前置失败即停，勿修改冻结配置绕过。

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$parkingPython = 'D:\conda\envs\endtoend-parking\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

& $parkingPython -X utf8 scripts/compare_footprint_execution.py --phase precheck --config runs/verification/footprint-control-formal-20261023/config.json
if ($LASTEXITCODE -ne 0) { throw 'LT22预检失败，停止；不要重冻SHA或启动正式运行' }
if (Test-Path -LiteralPath 'runs/verification/footprint-control-formal-20261023/results') { throw '正式results已存在，禁止覆盖或自动续跑' }
if (Test-Path -LiteralPath 'runs/verification/footprint-control-formal-20261023/console.log') { throw 'console.log已存在，请先反馈异常，不要覆盖' }
if (Test-Path -LiteralPath 'runs/verification/footprint-control-formal-20261023/python.log') { throw 'python.log已存在，请先反馈异常，不要覆盖' }
```

```powershell
Start-Transcript -LiteralPath 'D:\program\endtoend Parking\runs\verification\footprint-control-formal-20261023\console.log' -NoClobber -ErrorAction Stop
& $parkingPython -X utf8 scripts/compare_footprint_execution.py --phase run --config runs/verification/footprint-control-formal-20261023/config.json 2>&1 | Tee-Object -FilePath 'runs/verification/footprint-control-formal-20261023/python.log'
$parkingRunExit = $LASTEXITCODE
Write-Host "LT22 RUN EXIT $parkingRunExit"
Stop-Transcript
if ($parkingRunExit -ne 0) { throw 'LT22不完整，请反馈日志和results/report.json，勿自动重试' }
```

退出0只表示执行完整，质量未达标也是有效负结果。最后作只读核验：

```powershell
& $parkingPython -X utf8 scripts/inspect_footprint_execution.py --source runs/verification/footprint-control-formal-20261023/results
if ($LASTEXITCODE -ne 0) { throw 'LT22产物或联合裁决核验失败，请反馈异常' }
```

## 耗时、资源和正常进度

- 单进程，Torch固定CPU线程1，不需GPU；预留2–4GB内存与2GB磁盘。禁止同时训练、启动其他正式验证或改代码。
- 最终短检control四回合1181步约8.3秒，整个四臂短检40.25秒；全量场景/被动停车可能达到600步，多出的几何检查与磁盘开销不完全线性。估计 **10–40分钟**，慢机器可能更长；8小时为异常硬预算，不是预期耗时，不由Agent自动扩大。
- 先显示`ARM control BEGIN 300 tasks`，随后`[control n/300] #index success/失败类别`。每回合写JSON/NPZ，完整结束后才写最终raw-report/总汇总/产物登记。短暂无新行时检查进程与文件增长，勿另起重复任务。

## 输出和完整完成标志

- `results/config.json`：冻结配置副本；`results/report.json`：完整三百配对结果、旧成功丢失/新增成功、干预/零动作/余量统计、candidate_qualified。
- `results/control/episode-*.json/npz`：300套逐控制选择、成本/窗口/动作/实际状态/参考、物理接触和余量证据；`control/raw-report.json`：原生runner的候选报告。
- `results/summary.md`和`artifacts.json`：可读汇总及全部交付产物SHA；外部`console.log`和捕获原生stdout/stderr的`python.log`保留，退出码在转录结束前打印。默认控制器、旧checkpoint和LT21产物都不写入。
- 完整必须status=completed/mode=formal、300唯一任务、无复原失败、300套JSON/NPZ、当前输入/代码身份一致，最终inspect退出0。运行完整不等于模型通过。

## 验收与后续分工

Agent收到完成反馈后先核身份、日志尾部、完整300分母、全部产物SHA及逐回合账本，再按场景分析配对成功/失败、实际扫掠、余量和停车。

几何辅助系统实验的基础联合门槛：**成功≥210/300、碰撞≤30/300，原205成功不得丢失**；选中预测/实际物理接触安全还须与原矩形检查器一致。`candidate_qualified`只表示此条件实验的指标门槛，检查器明确返回geometry_source=simulator_ground_truth、production_qualified=false。规划余量侵入必须单列，不称严格0.2m安全；完整传感器闭环还需要BEV占用来源及遮挡/漏检/不确定性下的约束验证，不能直接推广。本任务无自动部署、教师银行或训练。

特别检查S4参考危险组35、S3/S8执行危险组42，以及S6原18条振荡；安全停车变超时不计收益。此次沿用已知300回归集，不作为新的泛化证据。

## 异常、恢复和禁止覆盖

- 预检失败、traceback、预算耗尽、状态/碰撞核对失败或Ctrl+C时停止，保留incomplete/running和所有部分文件；不得据部分结果晋升。
- 不支持自动恢复/续跑。先反馈异常，由Agent诊断；需要长重跑时再建新配置与新结果目录，更新计划后仍由用户运行。不要删除旧results/console.log伪装首次执行。
- 禁止覆盖LT21全部609产物/137旧源、旧R0/原数据/manifest、本轮迁移v2、声明审计/模型副本、两个固定pilot及独立审计、正式冻结config。不得手改stop阈值、车体、margin、600步预算、候选数量或采样种子。

## 完成后反馈

回复“LT22执行完毕”或异常，并保留固定目录。至少反馈`console.log`、`python.log`、`results/report.json`、`summary.md`、`artifacts.json`、`config.json`和`control/raw-report.json`；300套JSON/NPZ全部保留供独立核验。不串行启动下一阶段。

## 执行结果

- 执行：`status=completed`、`mode=formal`、`elapsed_seconds=568.68`（9.5分钟）、退出码0，`inspect_footprint_execution`返回`completed_and_verified`；300个唯一索引、300套JSON/NPZ、604产物SHA、143代码与14输入SHA全部重算一致；冻结配置与`results/config.json`逐字节相同；产物89.1 MiB。
- 配对结果（相对缓存的LT21原生300条，`rows.native`与缓存逐回合零差异）：**243成功（81.0%）、0碰撞**；旧成功丢失`[168]`，新增39（全部来自原77碰撞），净+38；失败57条=55振荡+2超时。几何门槛（≥210成功、≤30碰撞）**满足**，唯一未过项是"原205成功不得丢失"，故`candidate_qualified=false`、`formal_joint_pass=false`。
- 分组对照（按LT21冻结v2的**碰撞当步**分类，旧95失败）：原"当步参考安全而执行碰撞"42条→36成功/5振荡/1超时；原"当步消费危险参考"35条→3成功/31振荡/1超时；原"无接触未到达"18条→0成功/18振荡。旧成功#168（S4，原187步、末yaw 0.163rad）在干预下走满600步振荡，末位置0.2277m在容差内但末yaw 0.2354rad超出0.1745rad，故丢成功。
- 口径更正：此前36/41是以整段历史中是否曾有危险参考优先分组，对应33/36与6/41恢复；它不是碰撞当步的因果分组。6条曾有危险参考、但碰撞当步参考安全的回合为123/132/139/192/211/279，不能全部归入"当步参考本身危险"。本次复核采用上面的42/35与36/42、3/35，历史TRACE保留原记录，以本勘误为准。
- 分场景（原生→约束）：S1 52→55、S2 55→55、S3 29→52（+23）、S4 16→17（+2−1）、S6 21→22（+1）、S5 32→32、S8 0→10（+10）。S8与S3的"执行偏离"组几乎全部转为成功，S4的"参考危险"组基本无解。
- 干预构成：95,225控制步中79,814步（83.8%）原样使用原CEM均值；干预15,411步（16.2%），其中移位热启动14,049、原采样安全序列1,060、零动作302；`unsafe_selected_horizons`合计0，inspect核验全部归档帧的`selected_horizon_safe`与`actual_contact_safe`声明为真。完成反馈后的独立复核另用原车辆动力学与矩形检查器推进全部95,225个归档实际动作，真实状态逐位一致、实际接触0、余量侵入17,337步；没有独立重跑全300的CEM选择或10步候选时域，不能把声明核验等同完整求解复算。[独立复核](../../runs/verification/lt22-independent-review-20261023/report.json)。
- 余量单列：规划余量侵入**17,337步/104条**（原生同300为3,164步/115条，最大#295为536步），S4 8,628、S8 3,928、S3 3,484为主；参考本身接触不安全的窗口13,905步（81,320步安全），网络危险参考未修。物理无碰撞不等于0.2m余量安全，也不等于合格教师。
- 判读：该几何辅助条件实验把碰撞从77降到0、成功率从205提到243（净+38），代价是16.2%的控制步被干预、旧成功丢1条、余量侵入约为原生5.5倍，且对"参考本身危险"与"振荡未到达"两组基本无效。指标门槛虽满足成功/碰撞两条，但按既定联合条件（旧205不得丢失）不通过，`production_qualified`固定false；不使用GT几何以外的信息，不替换默认控制器，不部署、不训练，主联合验收68.3%/25.7%不变。
