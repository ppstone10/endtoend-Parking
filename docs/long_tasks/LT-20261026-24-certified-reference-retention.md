# LT24：带到达证书的网络参考保持同300验证

状态：**已执行完成**（用户要求运行正式同300；497.95秒、退出码0、inspect返回`completed_and_verified`）。263成功/0碰撞，LT23的255成功零丢失、新增8（全在S6）、余量14,944不增，`candidate_qualified=true`、`production_qualified=false`；明细见文末「执行结果」。153执行源/26输入冻结，固定六例短检、干净控制器独立证书重建与正式配置precheck此前已通过。任务T-093105，对应REF-KEEP-001..005。

## 目的与唯一主要变量

相对LT23仅增加带到达证书的网络参考保持：在原重规划点，当前距目标≤3×本车位原tol_pos、当前满足0.2m余量、原始网络参考含原联合合格点时，用完整复制的LT23控制上下文/RNG最多推演80步（并受600-已执行步限制）。只有该原参考在私有控制执行中无实体接触、无0.2m余量侵入并严格到达，才保持同一参考对象到证书结束。每个实际前态/动作/后态与证书精确核对，漂移就中断实验；没有证书仍执行原LT23。

感知与网络每10步仍正常执行，原始新参考与实际使用参考分别保存。不会生成新路径，不增加专家、平移/目标偏置、训练或改变NN停止阈值。LT23终端朝向模式继续保留，原车位容差不变。

这仍是**仿真完整几何辅助条件实验**，不是BEV占用约束接入或纯网络收益。静态场景、精确确定性模型是证书条件，不作动态障碍/实车保证；production_qualified固定false，不自动部署、训练或回灌val。

## 固定前置证据

- 原上下文精确重放后，从step500只保持原参考：#267在step528到达，#167/#247未恢复，故未采用无条件保持。
- 新候选固定0/167/168/247/267/292六例，13.326秒：LT23 3成功→4成功、零碰撞、旧成功无丢失；#267用step400的20步证书在420步到达，#0用step170的18步证书在188步到达（原285）。#167/#247仍600步失败。
- 7次私有尝试取得3个证书（共41步），实际只保持2次原重规划更新；#168的3步证书不跨下次重规划。3份证书由干净LT23控制器独立重建热启动/RNG/CEM/动作/状态/原车位标准与0.2m余量，全部精确一致；全部2411实际步/原始与使用参考核验一致。未保持的167/168/247/292全部NPZ数组与LT23逐位一致。
- 本组余量侵入607→607，来自旧失败/旧成功路径，不能称已解决规划余量。按步归一比例可能因无侵入回合提前到达而变化，正式必须另列。
- 87项聚合测试、6文件编译通过。短检`runs/verification/reference-retention-pilot-20261026/smoke/`；独立审计`runs/verification/reference-retention-audit-20261026/report.json`。

## 输入、代码和固定参数

- 冻结配置：`runs/verification/reference-retention-formal-20261026/config.json`。153个源/26输入均逐文件SHA，运行环境固定；当前工作树未提交，以SHA身份为准。输入包含LT21/LT22/LT23原证据、模型/数据谱系及新短检/审计，旧配置/代码不更新。
- 数据：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，SHA `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`，原manifest不变，固定索引0..299。
- 模型：`runs/verification/input-contract-corrected-20261023/deployment-image-contract.pt`，SHA `e917fed2ee965de2b497f744a8cd0f398bf85787c87176ab90f12aafc0973009`；原权重/实际image与semantic输入不变，net-v1/hidden128/max_horizon320，NN dt=.2，stop_threshold=.26000000000000006。
- 原6×3m车辆、max_v2/max_omega1、控制dt=.1/H10、CEM population80/iterations5、原约束最多16候选、seed=index、replan_every10/max_steps600，原tol_pos/tol_yaw取自车位（不是统一0.3m/10°）。实体0余量与规划0.2m分别报告。
- 唯一新增策略固定近端半径3×tol_pos、私有前瞻≤80步且≤剩余原步数；没有参数搜索。保持轨迹点/dt/dtype/对象进度；新原始NN参考照常计算，控制RNG不被私有尝试消费。
- 缓存基线LT23同300为255成功/0碰撞、14,944余量侵入；其255成功已覆盖原生205与LT22的243成功，不重复跑整套基线。

## PowerShell命令（严格按顺序）

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$parkingPython = 'D:\conda\envs\endtoend-parking\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

& $parkingPython -X utf8 scripts/compare_reference_retention.py --phase precheck --config runs/verification/reference-retention-formal-20261026/config.json
if ($LASTEXITCODE -ne 0) { throw 'LT24预检失败，停止，不得重冻SHA绕过' }
if (Test-Path -LiteralPath 'runs/verification/reference-retention-formal-20261026/results') { throw 'results已存在，禁止覆盖或自动续跑' }
if (Test-Path -LiteralPath 'runs/verification/reference-retention-formal-20261026/console.log') { throw 'console.log已存在，请先反馈异常' }
if (Test-Path -LiteralPath 'runs/verification/reference-retention-formal-20261026/python.log') { throw 'python.log已存在，请先反馈异常' }
```

```powershell
Start-Transcript -LiteralPath 'D:\program\endtoend Parking\runs\verification\reference-retention-formal-20261026\console.log' -NoClobber -ErrorAction Stop
& $parkingPython -X utf8 scripts/compare_reference_retention.py --phase run --config runs/verification/reference-retention-formal-20261026/config.json 2>&1 | Tee-Object -FilePath 'runs/verification/reference-retention-formal-20261026/python.log'
$parkingRunExit = $LASTEXITCODE
Write-Host "LT24 RUN EXIT $parkingRunExit"
Stop-Transcript
if ($parkingRunExit -ne 0) { throw 'LT24不完整，请反馈日志和部分产物，禁止自动重试' }
```

退出0只表示完整执行，负结果也应保留。最后只读核验：

```powershell
& $parkingPython -X utf8 scripts/compare_reference_retention.py --phase inspect --source runs/verification/reference-retention-formal-20261026/results
if ($LASTEXITCODE -ne 0) { throw 'LT24完整性或配对裁决核验失败，请反馈异常' }
```

正式不调用只接受固定短检的`--phase audit`；Agent收到完成反馈后按预算另做独立实际动作及必要证书审计。不自动串行运行其他任务。

## 耗时、资源与进度

单进程CPU，Torch线程1，无需GPU，预留2–6GB内存、2GB磁盘。LT23同3008.1分钟；本轮六例2411实际步加7次私有前瞻共13.3秒。全量近端候选数量不确定，最坏每次增加80控制步私有求解，估计**10–60分钟**，CPU/磁盘差异可能更慢；8小时为异常硬预算，不是预期时长。

正常进度为`[retention n/300] #index success/oscillation/...`；每回合写episode与retention文件。某回合可因私有前瞻稍久不输出，不另启重复作业，不同时训练、修改代码/参数或删除目录。

## 输出与验收

输出`runs/verification/reference-retention-formal-20261026/results/`：config.json/report.json/summary.md/artifacts.json/control/raw-report.json；300套episode-XXXX.json/npz（真实状态/动作/参考/车体安全/余量），另300套retention-XXXX.json/npz（全部原始NN参考/查询SHA/实际使用SHA/副本热启动与RNG/证书/保持统计）。通常1204登记产物，artifacts.json自身及外部日志不计入；全部文件保留。同级console.log/python.log记录实际退出码。

完整条件：completed/formal、300唯一索引、无复原失败、两种300套账本齐全、代码/输入/产物SHA不变、退出码0、inspect返回completed_and_verified。

条件实验联合门槛：**LT23的255成功全部保持、相对LT23至少新增1成功、实际碰撞0、规划余量侵入≤14,944步**，并满足基础成功≥210/300。不得以新增抵消旧成功回退。报告逐场景S4/S6/S8、每条新增/丢失、余量步数及归一比例、证书尝试/成功/保持次数、正常感知推理覆盖、零动作及终端模式。所有实际已保持转移须精确遵循证书，证书原预算/实体/余量/原车位到达须可独立核验。

即使candidate_qualified=true，仍production_qualified=false；原剩余45条不保证全部恢复，S4危险参考与安全保持、BEV来源约束、原有规划余量和独立泛化均未被本候选完整解决。不部署、不训练、不制作教师银行。

## 停止、恢复、重跑与保护

预检失败、traceback、预算耗尽、证书前态/动作/后态不精确、Ctrl+C时停止，保留incomplete和全部部分文件。当前无自动恢复/续跑；反馈后由Agent先诊断，需要长重跑时另建新配置/新目录并交付计划，仍由用户执行。不要手改容差/80步/600步/半径/模型或SHA来继续，不删除既有结果伪装首次运行。

保护所有LT21/LT22/LT23源码/配置/报告/产物、原模型/数据/manifest、声明校正副本、本轮短检/审计及正式config。停用新CLI即回退，不需要旧数据迁移。

完成后回复“LT24执行完毕”或异常，至少保留console.log、python.log和完整results（含全部episode/retention两套JSON/NPZ）。Agent先验身份/日志/完成/分母/产物，再分析和决定后续；不自动启动下一阶段。

## 执行结果

- 执行：`status=completed`、`mode=formal`、`elapsed_seconds=497.95`（8.3分钟）、退出码0，`inspect`返回`completed_and_verified`；300唯一索引、300套episode+300套retention JSON/NPZ、1204产物SHA、153代码与26输入SHA全部重算一致，`results/config.json`与冻结配置逐字节相同，产物95.7 MiB。
- 结果：**263成功（87.67%）/0碰撞**；相对LT23的255成功**零丢失**、**新增8**（#250/#260/#263/#267/#270/#274/#279/#284，全部S6_loading_face）；相对LT21原生205与LT22的243同样零丢失。失败37条=35振荡+2超时，分布S4 25（23+2）、S6 11、S8 1。
- 分场景（LT21/LT22/LT23/LT24）：S1 52/55/55/55、S2 55/55/55/55、S3 29/52/52/52、S4 16/17/25/25、**S6 21/22/25/33**、S5 32/32/33/33、S8 0/10/10/10。
- 参考保持：291次候选尝试→192份证书→158次实际保持，分布在104个回合；证书长度1–37步（上限80、且≤剩余预算），全部证书内部`contact_safe`与`planning_margin_safe`为真。8个新增成功全部依赖保持（每个1–3次），失败回合中没有任何一次保持（holds in failing episodes = 0）。
- 证书可核验性（Agent独立重算，只读）：192/192份证书的`states_before`/`states_after`/`controls`与归档episode NPZ对应`start_step`切片**逐元素完全相等**；8,758次重规划查询中158次`holding_previous=true`且**全部**有证书、且`used_sha256≠raw_sha256`（确实延用上一参考），其余查询100%使用当次原始NN参考，无静默替换。
- 干预与余量：控制步86,334（LT23为91,103，−4,769，来自提前到达），干预12,639步、零动作285、`unsafe_selected_horizons=0`、参考接触不安全窗口11,035步。余量侵入14,944步与LT23完全相同（门禁≤14,944通过），但**按步归一比例由0.1640升至0.1731**——分母因提前到达变短，须单列，不能作为余量改善。
- 联合条件逐项：LT23的255成功零丢失 ✅、相对LT23新增8≥1 ✅、实际碰撞0 ✅、余量14,944≤14,944 ✅、成功263≥210 ✅ → `candidate_qualified=true`；`production_qualified`保持false。
- 判读：带到达证书的参考保持只影响104个回合、158次重规划点，就把S6从25成功提到33、并让#267从600步振荡变为420步到达，且不丢任何既有成功、不增碰撞、不增余量总量；说明"S6远端反复重规划打断可执行参考"确实是一个可独立修复的缺口。但S4仍有25条失败（危险参考与远端振荡）、S6仍有11条、S8 1条，网络危险参考11,035窗口未修，余量比例反而略升；仍是仿真完整几何辅助的条件实验，不解决BEV来源、动态障碍与实车动力学，主联合68.3%/25.7%与所有生产准入不变。

## 完成反馈后的独立复核

153源/26输入/1204产物与完整300裁决重新核验通过。另独立推进全部86,334实际动作，并恢复干净LT23控制器的热启动/RNG/进度，重新求解全部192份成功证书的2486个控制步；动作/状态/原车位首次联合到达/实体与0.2m余量均精确一致，约12.84秒。8758个原查询、原始与使用参考账本完整；未跨更新保持的196回合全部episode NPZ数组逐位等于LT23，含全部37个剩余失败。

剩余37条末态都未进入原位置容差，其中31条最后参考无联合合格点、6条有合格点仍未到达，28条末100步真实控制全零。按原生碰撞当步分组，42条安全参考执行碰撞恢复37，35条消费危险参考恢复11，18条无接触未到达恢复10。完整证据见[独立复核](../../runs/verification/lt24-independent-review-20261026/review.md)。非证书CEM/未成功私有尝试/网络与传感器未独立重跑，不作BEV、泛化或生产准入证明。
