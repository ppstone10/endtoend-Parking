# LT25：全部控制输出分支统一规划余量同300验证

状态：**已执行完成**（用户要求执行；681.46秒、退出码0、inspect返回`completed_and_verified`）。268成功/0碰撞、规划余量侵入降为0，但丢失LT24的6条旧成功，`candidate_qualified=false`；明细见文末「执行结果」。代码、冻结正式配置及短检独立复核此前已完成。任务T-144155，对应MARGIN-SEARCH-005..007。

## 目的与唯一主要变量

相对LT24，唯一主要行为变量是MPC最终选择的可行集：所有控制输出分支统一要求0.2m完整矩形扫掠安全。均值、原前16样本、移位热启动、零动作、终端完整转向与原零解之后补查同次400样本均使用同一检查器。参考仍来自正常BEV/网络，每10步更新；LT24原到达证书保持、原成本/采样/RNG/车位容差/600步预算不变。

该变量包括“约束一致化及受该约束的原样本覆盖”，不将其拆成没有运行过的独立因果消融；不能宣称单独扩大样本即有效。仿真引擎仍独立按实体0余量检查碰撞，0.2m规划侵入单列。所有300归档起点与名义目标均满足0.2m；任意外部已经侵入的状态中断，本轮不实现退出恢复。

仍为GT完整几何辅助、静态确定性差分驱动且可立即停车的条件实验，production_qualified固定false。BEV占用约束来源、动态障碍、实车惯性未验证；不修改默认入口、不部署、不训练、不回灌val。

## 前置证据及局限

- 只在新补查动作上加0.2m的首阶段八例4→5成功，但余量607→1082。独立审计确认新增改选合规，后续普通均值/热启动重新侵入；首阶段不进入正式。
- 统一全部分支后的固定0/167/168/172/183/267/268/292八例，39.240秒：**4→5成功、碰撞0、四个旧成功无丢失、余量607→0**。#172由600步失败转396步到达；#292保持成功且420→327步、270侵入→0。#167/#183/#268仍失败，不能认为整个未到达瓶颈已解决。
- 60项相关聚合unittest和10个新增Python文件编译通过。独立审计22.290秒核对全部3314实际动作/状态/实体及余量、所有完整H10、原始/使用NN参考与5份证书；888个原零解上下文重建原400样本、165次改选成本/排名/序列精确。另7个终端旋转完整角程余量独立复核通过。
- 短检：runs/verification/planning-margin-pilot-20261026/smoke/；审计：runs/verification/planning-margin-audit-20261026/report.json；分点证据：runs/verification/planning-margin-review-20261026/{report.json,review.md}。
- 小样本旧成功保持不证明原263全部保持；统一严格余量可能减少旧有可达性、引起更多安全停车，本正式实验正是检查该风险。原LT24余量14944并非严格余量安全，不可把“保留成功”和“全面安全”混为一谈。

## 输入、代码身份与固定参数

- 冻结配置：runs/verification/planning-margin-formal-20261026/config.json，**165执行源/32输入**逐文件SHA和运行环境固定。当前工作树未提交，以SHA为实际版本；不得更新旧SHA绕过预检。32输入包括LT24谱系和新固定短检/独立审计证据。
- 数据：data/task_dataset/tracked_pivot_v8_3000/val.npz，SHA ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963，原manifest，索引0..299唯一300。
- 模型：runs/verification/input-contract-corrected-20261023/deployment-image-contract.pt，SHA e917fed2ee965de2b497f744a8cd0f398bf85787c87176ab90f12aafc0973009；权重/实际image及semantic输入不变，net-v1、hidden128/max_horizon320、NN dt=.2、stop_threshold=.26000000000000006。
- 6×3m居中矩形，max_v2/max_omega1，MPC dt=.1/H10，population80/iterations5，seed=index，replan_every10/max_steps600；车位原tol_pos/tol_yaw逐任务复原，不统一替换为默认值。
- 规划碰撞检查器以原外廓增加0.2m，原几何分辨率不变；实体0余量另列。普通原均值若满足规划约束直接保留；危险时使用原候选/热启动/零动作；原全零介入时稳定检查原16之外的同次样本，仅选严格低于零解成本且规划安全的首个样本。不增加随机采样或改成本。
- LT24证书近端3×原tol_pos、最多80步且受剩余原预算限制，感知和NN推理继续；原参考对象/进度/dtype保留，实际动作须精确匹配证书。
- 唯一正式臂为本候选；缓存LT24同300263成功/0碰撞/14944余量只读复用，不重新跑基线或另启训练。

## PowerShell命令（严格依次执行）

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$parkingPython = 'D:\conda\envs\endtoend-parking\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

& $parkingPython -X utf8 scripts/compare_planning_margin.py --phase precheck --config runs/verification/planning-margin-formal-20261026/config.json
if ($LASTEXITCODE -ne 0) { throw 'LT25预检失败，停止；不得重冻SHA绕过' }
if (Test-Path -LiteralPath 'runs/verification/planning-margin-formal-20261026/results') { throw 'results已存在，禁止覆盖或自动续跑' }
if (Test-Path -LiteralPath 'runs/verification/planning-margin-formal-20261026/console.log') { throw 'console.log已存在，请先反馈异常' }
if (Test-Path -LiteralPath 'runs/verification/planning-margin-formal-20261026/python.log') { throw 'python.log已存在，请先反馈异常' }
```

```powershell
Start-Transcript -LiteralPath 'D:\program\endtoend Parking\runs\verification\planning-margin-formal-20261026\console.log' -NoClobber -ErrorAction Stop
& $parkingPython -X utf8 scripts/compare_planning_margin.py --phase run --config runs/verification/planning-margin-formal-20261026/config.json 2>&1 | Tee-Object -FilePath 'runs/verification/planning-margin-formal-20261026/python.log'
$parkingRunExit = $LASTEXITCODE
Write-Host "LT25 RUN EXIT $parkingRunExit"
Stop-Transcript
if ($parkingRunExit -ne 0) { throw 'LT25不完整；反馈日志与部分产物，禁止自动重试' }
```

退出0仅表示完整执行，负结果也保留。最后只读核验：

```powershell
& $parkingPython -X utf8 scripts/compare_planning_margin.py --phase inspect --source runs/verification/planning-margin-formal-20261026/results
if ($LASTEXITCODE -ne 0) { throw 'LT25完整性或裁决核验失败，请反馈异常' }
```

audit入口只接受固定短检，正式完成后由Agent按预算复核实际控制及必要证据，不自动调用短检audit来扩大运行范围。通用comparison.pilot_benefit仅判断碰撞严格下降（0→0为false），不参与本协议；本策略短检/正式使用planning_margin_benefit及candidate_qualified。

## 耗时、资源与正常进度

单进程CPU，Torch线程1，无需GPU，预留2–6GB内存、2GB磁盘。LT24同300约8.3分钟；本八例3314步39.24秒，按LT24的86334步线性约17分钟，但严格余量可产生更多600步失败和大量样本扫掠，估计**15–60分钟**，CPU/磁盘及安全干预差异可能更慢；8小时仅为异常硬预算。上述估计不是保证。

正常逐回合输出[planning-margin n/300] #index success/oscillation/...，并保存episode/retention文件；回合内会做私有证书或样本完整扫掠，可能稍久无进度。不要重复启动、同时修改代码/配置/训练或删除产物。

## 输出、成功标志与Agent后续验收

输出runs/verification/planning-margin-formal-20261026/results/：config.json/report.json/summary.md/artifacts.json/control/raw-report.json，300套episode-XXXX.json/npz和300套retention-XXXX.json/npz。通常1204登记产物，artifacts.json自身及外部日志不计入；控制记录包含所有真实状态/命令/完整序列/消费窗口/余量与补查上下文，保持记录包含所有原始网络参考、使用SHA、证书/RNG与进度。日志为同级console.log/python.log。

完整标志：退出0、completed/formal、300唯一索引、无复原失败、两种300套账本完整、代码/输入/配置/产物SHA一致、inspect返回completed_and_verified。Agent收到反馈后先验任务身份/日志尾/完成/分母/必要文件，再作指标分析，不凭单个汇总晋升。

严格条件实验联合门槛：**LT24全部263成功保持、至少新增1成功、碰撞0、规划余量侵入0**（有效成功至少264/300）。不以新增抵消旧成功损失；若未通过保留负结果，不晋升。即使通过candidate_qualified，production_qualified仍false。逐场景/新增/丢失、碰撞与余量步数及比例、真实零动作、终端模式、样本检查/改选和证书/推理覆盖分别报告；独立核验实际转移、完整H10、必要终端旋转与补查上下文。

本候选可以消除安全侵入而仍不能到达，必须保留这种失败；不宣称37失败全修复，也不把安全停车当成功。

## 异常停止、恢复、重跑与保护

预检失败、traceback、当前规划余量已违反、证书/动作漂移、预算耗尽或Ctrl+C时停止，保留running/incomplete报告和全部部分产物，均不算完整。当前无自动恢复或断点续跑；反馈后先诊断，如需长重跑，另建新配置/新目录及恢复计划，仍由用户执行。不要删除旧目录或修改容差/.2/80步/600步/模型/样本数或SHA来继续。

保护LT21–LT24全部源码/配置/结果，原模型/数据/manifest、声明修正版、前轮物理补查负结果、本轮首阶段负结果、统一分支短检/审计及正式config。停用本新CLI即可回退默认行为，无数据迁移。未执行git commit或push。

完成后反馈“LT25执行完毕”或异常，保留console.log/python.log及完整results（包含所有episode/retention JSON/NPZ）。Agent复核后决定下一步，不自动启动下一阶段长任务。

## 执行结果

- 执行：`status=completed`、`mode=formal`、`elapsed_seconds=681.46`（11.4分钟）、退出码0，`inspect`返回`completed_and_verified`；300唯一索引、300套episode+300套retention、**1204**产物SHA、165代码与32输入SHA全部重算一致，`results/config.json`与冻结配置逐字节相同，产物122.3 MiB。
- 结果：**268成功（89.33%）/0碰撞/规划余量侵入0步**（83,710个实际步逐帧`planning_margin_safe`全为真）。失败32条=31振荡+1超时。
- 联合门槛：LT24的263成功**未全部保持**——丢失#131/#132/#139/#161（S3）、#279（S6）、#296（S8）；新增11条 #172/#173/#181/#185/#188/#202/#205/#207/#247/#257/#280（S4×8、S6×3），净+5。故"263全保持+新增≥1+碰撞0+余量0"中三项满足、旧成功保持一条不满足 → `planning_margin_benefit=false`、`candidate_qualified=false`。
- 分场景（LT21/LT22/LT23/LT24/LT25）：S1 52/55/55/55/55、S2 55/55/55/55/55、S3 29/52/52/52/**48**、S4 16/17/25/25/**33**、S6 21/22/25/33/**35**、S5 32/32/33/33/33、S8 0/10/10/10/**9**。
- 丢失去向：6条全部为安全停车/振荡而非碰撞——#131/#132/#139/#161走满600步（末位置误差6.55–7.89 m）、#279走满600步（1.45 m）、#296超时（8.25 m）；余量侵入均为0。#131原本也是LT21原生205成功之一，说明统一严格余量确实以可达性为代价。
- 补查与分支：零解补查调用9,647次，其中1,922次在原16之外同次样本中改选（共检查2,606,450个样本，选中1基排名最大393），其余保留零解；zero_hold选择标签429步（真实全零命令另计7,725步）、终端完整转向135次；参考保持290次候选→250证书→225次实际保持。inspect核对身份与账本声明；全部实际转移与完整H10另经完成后的独立几何重算通过。
- 判读：把0.2m余量施加到**全部**输出分支后，余量侵入从LT24的14,944步降到**0**、碰撞仍为0，并把S4的8条振荡/超时转成到达（216–446步）；代价是S3四条、S6一条、S8一条原本成功的回合同样被安全约束拦停（其中#131是原生成功）。因此当前候选未能同时满足严格余量与全部旧成功保持，按预声明门槛为负结果；这不证明安全路径不存在或安全到达不可兼得，不晋升、不替换默认控制器；仍为GT几何辅助、静态确定性的条件实验，`production_qualified=false`，主联合68.3%/25.7%不变。

## 完成反馈后的独立复核

165源/32输入/1204产物SHA、配置字节、完整300进度及退出0重新核验；python.log按实际UTF-16编码读取。独立重算27.003秒，83,710实际动作/状态及完整H10全部满足实体0余量和规划0.2m，原车位首次联合到达及误差精确；8495原始/使用参考、250证书3559实际步、135完整终端角程核验通过。没有重采样全部CEM或重推网络/感知，不把inspect的声明核对当作独立几何求解。

六条丢失的LT24原路径各有58–412侵入步、共897步。首次动作分歧前态完全相同，六条原完整H10及当时发出的完整NN参考均不满足余量；新H10全部满足。五条旧首动作尚安全却后续实际侵入，说明只验首步不足；两条消费近端窗口安全却执行H10不安全，不能把失败全归给网络或全归给MPC。当前证据支持继续验证安全参考可执行性与受约束推进能力，不支持宣称安全与到达不可能兼得。

剩余32失败全部末位置未达原容差，14条末100步真零动作，28条末已裁剪参考无联合点；真零命令7725步，zero_hold标签429仅是选择标签。详见[正式独立复核](../../runs/verification/lt25-independent-review-20261026/review.md)。本轮未改冻结算法/模型/数据、未替换默认入口或启动下一长任务。
