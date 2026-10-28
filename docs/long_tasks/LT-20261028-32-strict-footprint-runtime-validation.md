# LT32：严格车体约束完整闭环配对验证

状态：**执行完毕，独立复核完成；联合候选验收未通过**。用户完成正式评测，原计划与冻结身份保留。

## 目的与唯一主要变量

比较同一LT31 control模型在原生E5（none）与新strict_footprint运行策略下的真实闭环表现。唯一主要变量是运行策略配置；新配置包含完整参考审查、现有专家回退、共享0.2m约束的MPC及带完整角程证书的终端朝向接管，不能从整体对比单独推断每一项的收益。

这是**仿真完整几何辅助的混合闭环**验证。网络输入仍来自sensor BEV，但安全检查使用simulator_ground_truth并允许专家重规划，不是纯网络或传感器来源安全证明；production_qualified始终false。不训练、不调停止阈值、不替换默认模型或默认none模式。

## 已完成短检与正式前置条件

- 最终代码83项相关测试通过；源码快照保留source-before（171文件）与v1（140项冻结代码），旧配置/产物未改。
- 固定四例[0,163,247,292]：native 1到达/3碰撞 → strict 3到达/0碰撞；新增163/247，原生成功无丢失，292仍600步振荡。
- 固定历史成功抽查[55,110,162,212,245]：native 1到达/2碰撞/2振荡 → strict 5到达/0碰撞。此前v1的55失败已保留，新版仅在原位置容差内且完整转角满足0.2m时接管朝向。
- 合计九例strict 8到达/0碰撞/0余量侵入；2325实际控制步、237完整发出参考、全部选中H10及16个终端证书均落盘独立核验；native为2到达/5碰撞/2振荡。九例只覆盖旧205成功中的6例，不代表全体保持或泛化。
- 两个v2产物均completed且inspect通过，pilot_eligible/retention_eligible分别为true；正式prepare/precheck均0回合通过。

## 冻结输入、模型、代码与参数

- 模型：`runs/training/lt31-terminal-hold-20261027/control/last.pt`，epoch3、原生stop_threshold=0.5、image/semantic/per_view_v2，不使用best/deployment或hold臂。
- 数据：`runs/verification/surround-target-rebuild-lineage-20261027/val/val.npz`，300条、SHA `b8250edeb24f966374cb1900a101c6239723888183a37c62046214d14885cc38`；同目录manifest。已按前轮身份确认与历史同300任务/顺序/标签一致。
- 历史旧成功来源：`runs/validation/live-20261018-170806/full/E5/report.json`，完整300中205成功；不是把其指标归给LT31。
- 正式配置：`runs/prepared/strict-footprint-20261028/formal-config.json`，SHA **`7f331e90fa17e8569072e58d2e5f679f5ff84034270d759d1b5781db2ad46c70`**，冻结141代码/9输入（含两份已通过短检的report/artifacts）。所有模型/manifest/历史身份SHA在配置中逐项保存。
- 两臂完整相同0..299，先native 300再strict 300；seed0，MPC seed=dataset_index，控制dt0.1/H10，重规划10步，max_steps600，原车辆/每车位tol_pos/tol_yaw与严格小于判定。网络dt0.2，规划余量0.2m，实际实体检查0余量。
- CPU单进程、Torch/OMP/MKL线程1；禁止边运行边改代码、模型、数据或启动重复评测。

## 可复制PowerShell命令

按先后依赖依次执行。正式根目录当前尚不存在；已有目录/日志时停止反馈，不能删除后伪装首次运行。

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$strictPython = 'D:\conda\envs\endtoend-parking\python.exe'
$strictRoot = 'D:\program\endtoend Parking\runs\verification\strict-footprint-formal-20261028'
$strictConfig = 'runs/prepared/strict-footprint-20261028/formal-config.json'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

if ((Get-FileHash -LiteralPath $strictConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne '7f331e90fa17e8569072e58d2e5f679f5ff84034270d759d1b5781db2ad46c70') { throw 'LT32配置身份不符，停止' }
& $strictPython -X utf8 scripts/validate_strict_footprint.py --phase precheck --config $strictConfig
if ($LASTEXITCODE -ne 0) { throw 'LT32预检失败，停止；不要重冻SHA' }
if (Test-Path -LiteralPath $strictRoot) { throw 'LT32根目录已存在；保留现场并反馈，不覆盖或自动续跑' }
New-Item -ItemType Directory -Path $strictRoot -ErrorAction Stop | Out-Null
```

```powershell
Start-Transcript -LiteralPath (Join-Path $strictRoot 'console.log') -NoClobber -ErrorAction Stop
& $strictPython -X utf8 scripts/validate_strict_footprint.py --phase run --config $strictConfig --output (Join-Path $strictRoot 'results') 2>&1 | Tee-Object -FilePath (Join-Path $strictRoot 'python.log')
$strictRunExit = $LASTEXITCODE
[ordered]@{task='LT-20261028-32';exit_code=$strictRunExit} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $strictRoot 'exit.json') -Encoding UTF8
Write-Host "LT32 RUN EXIT $strictRunExit"
Stop-Transcript
if ($strictRunExit -ne 0) { throw 'LT32不完整；反馈日志与results/report.json，勿自动重试' }
```

```powershell
& $strictPython -X utf8 scripts/validate_strict_footprint.py --phase inspect --output (Join-Path $strictRoot 'results')
if ($LASTEXITCODE -ne 0) { throw 'LT32产物/分母/身份/裁决核验失败，保留结果并反馈' }
```

不要另行执行prepare、修改模式、阈值、样本数或重复运行。exit=0/completed表示完整运行，candidate_qualified=false也是有效负结果。

## 耗时与资源

最终固定4例双臂16.55秒、固定5例双臂20.43秒，总18回合36.98秒；按600回合线性约20.5分钟。正式含难场景、8秒上限专家重规划及参考/H10完整几何审计，估计 **10–45分钟**，CPU/磁盘差异可能更久。8小时是异常硬预算，不是预计耗时，不能自动扩大。

CPU单进程、无需GPU；预留2–4GB内存和2GB可用磁盘。计算和审计共享预算，均在用户这次正式任务中执行。正常逐例输出`[native n/300] #i ...`，随后`[strict n/300]`；每回合写JSON/NPZ。无输出时先看日志/文件增长，不另起作业。

## 输出与完整完成标志

正式目录：`runs/verification/strict-footprint-formal-20261028/`。

- 外层console.log、python.log、exit.json；结果在results/。
- results/config.json、report.json、artifacts.json；两臂native/strict各raw-report.json及300套episode-XXXX.json/npz，共600回合/1200逐例文件。
- NPZ含真实前后状态、实际动作、完整选中H10、全部发出参考及其查询状态/dt、原车位容差、终端接管时刻；JSON含真实结果、独立安全审计和终端证书。原始报告另有专家回退与控制干预统计。
- 只有status=completed、两臂0..299唯一完整、无复原失败、全部必需产物登记且SHA一致、inspect通过及run exit=0，才算执行完整。部分结果/安全停止不能算质量通过。

## Agent后续验收

先核任务/配置SHA、两段日志尾部、exit、checkpoint原生阈值、141代码/9输入和全部产物，再逐例对齐身份/分母/原始结果/审计。

本次几何辅助候选联合门槛：完整300、**至少210到达、碰撞0、实际0.2m余量侵入0、危险选中时域0、危险发出参考0**；同模型native所有成功以及历史205成功逐例保持，新增成功不得抵消丢失。原主线≥70%到达/≤10%碰撞不放宽，本候选零碰撞要求更严。

单列S4/S6/S8、原5条抽查、#163/#247恢复、#292未到达；按场景/机动/噪声报告到达和回退。核验每个终端证书满足原位置容差、完整转角和H10余量，区分网络参考接受、专家回退、选中控制改选、纯朝向接管及安全停车。独立动作/时域审计不等于重算全部CEM种群/排名。

即便candidate_qualified=true，production_qualified和all_five_modules_qualified仍false；本计划不能证明纯网络改善、L1绝对质量、遮挡/未知区域安全、真实制动动力学或泛化。seed1/2或感知来源约束若需要，结果复核后另立长任务，不自动串行启动。

## 异常、恢复、保护与反馈

预检/身份漂移、非有限、预算、状态/首动作/扫掠/终端证书审计失败、Ctrl+C或非零退出均停止并保留目录。代码finally恢复观察工厂，报告标incomplete；不自动恢复、不重复某臂、不删除/覆盖。先反馈异常及最后阶段，Agent诊断后才另建配置和新目录重跑，仍由用户执行。

禁止覆盖LT21–LT31、模型/数据/manifest、source-before/source-v1、v1负结果、两个v2短检和本次正式配置。停用新CLI/strict_footprint可回退，无旧数据迁移。

完成后回复“LT32执行完毕”或异常，并保留外层三文件及results全部内容。Agent核验后决定后续修正；本计划不部署、不训练、不回灌val、不commit/push。

## 执行结果与完成反馈后的独立复核

- 身份/完整性：正式配置SHA与交付一致；141代码/9输入/1204产物SHA、两臂各300唯一分母、退出0及inspect均通过；执行1330.842秒（22.18分钟）。
- 结果：同LT31模型原生93到达/157碰撞/50振荡；strict 287到达（95.67%）/0碰撞/12振荡+1超时，实际余量侵入0、危险H10/完整发出参考0。相对原生新增197、丢3（80/81/196），净+194；历史205保持198，丢7（22/80/81/83/85/96/196），故candidate_qualified=false，不事后放宽门槛。
- Agent独立落盘复算：600回合全部180,577实际转移/18,287参考/全部H10/724终端证书，逐例与原audit完全一致，70.561秒；无网络/CEM重跑。严格86,431步/8760参考全部安全，未重算CEM种群/排名。
- 归因：参考拒绝/回退3121/8760（35.63%）；成绩属于GT/专家混合系统，原生网络本身仅31%。剩余13条近端收敛5、远端参考替换/进度6、S8控制推进2；9条失败MPC安全改选0，不能统称安全限停。80/81/196首次分歧为200/190/0，前态相同而参考不同。
- 决定：正式任务完整完成、安全证据通过、旧能力保持门槛失败；不晋升默认入口或纯网络/五模块资格，不启动下一长任务。先针对参考替换后的连续性/进度恢复同模型三条旧成功，近端联合支撑和S8推进分开处理。
- [独立报告与证据](../../runs/verification/lt32-independent-review-20261028/review.md)。原正式目录、数据、权重、阈值、容差、步数与冻结SHA不改。
