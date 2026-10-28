# LT26：合法参考窗口内的控制候选分阶段验证

状态：**已执行完成**（用户要求执行；六教师→14NN→审计→inspect 全部退出0）。教师6/6合格，但真实NN仍5成功、无新增且未恢复LT24子集6条，`candidate_qualified=false`；明细见文末「执行结果」。v2代码、单元测试、两个冻结状态预检、配置及入口检查此前已完成。任务T-171525，契约REF-CONTROL-001..008。

## 目的与唯一主要变量

相对LT25，仅在原安全处理最终选全零、当前消费窗口连接满足原0.2m完整车体扫掠时，补充一条从该窗口和真实状态按原差分模型推导的控制序列。完整H10安全且原成本严格低于停车才接受。原采样、成本、随机流、车位容差、更新节奏、证书保持和步数预算不变。参考窗口不合法则精确保留LT25，不推导动作。

这是完整闭环MPC执行策略的实验修改。六教师控制单元用于隔离控制缺口；14例从任务起点运行正常传感器→BEV→NN→MPC→物理推进，用来判定真实闭环收益。教师不进入NN回合，不训练、不回灌val、不替换默认控制器。

## 已有证据与边界

- LT25同300：268成功/0碰撞/0余量侵入，但丢失LT24六条旧成功，未晋升。
- 无合法窗口前置条件的v1：六安全教师6/6，#296由LT25安全教师下失败变230步到达；真实14例却从5成功降到4，丢#292、没有新增，碰撞/余量仍0。独立审计79.945秒通过，证明记录与选择一致，不证明任务效果合格。v1已拒绝正式。
- v2定点预检：#292原step14消费窗口不安全，精确保留LT25零解/成本6.850806831953879；#296教师初态窗口安全，仍接受安全候选，原成本2.7824856574422387降至0.034779777488577204。两点的原随机流一致，旧400样本与新增选择独立重建通过。**不据此宣称#292整回合保持或v2六教师通过。**
- 静态仿真GT几何、确定性差分模型且可立即停车。BEV占用约束、动态障碍、实车惯性未验证，production_qualified固定false。当前已侵入规划余量状态中断，未实现退出恢复。

## 版本、输入与固定参数

- 冻结配置：`runs/verification/feasible-reference-pilot-20261026/config.json`，**174执行源、38输入**及运行环境逐文件固定；未提交工作树以SHA为准。配置SHA：`5fffc562f43d85960ac29ba3bbb96d65352bd8163a1acf186b3dca7bfa542648`。
- 输入包含LT25及其冻结谱系、v1负结果/manifest/独立审计；旧165源及v1新增5源均未改变。禁止更新SHA绕过预检。
- 数据：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，SHA `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`。教师只读该数据已有轨迹，不能算新训练数据。
- 模型：`runs/verification/input-contract-corrected-20261023/deployment-image-contract.pt`，SHA `e917fed2ee965de2b497f744a8cd0f398bf85787c87176ab90f12aafc0973009`。原net-v1、hidden128/max_horizon320、NN dt=.2、停止阈值.26000000000000006；权重/实际输入不变。
- 车辆6×3m、max_v2/max_omega1；MPC dt=.1/H10、population80/iterations5、seed=index、replan_every10/max_steps600。逐车位复原原tol_pos/tol_yaw；实体0余量与规划0.2m分别核验。到达证书仍为最多80步且受原剩余预算限制。
- 教师固定六例：131/132/139/161/279/296；NN固定14例：0/131/132/139/161/167/168/172/183/267/268/279/292/296。不得替换案例、扫参数或重跑原基线。

## 可复制PowerShell命令（依次执行）

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$parkingPython = 'D:\conda\envs\endtoend-parking\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$parkingPilot = 'runs/verification/feasible-reference-pilot-20261026'
$parkingAudit = 'runs/verification/feasible-reference-audit-20261026'
$parkingEntry = 'scripts/compare_feasible_reference_candidate.py'

function Invoke-LT26Python {
    param([string[]]$ParkingArgs)
    & $parkingPython -X utf8 @ParkingArgs
    if ($LASTEXITCODE -ne 0) { throw "LT26阶段异常退出 $LASTEXITCODE；停止并反馈，不自动重试" }
}

Invoke-LT26Python -ParkingArgs @($parkingEntry, '--phase', 'precheck', '--config', "$parkingPilot/config.json")
foreach ($parkingTarget in @("$parkingPilot/teacher", "$parkingPilot/smoke", "$parkingPilot/console.log", $parkingAudit)) {
    if (Test-Path -LiteralPath $parkingTarget) { throw "已有产物 $parkingTarget，禁止覆盖或自动续跑，请先反馈" }
}
```

以下为一个连续阶段块。六教师不全合格时立即停止，不能绕过而启动NN。

```powershell
Start-Transcript -LiteralPath 'D:\program\endtoend Parking\runs\verification\feasible-reference-pilot-20261026\console.log' -NoClobber -ErrorAction Stop
try {
    Invoke-LT26Python -ParkingArgs @($parkingEntry, '--phase', 'teacher', '--config', "$parkingPilot/config.json")
    $parkingTeacherReport = Get-Content -LiteralPath "$parkingPilot/teacher/report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($parkingTeacherReport.status -ne 'completed' -or $parkingTeacherReport.qualified -ne $true -or $parkingTeacherReport.rows.Count -ne 6) {
        throw '六教师未全部合格；保留负结果并反馈，停止后续NN'
    }
    Invoke-LT26Python -ParkingArgs @($parkingEntry, '--phase', 'smoke', '--config', "$parkingPilot/config.json")
    Invoke-LT26Python -ParkingArgs @($parkingEntry, '--phase', 'audit', '--source', "$parkingPilot/smoke", '--output', $parkingAudit)
    Invoke-LT26Python -ParkingArgs @($parkingEntry, '--phase', 'inspect', '--source', "$parkingPilot/smoke")
    Write-Host 'LT26全部阶段执行完成；等待Agent按联合门槛复核，不自动开始全300'
}
finally {
    Stop-Transcript
}
```

若NN指标为负，仍运行本计划中的审计和inspect以区分实现正确性与效果失败；它们正常退出不等于效果合格。六教师不合格则直接反馈六教师产物，不运行后面三阶段。配置用途为smoke，不使用`--phase run`，本轮不准备正式300配置。

## 预计耗时、资源与正常进度

估计总计**5–15分钟**，单进程CPU、NN线程1、无需GPU；预留2–6GB内存、2GB磁盘。依据：v1六教师3.74秒、14NN103.57秒、独立审计79.95秒，另有输入/代码/大JSON核验。v2可能改变轨迹、600步失败数和样本检查密度，耗时不能可靠按v1保证，因此按长任务交用户执行。教师硬预算90秒、NN180秒、审计120秒均只约束各自计算阶段，超时应反馈而非擅自放宽。

正常依次输出`[reference-teacher] #index ...`、`[reference-candidate n/14] #index ...`，随后审计completed和inspect的completed_and_verified。回合间会保存episode/retention账本，独立审计中可能约1–2分钟没有进度文本。运行期间不并发修改冻结文件、不重复启动。

## 输出与完成反馈

- `runs/verification/feasible-reference-pilot-20261026/console.log`。
- 同目录`teacher/`：config.json/report.json/artifacts.json及六套episode-XXXX.json/npz，共14登记文件，manifest自身不计入。
- 同目录`smoke/`：config.json/report.json/summary.md/artifacts.json、control/raw-report.json，14套episode与14套retention的JSON/NPZ，通常60登记文件，manifest自身不计入。
- `runs/verification/feasible-reference-audit-20261026/report.json`。

完成后反馈“LT26执行完毕”；异常则反馈阶段、错误和已有产物目录，全部保留。Agent先核任务身份/配置/日志尾/退出及完成状态/六与14唯一分母/文件完整性及SHA，再分析按场景新增与回退。

## Agent后续验收

教师必须6/6原车位严格到达、碰撞0/余量0；NN必须保持LT25此子集全部5条旧成功、至少新增1，同时恢复LT24此子集全部旧成功，碰撞0/余量0；审计必须覆盖所有真实转移/H10/终端、原始使用参考/证书、原400样本及接受/拒绝/非法窗口跳过。原车位容差/600步/正常NN更新不变。

只有上述全部满足才另行准备同300计划；本计划完成不会自动触发它。正式还需保持LT25全部268与LT24全部263旧成功、新增≥1、安全指标0。安全教师收益不等于真实NN闭环收益，少数点一致不等于全程保持；即便正式通过也仍非生产准入。

## 异常停止、恢复与保护

身份漂移、traceback、超预算、六教师不全合格、分母/证书不符、Ctrl+C均停止，保留running/incomplete/负结果及日志。当前无断点恢复、自动重试或下一阶段续跑。反馈后Agent诊断，必要重跑使用新配置/新目录及新计划，不能删除旧目录或修改SHA/.2/容差/预算/模型绕过。

保护LT21–LT25所有源码/输入/配置/结果、v1负结果/审计、原模型/数据/manifest和v2冻结配置；不覆盖定点预检证据。停用新CLI可回退原行为，无数据迁移；未执行git commit/push。

## 执行结果

- 执行：预检`passed`（purpose=smoke，episodes_run=0）→ teacher（3.97秒）→ smoke（107.69秒）→ audit（80.34秒）→ inspect；全部退出0，`inspect`返回`completed_and_verified`。冻结配置SHA `5fffc562f43d85960ac29ba3bbb96d65352bd8163a1acf186b3dca7bfa542648`与计划一致；174代码/38输入SHA重算零漂移；teacher 14个登记产物、smoke 60个登记产物SHA全部一致，teacher 6套、smoke 14套episode+14套retention齐全。
- 六教师：**6/6 qualified**，全部原车位严格到达、碰撞0、规划余量0——#131 166步、#132 388步、#139 400步、#161 238步、#279 176步、#296 230步（末位置0.187–0.482 m，均在逐车位容差内）。
- 真实NN 14例：**5成功（#0/#168/#172/#267/#292）**、8振荡、1超时（#296）；碰撞0、规划余量0、干预3,898步；zero_hold选择标签64步，实际首命令全零为2,768步，两者不可混用。
- 对照判据：相对LT25子集**5条旧成功全保持、零丢失** ✅；**新增成功0** ❌（要求≥1）；相对LT24子集**6条旧成功一条未恢复**（#131/#132/#139/#161/#279/#296）且净−5 ❌。故`reference_candidate_benefit=false`、`formal_joint_benefit=false`、`candidate_qualified=false`。
- 机制证据：非法消费窗口跳过2,768次，而参考候选仅尝试10次（全部接受、0不安全），实际被选中10个控制步（#292七步、#296三步）；真实回合中原零解补查3,085次，其中307次选中原400样本中的扩展安全样本（1,922是LT25全300统计，不属于LT26），64次zero_hold标签、实际零命令2,768步、7次终端完整转向。候选入口仅10步，不能据计数直接推断这些步是否决定成败。
- 判读：六教师证明这些状态的**控制单元确实可解**（安全、低成本、严格到达），但同一机制在真实NN闭环中既不新增成功也不恢复丢失成功——控制单元可解 ≠ 闭环可救。v2门槛按预声明判负，不准备正式300、不晋升、不替换默认控制器；仍为GT静态确定性条件实验，`production_qualified=false`。


## 完成反馈后的独立复核（18:37:09开始）

Agent重新运行inspect通过，174源/38输入和60个NN产物身份一致；教师14产物逐SHA/6分母核验，teacher/smoke配置与原冻结配置逐字节相同。审计code/NN报告SHA/教师报告SHA/6+14分母核对一致。console.log有正常结束标记；未逐阶段记录原生退出数字，完成标记与Invoke-LT26Python非零即中断逻辑支持连续阶段正常完成，不另造退出码证据。

3.440秒独立重算20回合共8,517个实际转移/完整H10/原车位首次联合到达与消费窗口合法性，全部通过；原400样本/候选成本复用已核身份的80.336秒用户独立审计，未重复求解或NN推理。

新证据：14条中12条动作与前后态逐值完全等于LT25，包括#131/#132/#139/#161/#279五条旧成功损失；只有#292/#296发生变化。292保持成功、327→332步；296末位置误差8.25→3.03m但仍超时。六个原损失任务合计2,065步非法窗口跳过，其中五条完全未触发新增候选；修复控制覆盖并未改善这些真实参考。

九个失败回合最后一次完整NN参考全部不满足0.2m连接扫掠；其中八条完整输出中没有满足原位置+航向联合标准的点，#296有联合达标点但整条参考仍不安全。该结果定位到真实NN向MPC交付参考的可执行性/末段目标支撑缺口，尚不能区分上游融合质量与轨迹预测内部的具体责任，也不证明所有失败仅有一个原因。短滚动窗口本身不要求包含终点；这里检查的是末轮完整输出。

结论：LT26验收完整完成但质量门槛失败，不晋升、不创建全300。停止继续堆叠控制候选，下一步优先针对失败访问状态的实际参考，在轨迹预测及NN→MPC耦合验收中检查完整车体可行性与近终端可达支撑。证据：runs/verification/lt26-independent-review-20261026/{report.json,review.md}。
