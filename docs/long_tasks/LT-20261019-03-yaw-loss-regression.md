# LT-20261019-03：航向损失训练后的69条闭环配对

状态：**已运行完成（2026-10-19 11:21–11:31，Agent 代跑并已核验）**。8 组 × 69 条全部 completed，四轮均未通过 §7 预声明筛选，yaw wrapped 单变量对照已停止；输出根 `runs/validation/yaw-loss-ab-20261019-regression` 不再重复跑。

## 1. 审核结果与目的

W0/W1各4轮完成，8份epoch快照和2份last已实际加载，参数/优化器有限、每轮400步更新、历史完整、父模型/数据/执行代码/配置/运行环境身份一致，阈值固定约0.26，没有自动校准/部署、错误日志或残留锁。W0每轮权重与旧R2同轮逐参数完全相同；W1同轮权重与W0不同，确认唯一主要变量是航向损失模式。

| 轮次 | W0 val ADE（m） | W1 val ADE（m） |
|---|---:|---:|
| 1 | 0.202248 | 0.208945 |
| 2 | 0.185008 | 0.186681 |
| 3 | 0.181743 | 0.181826 |
| 4 | 0.195090 | 0.197576 |

ADE按自由反馈、GT有效长度计算，非闭环指标；W1四轮均略高，最大差约0.00670m。目标函数已变，不能用更低loss宣布改善。审核原始证据：训练目录下 `agent-training-audit-20261019.json`。

## 2. 范围、输入与固定协议

- 8组，每轮W0→W1，轮次1→4，共552回合，串行执行。不得按开环指标跳过某轮或换用best/last。
- 原val：`data/task_dataset/tracked_pivot_v8_3000/val.npz`。固定69个已知案例：`runs/validation/independent-acceptance/p2-regression/protocol.json`。R0为36成功/27碰撞/6振荡，引用原报告，不另跑R0。
- E5：模拟传感器→网络→原MPC，无安全过滤、无专家回退。全局控制seed0，replan_every10，max_steps600；原MPC dt0.1/horizon10，模型dt0.2，停止阈值0.26000000000000006，OMP/MKL线程各1。
- 工作目录：`D:\program\endtoend Parking`。唯一Python：`D:\conda\envs\endtoend-parking\python.exe`。运行环境与训练相同。
- 训练根：`runs/training/yaw-loss-ab-20261019-v2`，manifest SHA256=`0a29bf3172e311b1a48017923cb18ccaa560ecd8c314d04318f2ab781af343be`。
- 新评测冻结清单：训练根下 `regression-handoff-20261019.json`，SHA256=`cb3c4b5389506a0417f878925997bcb80db45fc00b65719091e0e2971d401309`。103个输入/代码/快照/审核文件摘要，8个作业、期望索引、环境和固定协议已写入；实际代码摘要包含未提交代码，不以Git HEAD代替。
- 输出根：`runs/validation/yaw-loss-ab-20261019-regression`，交付时不存在；每组独立 `report.json` 与 `console.log`。

这是含已知失败的回归筛选，不代表完整验证集成功率或独立泛化。本计划不运行独立300条、主300条或S9，不训练、不调阈值、不校准、不部署。已确认W0权重复现旧R2，本阶段仍按预声明保留全部同epoch配对，不改变比较口径。

## 3. 耗时与资源

8组估计 **10～25分钟**。依据为前轮同协议552回合实际约9.9分钟；新模型若有更多600步失败，会增加耗时，本次尚未实测。CPU串行，无需GPU，建议4GB可用内存和1GB磁盘余量（非实测峰值）。电脑接电、避免休眠，不并行训练或另一份评测。

不要修改代码、配置、输入、依赖环境或快照；不要覆盖任何旧结果/训练模型，身份变化会拒绝运行。

## 4. 第一步：只读预检

在同一PowerShell窗口执行并保留变量；此段不会启动评测：

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$handoffPath = 'runs/training/yaw-loss-ab-20261019-v2/regression-handoff-20261019.json'
$expectedHandoffHash = 'cb3c4b5389506a0417f878925997bcb80db45fc00b65719091e0e2971d401309'
if ((Get-FileHash -LiteralPath $handoffPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHandoffHash) {
    throw '交接清单已变化：停止并反馈Agent。'
}
$evaluationHandoff = Get-Content -LiteralPath $handoffPath -Raw -Encoding UTF8 | ConvertFrom-Json
function Assert-EvaluationInputs {
    foreach ($frozenFile in $evaluationHandoff.files) {
        if (-not (Test-Path -LiteralPath $frozenFile.path -PathType Leaf)) { throw "输入缺失：$($frozenFile.path)" }
        if ((Get-FileHash -LiteralPath $frozenFile.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $frozenFile.sha256) {
            throw "输入/代码/快照摘要变化：$($frozenFile.path)"
        }
    }
}
Assert-EvaluationInputs
if (Test-Path -LiteralPath $evaluationHandoff.output) { throw '输出根已存在：禁止覆盖或直接重跑，先交给Agent核验。' }
& 'D:\conda\envs\endtoend-parking\python.exe' -B -c "import json,sys,numpy,torch; from pathlib import Path; r=json.loads(Path('runs/training/yaw-loss-ab-20261019-v2/regression-handoff-20261019.json').read_text(encoding='utf-8')); actual={'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'torch':str(torch.__version__)}; assert actual==r['runtime'], 'runtime changed'; print('Runtime verified')"
if ($LASTEXITCODE -ne 0) { throw '运行环境不符：停止。' }
Write-Host 'PRECHECK PASSED: 8 snapshots, 69 cases each. No evaluation started.'
```

## 5. 第二步：正式评测

仅预检通过后，在同一窗口执行；出错即停，不会自动扩大或重试：

```powershell
Assert-EvaluationInputs
if (Test-Path -LiteralPath $evaluationHandoff.output) { throw '输出根已存在，禁止重复启动。' }
New-Item -ItemType Directory -Path $evaluationHandoff.output | Out-Null
foreach ($evaluationJob in $evaluationHandoff.jobs) {
    Assert-EvaluationInputs
    if (Test-Path -LiteralPath $evaluationJob.output) { throw "作业目录已存在：$($evaluationJob.output)" }
    New-Item -ItemType Directory -Path $evaluationJob.output | Out-Null
    Write-Host "START $($evaluationJob.id)"
    & 'D:\conda\envs\endtoend-parking\python.exe' -u -B scripts/run_validation_experiment.py --experiment E5 --data $evaluationHandoff.data --model $evaluationJob.checkpoint --indices-file $evaluationHandoff.indices --samples 0 --seed 0 --replan-every 10 --max-steps 600 --output $evaluationJob.output 2>&1 | Tee-Object -FilePath $evaluationJob.log
    if ($LASTEXITCODE -ne 0) { throw "作业失败：$($evaluationJob.id)。停止并反馈Agent。" }
    $reportPath = Join-Path -Path $evaluationJob.output -ChildPath 'report.json'
    $evaluationReport = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($evaluationReport.status -ne 'completed' -or $evaluationReport.overall.evaluated_samples -ne 69 -or $evaluationReport.overall.reconstruct_failures -ne 0) {
        throw "结果不完整：$($evaluationJob.id)。不得缩小分母。"
    }
    if (($evaluationReport.protocol.evaluated_indices | ConvertTo-Json -Compress) -ne ($evaluationHandoff.expected_indices | ConvertTo-Json -Compress)) {
        throw "实际任务索引不符：$($evaluationJob.id)"
    }
    if ($evaluationReport.experiment.name -ne 'E5' -or $evaluationReport.experiment.safety_mode -ne 'none' -or $evaluationReport.protocol.checkpoint -ne $evaluationJob.checkpoint) {
        throw "实验/模型身份不符：$($evaluationJob.id)"
    }
    Write-Host "COMPLETE $($evaluationJob.id)"
}
Write-Host 'ALL 8 REGRESSION JOBS COMPLETED. STOP AND RETURN TO AGENT.'
```

## 6. 进度、异常与恢复

每组依次显示 `[1/69]` 至 `[69/69]`，最后输出汇总和报告位置。碰撞/振荡/超时属于评测结果，不是程序异常，不能据此修改参数或重跑挑结果。`console.log`由PowerShell保存，Windows PowerShell可能为UTF16，原样保留。

Traceback、非0退出码、摘要/任务索引错误、复原失败时停止，保留目录并反馈作业名、错误和日志尾部。该入口无逐回合续跑，不能直接重新执行整个计划。Agent核验已完成组后，另给仅剩余作业的计划；中断组使用新目录保留旧证据。不得删目录、覆盖报告、自动重试或开第二个评测进程。

若长时间无输出，先只读查看当前作业日志/进程，再反馈；不要盲目重开。

## 7. 交回文件与判断门槛

完成后回复“W0/W1八组评测完成”，保留输出根下8组report.json/console.log、整个训练根、冻结清单和审核文件。**到此停止，不自动训练、扩大或部署。**

Agent复核分母、全部任务身份/控制协议/快照、失败分类及恢复/丢失案例，尤其S4的164/165/168/170/175；重新计算相同epoch的四项联合门槛（沿用训练前声明）：

1. W1成功数 ≥ `max(36, 同轮W0成功数)+3`。
2. W1碰撞数 ≤ `min(27, 同轮W0碰撞数)-3`。
3. 至少保留33/36条R0原成功。
4. 每场景R0原成功丢失≤1条。

同一候选必须全部满足，不能拼接不同轮次；四轮都失败则停止本单变量，不自行加轮、调阈值或混入其他改动。通过也不是正式达标，后续独立选型和主300条须另交人工计划；最终仍是同一候选成功率≥70%、碰撞率≤10%。

## 8. 运行结果（2026-10-19 11:21–11:31，Agent 代跑）

- 只读预检通过后，8 组串行完成：11:21:39–11:31:48，10.2 分钟；8/8 `completed`、69/69、复原失败 0、索引与协议逐组核对一致。
- 结果（成功/碰撞/振荡）：R0 36/27/6；W0 38/27/4、42/24/3、32/31/6、25/33/11；W1 38/30/1、42/24/3、31/34/4、27/34/8。W0 与上一轮 R2 计数完全一致，说明链路确定性复现。
- 四项门槛（同轮）：第 1 轮需 W1 ≥41/≤24，实际 38/30；第 2 轮需 ≥45/≤21，实际 42/24；第 3 轮需 ≥39/≤24，实际 31/34；第 4 轮需 ≥39/≤24，实际 27/34。保留数 20/30/21/18 均低于 33，每场景丢失同样超界。
- 结论：wrapped 在所有轮次都没有降低碰撞（第 1、3、4 轮碰撞反而 27→30、31→34、33→34）；振荡减少（4→1、6→4、11→8）但同时转成碰撞或未转成功。S4（1～7/12）与 S8（W1 第 2 轮 0/6）是两臂共同瓶颈，退化与 yaw 模式无关。
- 按本节：四轮全败，停止该单变量；不追加轮次、不调阈值、不进入 300 条、不替换部署。主成绩 68.3%/25.7% 未变。
- 逐组指标、分场景与门槛对照见 `runs/validation/yaw-loss-ab-20261019-regression/summary.md`；后续建议先做 S4/S8 同状态回放与参考/规划归因，另交长任务计划。

## 9. 独立复核与结论边界

103个冻结文件、552条任务身份/控制协议/容差、日志和全部计数复核通过。W0逐回合数值（不含推理耗时）复现旧R2，不只是计数相同。W1四轮保留原成功20/30/21/18，每场景原成功丢失也均超界，通过轮次为空。

更正§8“退化与yaw模式无关”的过强判断：第2轮W1救回3条S3（113/117/126）但丢掉S6的246和S8的292/294，虽总计数相同，具体行为不同；其他轮次S4也有恢复/丢失。只能认定本次wrapped单变量没有联合工程收益、没有解决第2轮5条S4原成功转碰撞，不能认定角度处理与退化无关或无任何局部影响。

两臂S3第3/4轮均0/12成功，后期共同退化仍需机制诊断。停止追加训练，先审计真实失效状态和实际MPC窗口。本轮未启动下一项长任务；独立审计与分析位于输出根下 `agent-regression-audit-20261019.json`、`agent-regression-analysis-20261019.md`。
