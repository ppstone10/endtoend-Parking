# LT-20261019-01：训练后69条闭环配对评测

状态：**已运行完成（2026-10-19 09:46–09:57，Agent 代跑并已核验）**。8 组 × 69 条全部 completed，未通过 §8 预声明筛选，对照已停止；输出目录 `runs/validation/recovery-balance-ab-20261019-regression` 不再重复跑。

## 1. 已核验的训练结果

R1/R2均完成4轮，报告耗时约6.36/6.17分钟。8份epoch快照及2份last已实际加载核对，摘要、各轮400/800/1200/1600优化器更新、固定阈值约0.26、父模型/输入/代码/配置/环境身份均一致，无自动校准、部署或残留锁。

训练期间自由反馈、按GT有效长度计算的验证ADE如下。它不是闭环成功率，也不能直接替代其他开环脚本的停止截断口径：

| 轮次 | R1 ADE（m） | R2 ADE（m） |
|---|---:|---:|
| 1 | 0.1867 | 0.2022 |
| 2 | 0.1964 | 0.1850 |
| 3 | 0.1938 | 0.1817 |
| 4 | 0.1653 | 0.1951 |

结果随轮次交叉，不按末轮、最低loss或ADE单独选模型。审计文件为训练目录下 `agent-training-audit-20261019.json`。

## 2. 本次范围与固定协议

- 8个快照，各69条，共552回合，按同轮R1→R2顺序执行，不并行。
- E5：模拟传感器→网络→原MPC，无安全过滤、无专家回退，不改阈值或控制器。
- 数据为原 `tracked_pivot_v8_3000/val.npz`，固定索引为 `runs/validation/independent-acceptance/p2-regression/protocol.json`。
- 全局控制seed0、replan_every10、max_steps600，原MPC dt0.1/horizon10，模型dt0.2；OMP/MKL线程各1。
- R0在这69条为36成功/27碰撞/6振荡，已与原报告及此前逐条复现核对，不重复跑R0。
- 这是含已知失败的回归集，不代表完整验证分布或独立泛化。不跑S9、独立300条或完整主300条，不自动训练、校准或部署。

## 3. 输入、输出与预计资源

- 工作目录：`D:\program\endtoend Parking`。
- Python：`D:\conda\envs\endtoend-parking\python.exe`。
- 训练目录：`runs/training/recovery-balance-ab-20261018-v2`。
- 冻结清单：训练目录下 `regression-handoff-20261019.json`，SHA256=`6811b2a9effd082142f929ee2c5ced3ea60a53b28e7a8647ad8c815d25f1f797`。它记录83个输入/代码/快照文件摘要、8个作业、期望索引及运行环境；8个模型均已严格构造加载，CLI的help预检通过。
- 输出根目录：`runs/validation/recovery-balance-ab-20261019-regression`，当前不存在。每组独立保存report.json及console.log。
- 预计15～40分钟，依据为此前同69条单组约1～2分钟，新模型可能有更多600步超时/振荡，因此预留到约5分钟/组；本轮尚未实测总耗时。
- CPU串行，建议4GB可用内存、1GB磁盘，不需要GPU；保持接电、避免休眠。不并行启动训练或第二份评测。

不要修改代码、输入、环境或快照，不覆盖训练/R0产物，不使用best.pt替换指定快照。

## 4. 先执行只读预检

在PowerShell执行下段，保留同一窗口的变量供下一段使用。此段不启动评测：

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$handoffPath = 'runs/training/recovery-balance-ab-20261018-v2/regression-handoff-20261019.json'
$expectedHandoffHash = '6811b2a9effd082142f929ee2c5ced3ea60a53b28e7a8647ad8c815d25f1f797'
if ((Get-FileHash -LiteralPath $handoffPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHandoffHash) {
    throw '交接清单变化：停止并反馈Agent。'
}
$evaluationHandoff = Get-Content -LiteralPath $handoffPath -Raw -Encoding UTF8 | ConvertFrom-Json
function Assert-EvaluationInputs {
    foreach ($frozenFile in $evaluationHandoff.files) {
        if (-not (Test-Path -LiteralPath $frozenFile.path -PathType Leaf)) {
            throw "输入缺失：$($frozenFile.path)"
        }
        if ((Get-FileHash -LiteralPath $frozenFile.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $frozenFile.sha256) {
            throw "输入/代码/快照摘要变化：$($frozenFile.path)"
        }
    }
}
Assert-EvaluationInputs
if (Test-Path -LiteralPath $evaluationHandoff.output) {
    throw '输出目录已存在：不要覆盖或直接重跑，请将已有产物交给Agent。'
}
& 'D:\conda\envs\endtoend-parking\python.exe' -B -c "import json,sys,numpy,torch; from pathlib import Path; r=json.loads(Path('runs/training/recovery-balance-ab-20261018-v2/regression-handoff-20261019.json').read_text(encoding='utf-8')); actual={'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'torch':str(torch.__version__)}; assert actual==r['runtime'], 'runtime changed'; print('Runtime verified')"
if ($LASTEXITCODE -ne 0) { throw '运行环境不符：停止。' }
Write-Host 'PRECHECK PASSED: 8 snapshots, 69 cases each. No evaluation started.'
```

## 5. 再执行正式评测

只有预检通过才执行；下段顺序完成全部8组，出错即停止，不会扩大到300条：

```powershell
Assert-EvaluationInputs
New-Item -ItemType Directory -Path $evaluationHandoff.output | Out-Null
foreach ($evaluationJob in $evaluationHandoff.jobs) {
    Assert-EvaluationInputs
    if (Test-Path -LiteralPath $evaluationJob.output) {
        throw "作业目录已存在，禁止覆盖：$($evaluationJob.output)"
    }
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
    Write-Host "COMPLETE $($evaluationJob.id)"
}
Write-Host 'ALL 8 REGRESSION JOBS COMPLETED. STOP AND RETURN TO AGENT.'
```

## 6. 正常进度、异常与恢复

每组逐回合输出 `[1/69]` 到 `[69/69]`，最后打印汇总和报告位置。回合碰撞/超时属于结果，不等于程序故障；不能因为指标不好而改参数或只留下成功组。

出现Traceback、非0退出码、摘要不符、复原失败、索引不符时停止，保留目录，反馈作业名和console.log末尾。不要启动第二窗口、删除输出或覆盖重跑。此入口没有逐回合恢复；Agent核验已完成报告后，另给仅执行剩余作业、为中断组换新目录的步骤。输出根目录一旦存在，第4节会主动拒绝整个清单重跑。

保持训练快照不变，不续训、不校准。无异常但某组长时间无输出时，先查看对应console.log及进程状态，再反馈，不盲目重开。

## 7. 跑完后交回

回复：

> LT-20261019-01八组评测跑完，结果在 runs/validation/recovery-balance-ab-20261019-regression，请核验。

保留8组report.json/console.log、原训练目录及冻结清单。Agent检查协议、任务身份、分母、快照、失败分类、分场景回归和恢复/丢失任务，再做同轮R1/R2及R0对照，不用只贴百分比。

## 8. 扩大条件，不自动执行

沿用预先声明的工程筛选：同一轮R2成功数至少为 `max(36, 同轮R1成功数)+3`，碰撞数至多为 `min(27, 同轮R1碰撞数)-3`；R0的36条原成功至少保留33条，每场景R0原成功数不得少超过1条。

该筛选不是正式≥70%/≤10%验收或统计显著性证明。4轮无可信收益则停止本对照，不自行加epoch或调阈值。有合格候选后，Agent再交独立300条选型、原主300条与控制seed复核的人工计划，由用户执行。

## 9. 运行结果（2026-10-19 09:46–09:57，Agent 代跑）

- 只读预检通过后，8 组串行完成：各组 69.5–86.9 s，合计 9.9 分钟；8/8 `completed`、69/69、复原失败 0、任务索引与协议字段逐组核对一致。
- 结果（成功/碰撞/振荡）：R0 36/27/6；R1 11/46/12、39/25/5、21/43/5、21/42/6；R2 38/27/4、42/24/3、32/31/6、25/33/11。
- 最好一组 R2-epoch0002 = 42 成功 / 24 碰撞 / 3 振荡（60.9%/34.8%）。第 2 轮成功门槛刚好达标（42），碰撞 24 超过 ≤22 的要求；第 1/3/4 轮两项均未达标。保留条件也不满足：R0 的 36 条成功只保留 31 条（要求 ≥33），且 S4 由 6 条降到 1 条（丢失 5 条，全部转碰撞），超过每场景 ≤1 条的界限。
- R2 每轮都优于同轮 R1，但第 2 轮净增仅 +3 成功/+1 碰撞，接近本轮波动量级；单控制 seed，不能据此认定数据配比收益。
- 结论：未通过预声明筛选，**停止本对照**。不追加 epoch、不调阈值/损失、不进入独立 300 条或主 300 条验收，不替换部署权重；主成绩 68.3%/25.7% 未变。逐组指标与退化归因见 `runs/validation/recovery-balance-ab-20261019-regression/summary.md`。
- 若继续，需先解释 S4 卸载区退化并重新声明单变量与门槛，另行交付长任务计划。

## 10. 独立产物复核（2026-10-19）

重新核对83个冻结文件、8个报告的全部552条任务身份/容差/控制协议/计数，并按BOM识别UTF-16日志验证完整性；预声明四项条件逐项计算，合格轮次为空。最好R2第2轮保留31/36条原成功，丢失164、165、168、170、175，均为S4且转碰撞。R1同轮S4也为1/12，因此不能把退化全部归因于新的数据配比；单seed结果不足以量化波动量级。仍不扩大、不部署、不追加训练。

独立证据：输出目录下 `agent-regression-audit-20261019.json` 与 `agent-regression-analysis-20261019.md`。原报告及既有summary保留，本次没有重跑评测。
