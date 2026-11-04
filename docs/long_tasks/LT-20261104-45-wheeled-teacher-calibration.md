# LT-20261104-45 · 固定22任务轮式教师与执行校准

**状态：待用户运行。代码、冻结配置与短预检已具备；正式22任务run/inspect未执行。**

## 目的与唯一主要变量

在原任务几何、目标、容差和分片身份固定的条件下，检验新轮式专家是否能生成合法监督，并由同一轮式MPC在原600步预算内完成。主要研究变量是轮式运动/控制契约，不更换任务、目标或放宽阈值。

旧train2400/val300的静态必要条件审核均未通过当前轮式速度变化约束，其中855/105条还有后轴几何/曲率必要条件缺口。因此先校准教师，不直接训练或批量重建数据。失败只表示当前算法/有限候选/预算的结果，不表示真实矿卡任务全局无解。

## 固定输入与版本

- 工作目录：`D:\program\endtoend Parking`。
- 唯一运行环境：`D:\conda\envs\endtoend-parking\python.exe`，Python **3.12.13**、NumPy **2.5.2**。本任务不使用GPU、不训练、不读取BEV大数组。
- 代码基线HEAD `b6285f6`加本次轮式改造工作树；**以prepared内333项源码/输入SHA为实际版本依据**，提交Git不改变内容时不影响前检。源码内容变动则停止，不手工改SHA放行。
- prepared：`runs/prepared/wheeled-calibration-20261104-v1/config.json`。
- prepared SHA256：`c55ec765070c06ee3272912f6477078a89e2f86da0f31f773656dd6fde3ac661`。
- 原数据：`data/task_dataset/tracked_pivot_v8_3000/{train,val}.npz`及同目录manifest，全部SHA在prepared中；不读取test，不接受恢复样本，不覆写原metadata。
- 轮式参数：`configs/vehicles/wheeled_research_v1.json`，身份`122399edc4d42fcf1c616c4e25636891827a34127e27e0a6be01db36642a1003`；均为`research_assumption`，尚非真实矿卡标定。
- 原外廓6×3m/规划余量0.2m保持。中心与后轴显式转换；初始v=0，明确假设初始转角0，不能把旧omega解释成转角。
- dt=0.1；RS最多48个词族候选，巡航请求上限0.5m/s，规划3秒/最多600步；采用停车转轮与采样加减速时序。
- CEM：horizon=8、population=8、iterations=1、seed=7；实际执行max_steps=600，单例执行资源预算120秒。到达使用**各原车位**tol_pos/tol_yaw；请求方向里程占比至少50%。这些是固定基础配置，不作本轮扫参。
- GT完整几何审查包含曲线弓高与采样保守膨胀；没有感知安全、网络能力或真实制动结论。T5事件如存在仍未动态注入，报告显式标记静态。

## 固定任务面板

按每分片scene×requested maneuver选择最早源索引，不能替换失败任务。完整task ID及实际顺序见prepared.cases。

| 场景 | 机动 | train索引 | val索引 |
|---|---|---:|---:|
| S1 | forward | 0 | 0 |
| S1 | reverse | 2 | 1 |
| S2 | forward | 430 | 56 |
| S2 | reverse | 433 | 55 |
| S3 | reverse | 860 | 110 |
| S4 | forward | 1290 | 163 |
| S4 | reverse | 1293 | 162 |
| S5 | reverse | 1720 | 212 |
| S6 | forward | 1975 | 246 |
| S6 | reverse | 1978 | 245 |
| S8 | forward | 2315 | 289 |

共22唯一源任务。此面板只是迁移关口，不等于完整任务矩阵、主300或200例正式专家基线。

## 已完成短预检

- 88项聚合测试通过，包括轮式模型/执行/专家/审核和旧MPC/执行/RS/车辆回归。
- prepare与precheck通过：22任务、333冻结文件，0正式回合。
- 首个真实train#0短检及逐控制重放通过，耗时0.844秒，94步位姿到达、位置误差0.2714m、航向误差0，原请求方向比1.0，预测首步误差0，GT转移检查通过。
- 首例证据：`runs/verification/wheeled-calibration-smoke-20261104-v1.json`。它只覆盖1例，不能替代22任务结果。
- 静态审核正式证据：`runs/verification/wheeled-legacy-audit-20261104-v2.json`。无v2后缀的同日JSON因NumPy标量序列化失败是不完整现场，不能使用或覆盖。

## PowerShell：同一窗口依次执行

### 1．环境、冻结身份和防覆盖检查

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$wheelPython = 'D:\conda\envs\endtoend-parking\python.exe'
$wheelPrepared = 'runs/prepared/wheeled-calibration-20261104-v1/config.json'
$wheelOutput = 'runs/verification/wheeled-calibration-formal-20261104-v1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $wheelPython -X utf8 -c 'import platform,numpy; assert platform.python_version()=="3.12.13" and numpy.__version__=="2.5.2", "环境版本与计划不一致"'
if ($LASTEXITCODE -ne 0) { throw '环境检查失败，停止并反馈' }
if ((Get-FileHash -LiteralPath $wheelPrepared -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'c55ec765070c06ee3272912f6477078a89e2f86da0f31f773656dd6fde3ac661') { throw 'prepared身份漂移，停止并反馈' }
foreach ($wheelExisting in @($wheelOutput, ($wheelOutput + '-run.log'), ($wheelOutput + '-run-exit.json'), ($wheelOutput + '-inspect.log'), ($wheelOutput + '-inspect-exit.json'))) {
    if (Test-Path -LiteralPath $wheelExisting) { throw "已有产物，保留现场：$wheelExisting" }
}
& $wheelPython -X utf8 scripts/calibrate_wheeled_tasks.py --phase precheck --prepared $wheelPrepared
if ($LASTEXITCODE -ne 0) { throw '源码或输入前检失败，停止并反馈' }
```

### 2．运行22任务（不预先创建正式输出目录）

```powershell
if ((Test-Path -LiteralPath $wheelOutput) -or (Test-Path -LiteralPath ($wheelOutput + '-run.log'))) { throw '已有运行现场，不覆盖' }
& $wheelPython -X utf8 scripts/calibrate_wheeled_tasks.py --phase run --prepared $wheelPrepared --output $wheelOutput 2>&1 | Tee-Object -FilePath ($wheelOutput + '-run.log')
$wheelRunExit = $LASTEXITCODE
[ordered]@{ task='LT-20261104-45'; phase='run'; exit_code=$wheelRunExit } | ConvertTo-Json | Set-Content -LiteralPath ($wheelOutput + '-run-exit.json') -Encoding utf8
if ($wheelRunExit -ne 0) { throw '运行异常，保留所有产物并反馈，不重跑或继续inspect' }
```

### 3．完整产物与逐控制核验

```powershell
if (Test-Path -LiteralPath ($wheelOutput + '-inspect.log')) { throw '已有inspect日志，不覆盖' }
& $wheelPython -X utf8 scripts/calibrate_wheeled_tasks.py --phase inspect --prepared $wheelPrepared --output $wheelOutput 2>&1 | Tee-Object -FilePath ($wheelOutput + '-inspect.log')
$wheelInspectExit = $LASTEXITCODE
[ordered]@{ task='LT-20261104-45'; phase='inspect'; exit_code=$wheelInspectExit } | ConvertTo-Json | Set-Content -LiteralPath ($wheelOutput + '-inspect-exit.json') -Encoding utf8
if ($wheelInspectExit -ne 0) { throw '独立审核失败，保留现场并反馈' }
```

## 耗时与资源

粗估2–15分钟，依据首例0.844秒和当前少量短闭环；复杂环境、较长参考、规划失败候选以及机器负载会显著改变耗时，估计不可靠，因此按长任务交用户运行。22例规划预算合计66秒、执行资源预算合计44分钟，另有单步计算和I/O开销；这只是预算量级，不是严格进程墙钟上界。

CPU单线程，建议至少2GB可用内存、100MB新增磁盘空间。数据审核/任务复原不加载BEV大数组；输出仅配置、轨迹控制与审核记录。不要并行启动第二个作业。

## 输出与正常进度

- 正式目录：`runs/verification/wheeled-calibration-formal-20261104-v1/`。
- 内容：`config.json`、22份`NNN-split-index.json`、`report.json`、`artifacts.json`；两个日志和两份退出JSON位于目录同级。
- 正常stdout逐例显示`START i/22`、`DONE planned=... passed=...`；出现failed任务仍继续记录完整22分母，不替换样本。
- 作业完成：run退出0、report.status=completed且22条完整；inspect退出0、verified=true、cases=22。
- 科研关口通过：还需inspect的passed=22、gate_passed=true。**run/inspect退出0但gate_passed=false是完整负结果，不代表改造达标。** 不进入数据建设或训练。

## 异常停止、恢复与重跑

- 出现Python traceback、身份漂移、缺失文件或持续无进度时先停止并反馈日志，不改阈值/预算/源任务。
- 需要人工终止可按Ctrl+C；保留整个目录和日志。中断可能没有最终退出JSON/report/artifacts，inspect会拒绝不完整产物。
- 本版没有自动resume、重试或下一阶段。只有诊断后确定可重跑时，使用新目录，例如将`$wheelOutput`改为`runs/verification/wheeled-calibration-formal-20261104-r2`，重新执行前检和后续步骤，不删除旧目录。
- 若源码修复，旧prepared会明确拒绝；由Agent准备新的冻结版本与计划，不手工编辑旧SHA继续。
- 禁止覆盖：原train/val/test、manifest、履带检查点/报告、v1/v2静态审核、短检证据、prepared，以及任何已存在的正式输出和日志。

## 完成后反馈与Agent验收

反馈正式目录路径、两份退出JSON、run/inspect日志；若异常，说明最后一个START/DONE及错误。无需手工挑选成功案例。

Agent将核对任务/配置/源与代码身份、完整22分母、逐例teacher/actual状态和控制、原车位容差、请求机动、GT安全、日志完成状态及文件SHA，再分析失败的场景和原因。inspect重放model.step与几何审核，不声称重算全部CEM候选排名。仅22/22通过后再决定更大专家验证与必要重建计划；不自动训练，不宣称原主300、泛化或生产资格。
