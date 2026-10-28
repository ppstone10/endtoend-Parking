# LT30 · 环视修复输入的配对训练

状态：**已完成**（用户要求执行；两臂各20轮训练 + 两个原生`last.pt`开环300条诊断，四段入口退出码0）。开环平移误差显著下降、航向与长度误差略退化，**闭环收益与旧能力保持未测**；结果见文末「执行结果」。正式配置、入口、路径和真实小样本训练链路此前已预检。关联规范：[SURROUND-FIT-001..003](../../spec/surround_input_training.md)。

## 目的、唯一主要变量与边界

检验修复后的相机target能否转化为现有net-v1的学习收益。两臂同seed冷启动、同初始参数/监督/训练剂量，仅改变target生成版本及必要的输入声明和路径：旧`legacy_overlay_v1`对照与LT29的`per_view_v2`。新版包括逐视角标定/投影、正深度裁剪及逐视角噪声采样，不声称只改变一个矩阵或保持旧噪声实现逐值相同。

沿用既有冷启动对照和Trainer方法，不新增网络/损失/优化器或控制策略。旧初始化入口严格要求模型配置完全一致，本计划不放宽它、不迁移旧权重、不只改checkpoint声明。冷启动对照用于分离输入版本与训练血统，**不能替代相对原模型的旧能力保持验收**。本阶段不运行闭环、不搜索停止偏置、不回灌验证集、不晋升模型。

## 固定身份、配置和参数

- 工作目录：`D:\program\endtoend Parking`。Python仅`D:\conda\envs\endtoend-parking\python.exe`；预检环境Python3.12.13、torch2.13.0+cpu、numpy2.5.2、PyYAML6.0.3、matplotlib3.11.1。
- 冻结文件：`runs/prepared/lt30-surround-training-20261027/handoff.json`，SHA256 **`343fad47e224b00165d77266ee511e28bc6c8f2a671d93d3212e54a86ca1f164`**。登记156代码、16输入/证据和2份配置。工作树未提交，文件SHA作为本实验版本依据。
- 旧臂配置：`configs/training/net-v1-surround-legacy-control.yaml`。训练为原`data/task_dataset/v40_dagger_collision/train-balanced.npz`（2918条，SHA `b9531bbf…`），验证为原`tracked_pivot_v8_3000/val.npz`（300条，SHA `ed0dcb12…`）。
- 新臂配置：`configs/training/net-v1-surround-per-view.yaml`。训练为`runs/verification/surround-target-rebuild-lineage-20261027/train/train-balanced.npz`（SHA `137b13a6…`），验证为同根`val/val.npz`（SHA `b8250ede…`）。完整SHA在handoff中。
- 原模型参考：`runs/verification/input-contract-corrected-20261023/deployment-image-contract.pt`（SHA `e917fed2…`）；原E5逐任务报告`runs/validation/live-20261018-170806/full/E5/report.json`，205成功/77碰撞/18未到达。只冻结供后续保持验收，不拿来初始化或替代本轮配对控制。
- 两臂共同：net-v1、hidden128、5×160×160 BEV、horizon320、dt0.2、image target、semantic height；seed20260827；CPU单个intra-op线程；batch8；Adam lr3e-4、clip1；20 epoch、patience20，确保没有提前停训造成的剂量差。
- 损失沿既有冷启动配方：累计停止BCE权重0.2、不平衡正负停止项；endpoint_alignment0.5/tail8；terminal_pose1/yaw10；collision_loss0。teacher forcing从1.0在10轮降到0.2；early-stopping从epoch索引10开始，但patience20大于本轮监控窗口。
- `balance_recovery_batches=false`：每轮精确遍历2918条，含旧2400条与518恢复帧；两臂同排列，不重采样或按task_id去重。每轮365次更新，20轮共7300次。关闭闭环选型。
- **主要比较固定最终`last.pt`，原生停止阈值0.5**，epoch从0计应为19。runner会另外保存`best.pt`和基于val校准的`deployment.pt`；这两项只作辅助，不能把阈值校准或不同best轮次当成输入修复的原生收益。

## 已完成的短时预检

[真实预检报告](../../runs/verification/lt30-training-precheck-20261027/report.json)：共9.790秒。每臂真实8条训练（源索引0/7/23/100/2400/2413/2682/2917，覆盖恢复标签边界）和4条验证（0/7/100/299），正式模型/完整320点输出，各1次真实Adam更新，保存last/best/deployment、报告和曲线，权重有限、原生阈值0.5。两臂初始参数SHA完全相同，非target和监督相同，epoch0/10/19排列一致，交叉输入版本拒绝。完整正式目录未创建。单次更新的误差不用于评价学习效果。

配置结构与路径加载通过，轻量Spec结构检查通过；两个`analyze_predictions.py`入口在各自4条预检验证数据/原生last上实际运行成功，计划4段PowerShell通过原生解析器语法检查。预检使用的子集仅存于`runs/verification/lt30-training-precheck-20261027`，**不得把其中的模型当作正式训练结果**。

## 耗时与资源

暂估每臂20–60分钟，两臂加原生开环诊断约40–130分钟；繁忙机器或内存换页可更久。依据是本机单线程、完整模型/输出的8条真实更新与自由滚动预检，未实测完整epoch，因此估算不确定，首个epoch输出可用于校正预期，不能据此临时改变20轮剂量。

CPU运行，无GPU；建议空闲内存至少8GB，保守预留10–12GB。完整训练BEV展开约1.49GB、验证约0.154GB，shuffle及张量/激活会额外占内存。磁盘建议预留1GB；两个任务顺序执行，不能并行或后台自动续跑。

## 有依赖顺序的PowerShell命令

在同一个PowerShell窗口执行。身份/版本/产物占用不符立即停止，不修改SHA绕过。任一步报错均不要继续后续步骤。

### 1. 环境与身份预检

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt30Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt30Handoff = 'runs/prepared/lt30-surround-training-20261027/handoff.json'
if (-not (Test-Path -LiteralPath $lt30Python)) { throw '指定conda Python不存在' }
if ((Get-FileHash -LiteralPath $lt30Handoff -Algorithm SHA256).Hash.ToLowerInvariant() -ne '343fad47e224b00165d77266ee511e28bc6c8f2a671d93d3212e54a86ca1f164') { throw 'LT30 handoff身份变化' }
$lt30Frozen = Get-Content -LiteralPath $lt30Handoff -Raw -Encoding UTF8 | ConvertFrom-Json
function Assert-LT30Frozen {
    foreach ($group in @($lt30Frozen.code, $lt30Frozen.inputs, $lt30Frozen.configs)) {
        foreach ($entry in $group.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "冻结文件漂移：$($entry.Name)" }
        }
    }
}
Assert-LT30Frozen
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
@'
import json, sys
from pathlib import Path
import torch, numpy, yaml, matplotlib
frozen = json.loads(Path('runs/prepared/lt30-surround-training-20261027/handoff.json').read_text(encoding='utf-8'))
actual = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=numpy.__version__, yaml=yaml.__version__, matplotlib=matplotlib.__version__)
assert actual == frozen['versions'], (actual, frozen['versions'])
assert Path(sys.executable).resolve() == Path(frozen['python']).resolve()
assert torch.get_num_threads() == 1
print('LT30 environment and single-thread checks passed', actual)
'@ | & $lt30Python -X utf8 -
if ($LASTEXITCODE -ne 0) { throw 'LT30环境检查失败' }
$lt30Root = $lt30Frozen.run_root
if (Test-Path -LiteralPath $lt30Root) { throw '正式根目录已存在，保留产物并反馈Agent，勿覆盖或自行续跑' }
New-Item -ItemType Directory -Path $lt30Root | Out-Null
Copy-Item -LiteralPath $lt30Handoff -Destination (Join-Path $lt30Root 'handoff.json')
```

### 2. 完整训练旧输入控制臂

```powershell
Assert-LT30Frozen
& $lt30Python -u -X utf8 scripts/train_model.py --config configs/training/net-v1-surround-legacy-control.yaml 2>&1 | Tee-Object -FilePath (Join-Path $lt30Root 'legacy-control.console.log')
$lt30LegacyExit = $LASTEXITCODE
if ($lt30LegacyExit -ne 0) { throw "LT30旧臂训练失败：$lt30LegacyExit；停止后续阶段" }
$lt30LegacyReport = Get-Content -LiteralPath (Join-Path $lt30Root 'legacy-control/report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($lt30LegacyReport.status -ne 'completed' -or $lt30LegacyReport.history.train_loss.Count -ne 20 -or $lt30LegacyReport.data.train_samples -ne 2918 -or $lt30LegacyReport.data.val_samples -ne 300) { throw '旧臂完成状态/轮数/分母不符' }
```

### 3. 完整训练新版输入臂

```powershell
Assert-LT30Frozen
& $lt30Python -u -X utf8 scripts/train_model.py --config configs/training/net-v1-surround-per-view.yaml 2>&1 | Tee-Object -FilePath (Join-Path $lt30Root 'per-view.console.log')
$lt30NewExit = $LASTEXITCODE
if ($lt30NewExit -ne 0) { throw "LT30新版训练失败：$lt30NewExit；停止后续阶段" }
$lt30NewReport = Get-Content -LiteralPath (Join-Path $lt30Root 'per-view/report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($lt30NewReport.status -ne 'completed' -or $lt30NewReport.history.train_loss.Count -ne 20 -or $lt30NewReport.data.train_samples -ne 2918 -or $lt30NewReport.data.val_samples -ne 300) { throw '新版完成状态/轮数/分母不符' }
```

### 4. 原生last模型的完整300条分组诊断

对各自匹配的val运行，不交叉喂入。这里使用`last.pt`，不能改成deployment。

```powershell
Assert-LT30Frozen
& $lt30Python -u -X utf8 scripts/analyze_predictions.py --data data/task_dataset/tracked_pivot_v8_3000/val.npz --checkpoint (Join-Path $lt30Root 'legacy-control/last.pt') --output (Join-Path $lt30Root 'analysis-legacy-native') --batch-size 8 --device cpu 2>&1 | Tee-Object -FilePath (Join-Path $lt30Root 'analysis-legacy-native.console.log')
$lt30LegacyAnalysisExit = $LASTEXITCODE
if ($lt30LegacyAnalysisExit -ne 0) { throw 'LT30旧臂原生诊断失败，停止' }
& $lt30Python -u -X utf8 scripts/analyze_predictions.py --data runs/verification/surround-target-rebuild-lineage-20261027/val/val.npz --checkpoint (Join-Path $lt30Root 'per-view/last.pt') --output (Join-Path $lt30Root 'analysis-per-view-native') --batch-size 8 --device cpu 2>&1 | Tee-Object -FilePath (Join-Path $lt30Root 'analysis-per-view-native.console.log')
$lt30NewAnalysisExit = $LASTEXITCODE
if ($lt30NewAnalysisExit -ne 0) { throw 'LT30新版原生诊断失败，停止' }
Assert-LT30Frozen
$lt30Completion = [ordered]@{ task='LT-20261027-30'; status='completed'; phase='training_and_openloop_only'; legacy_train_exit=$lt30LegacyExit; per_view_train_exit=$lt30NewExit; legacy_analysis_exit=$lt30LegacyAnalysisExit; per_view_analysis_exit=$lt30NewAnalysisExit; production_qualified=$false }
$lt30Completion | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $lt30Root 'completion.json') -Encoding UTF8
$lt30Artifacts = [ordered]@{}
Get-ChildItem -LiteralPath $lt30Root -File -Recurse | ForEach-Object {
    $relative = $_.FullName.Substring($lt30Root.Length + 1).Replace('\','/')
    $lt30Artifacts[$relative] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$lt30Artifacts | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $lt30Root 'artifacts.json') -Encoding UTF8
Write-Host 'LT30两臂训练与原生开环诊断完成；等待Agent复核，不启动闭环或下一训练'
```

## 正常进度、输出与成功标志

训练每epoch打印`epoch n/20`、train/val loss、自由滚动ADE/FDE、teacher比例。每臂到`epoch 20/20`后还要恢复best、val阈值校准及绘图，最终打印完成和report路径才表示该入口完成。初轮无输出可能正在解压/准备数据，不同时启动第二进程。

正式根目录：`runs/training/lt30-surround-input-20261027/`。两臂各有`last.pt`、`best.pt`、`deployment.pt`、`history.json`、`report.json`、`stop_threshold_calibration.json`、`training_curve.png/.pdf`；两个`analysis-*-native/`目录有report和分组/叠加图；根目录有四份console日志、handoff、completion、artifacts。

执行成功标志是上述文件齐全、两臂20轮、2918/300分母及四个入口退出0。**这不代表输入修复已有闭环收益，更不代表旧能力保持或生产合格。**训练根report的metrics使用校准后的best，原生last结果应从`analysis-*-native/report.json`读取。

## 异常停止、恢复、重跑与禁止覆盖

- 身份漂移、NaN/Inf、内存不足、异常或非零退出均停止，不启动下一阶段。Ctrl+C可中断，保留日志、last/best和已有报告；强制中断时末尾原子临时文件不作为checkpoint。
- 本计划不提供自动续训或自动重试。恢复会改变配置身份/输出路径，须先反馈Agent诊断并另行冻结恢复方案，不能随意填`resume_from`、删目录、改seed/epoch/批量或覆盖已有权重。
- 即使旧臂已完成，也不要重新执行根目录创建和旧臂命令；反馈已完成阶段，Agent决定是否只补缺失阶段。停止于分析阶段也无需重新训练。
- 禁止覆盖原数据、LT28失败现场、LT29产物、旧checkpoint/原E5成绩、预检子集、冻结配置和任一正式/失败目录。不得把验证300条作为新增教师或据本轮结果继续称其无偏泛化集；它是已见诊断集。

## 完成后反馈与Agent后续验收

回复“LT30执行完毕”或首个异常及当前阶段；保留根`completion.json/artifacts.json/handoff.json`、四份console、两臂报告/history/三个checkpoint/阈值校准和两个原生analysis报告，原地保存即可。

Agent将核对冻结身份、退出与完成状态、20轮/7300更新、完整2918/300分母、checkpoint输入版本/原生0.5、权重和loss有限、初始配对及数据谱系、所有产物SHA，再比较原生ADE/FDE、航向、停止/长度与场景×任务×机动×噪声回归。单seed开环改善只形成下一阶段证据，不宣布安全闭环收益。

后续完整闭环必须另立人工执行计划：新版真实per_view_v2→网络→原控制链路，逐任务检查真实外廓扫掠/0.2m规划余量/联合终点；对原205成功任务逐例保持，并沿用主联合门槛（成功≥70%、碰撞≤10%），任何未满足项都不得晋升。旧输入配对臂若需闭环，只能用认证旧感知快照，不能绕过当前版本检查。原有绝对L1门槛及五层验收缺口仍独立保留。

## 执行结果

- 身份：handoff SHA `343fad47…f164`与计划一致；156代码/16输入/2配置零漂移（运行前后各断言一次）。两臂配置除`camera_projection`与train/val路径外逐项相同；同seed 20260827冷启动（预检`paired_initialization_exact=True`、初参SHA `11fba796…`）。
- 训练（两臂均`completed`）：各20轮、2918训练/300验证、365批/轮=7300更新、`stopped_early=false`。旧臂耗时约33分钟、best_epoch=16；新臂约32分钟、best_epoch=19。
- 原生`last.pt`（epoch19、stop_threshold=0.5）开环300条：

| 指标 | legacy_overlay_v1 | per_view_v2 | 变化 |
|---|---|---|---|
| ADE | 0.3685 m | **0.2593 m** | **−29.6%** |
| FDE | 0.4976 m | **0.3187 m** | **−36.0%** |
| FDE p90 | 0.8675 m | 0.6538 m | −24.6% |
| 航向 MAE | 0.2056 rad | 0.2082 rad | +1.3%（略差） |
| 预测长度 MAE | 16.83 点 | 22.33 点 | +32.7%（变差） |
| 长度偏差 | +10.96 点 | +17.58 点 | 过冲增大 |
| stop_found_rate | 1.0 | 1.0 | 相同 |

- 分组（原生last）：场景除S8外均改善——S1 ADE 0.2216→0.1187、S3 0.3201→0.1417、S5 0.1850→0.0880，S4 0.6380→0.5353、S6 0.5919→0.4793改善较小，**S8 FDE 0.1199→0.1685变差**（n=11）；机动上**reverse大幅改善**（ADE 0.4231→0.2475、FDE 0.6450→0.3215），**forward FDE 反而变差**（0.2538→0.3141，ADE基本持平）；噪声三档ADE均改善，high档改善最小。
- 辅助诊断（不参与主比较）：报告的`metrics`用校准后的best——旧臂ADE 0.3775/FDE 0.4385（校准阈值0.40），新臂0.2593/0.3187（阈值0.18）；`deployment.pt`差异部分来自阈值校准，不能当作输入修复的原生收益。
- 产物：根目录`completion.json`（四段exit=0、`production_qualified=false`）与`artifacts.json`（36个登记文件，SHA零缺失零漂移，仅清单自身未登记）；两臂各含`last/best/deployment.pt`、`history.json`、`report.json`、`stop_threshold_calibration.json`、曲线；两个`analysis-*-native/`含report与分组/最差样本图；四份console日志+handoff齐全。
- 判读：修复后的逐视角target确实转化为**平移精度**的学习收益（ADE/FDE大幅下降，reverse尤其明显），但**航向精度与预测长度没有改善甚至变差**，前向机动的FDE与S8的FDE出现退化——所以这是"部分维度的开环拟合改善"，不是整体方法通过。按SURROUND-FIT-003，本阶段只形成下一阶段证据：**闭环碰撞/规划余量/联合到达与旧205成功保持全部未测**，`production_qualified`不变，不晋升、不替换默认入口、不串行启动下一长任务。
- 边界：仅单seed；300条val为已见诊断集，不构成无偏泛化证据；源数据、LT28失败现场、LT29产物、旧checkpoint与冻结配置均未改动；未训练新控制策略、未跑闭环、未回灌验证集；未add/commit/push。

## 完成反馈后的独立复核

身份/完整性核验通过：156代码/16输入/2配置、36项产物SHA均一致，两臂last为epoch19/Adam7300/原生0.5，日志和报告完成。重新执行两模型各300条原生静态推理并独立重算总体/场景FDE，正式报告指标一致。

新增真实停止点诊断：停止点平均位置误差0.973→1.695m，停止点联合位置/航向匹配28/300→20/300；专家长度处的联合匹配则115/300→205/300。新版269条输出超过专家长度、包含5986个后缀点，按实际损失组合核验其直接坐标梯度为0。L2内部存在专家长度监督/指标与模型停止长度输出的对齐缺口，需先补实际输出验收与对应训练约束；不能由开环平移改善推定到达/碰撞或旧能力通过。静态匹配数不是闭环成功率。证据：[独立复核报告](../../runs/verification/lt30-independent-review-20261027/review.md)。本轮未改训练代码/权重或执行后续长任务。
