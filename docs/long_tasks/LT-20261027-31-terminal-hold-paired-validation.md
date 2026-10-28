# LT31 · 专家终点保持训练配对验证

状态：**已完成（负结果）**。两臂各4轮/1460新更新、2918/300分母、两个原生300条诊断，四入口退出0；但预先声明的阶段筛选`stage_screen_passed=false`（12项仅2项通过），终点保持臂几乎全面退化。结果见文末「执行结果」。规范：[EMITTED-TERM-001..003](../../spec/emitted_terminal_alignment.md)。

## 目的、唯一主要变量与边界

验证专家终点保持能否减少网络实际发出轨迹末点的偏离，同时保持原有效前缀预测能力。两臂均从LT30修复输入模型的同一`last.pt`加载模型权重，以相同数据/顺序/优化器/学习率训练相同剂量，唯一主要变量为终点保持策略：control的`terminal_hold_weight=0`；hold的权重0.25，并同步将teacher forcing后缀填为专家末位姿。损失与teacher补齐共同构成该策略，不把它描述成仅比较一个损失系数。

尾部重复末帧参考了[Diffusion Policy官方采样实现](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/common/sampler.py)和[robomimic SequenceDataset](https://robomimic.github.io/docs/modules/dataset.html)的序列边界填充惯例；本项目独立实现，不引入依赖。这些资料不证明本训练假设有效，必须由配对结果决定。

保持原target/mask/停止BCE、网络结构、运行时停止规则和0.5阈值、MPC、车位容差及源数据不变。模型权重来自同版本per_view_v2，沿既有initialize_from只加载权重，不继承LT30的Adam7300/epoch进度。本阶段只训练与静态诊断，不运行闭环、不部署、不自动加剂量/换权重。

## 冻结身份与固定参数

- 工作目录`D:\program\endtoend Parking`；Python仅`D:\conda\envs\endtoend-parking\python.exe`。Python3.12.13、torch2.13.0+cpu、numpy2.5.2、PyYAML6.0.3、matplotlib3.11.1，CPU单intra-op线程。
- 冻结`runs/prepared/lt31-terminal-hold-20261027/handoff.json`，SHA256 **`a0be0a5aa14c34e99f6debfd3fb63fe81c8fc52b0c6c8c60c426e2cbbd47b71c`**，167代码/11输入证据/2配置。工作树未提交，文件SHA是版本依据；漂移应停止，不能改hash绕过。
- 两配置`configs/training/net-v1-terminal-hold-control.yaml`与`net-v1-terminal-hold-hold.yaml`。共同初始化`runs/training/lt30-surround-input-20261027/per-view/last.pt`，SHA `56eadf9928ceb255f290bd7bb584cc798e3466575b3baa3a0dbbff56eff0f35f`；source_epoch19。
- 数据共同为LT29的`runs/verification/surround-target-rebuild-lineage-20261027/train/train-balanced.npz`（2918条，SHA `137b13a6…`）和同根`val/val.npz`（300条，SHA `b8250ede…`）。不使用预检子集做正式训练，不回灌val。
- net-v1、hidden128、horizon320、dt0.2、image/per_view_v2、semantic height；batch8、seed20260827、每轮同shuffle、恢复重采样关闭，旧2400+518条监督每轮全覆盖。
- 两臂各4 epoch、365批/轮，共**1460次新Adam更新**；lr1e-5、clip1、patience4、early_stopping_start_epoch0确保不提前停止。teacher forcing固定0.2（start=end0.2/decay1）。累计停止权重0.2、不平衡正负停止；endpoint_alignment0.5/tail8、terminal_pose1/yaw10、collision0，其余沿原配置。
- 终点保持按每个有后缀样本等权，后缀监督为该样本专家末位姿，沿用raw航向误差模式；不同时改变航向损失定义。
- 主要比较固定第4轮`last.pt`（epoch从0计为3、原生阈值0.5）。best/deployment及自动val阈值校准只作辅助；不同loss配方的train/val loss不可直接横向当作效果优劣。
- 原E5报告仅用于205个原成功任务身份的静态回归分组，不能把对应输入匹配率当作旧闭环成功保持。

## 已完成短预检

68项定向测试通过，覆盖后缀梯度/均权、teacher副本、默认旧行为、旧checkpoint缺字段按0及跨策略恢复拒绝、实际末点/严格容差/无停止/缺失证据、完整逐样本导出与原训练/模型回归。

[真实预检](../../runs/verification/emitted-terminal-precheck-20261027/report.json)7.956秒：两臂同LT30权重，真实8训练（含恢复边界）+4验证，各1次新Adam更新，完整320点输出、保存/恢复/校准/原生分析成功，初始来源SHA相同。真实坐标梯度核验表明新项为后缀增加直接梯度、原前缀直接梯度精确不变；更新前teacher forcing前缀预测也精确一致。使用归档LT30各300条输出，新增公共指标复现独立审计的末点联合数28/20、停止前联合数232/227和均距0.973/1.695m。该核验不是新模型学习收益。

全部2918训练专家末点都满足各自原车位联合容差，见`source-label-audit.json`。修改前156份LT30源码已按旧SHA保存至预检目录`source-before/`，旧正式产物保持原样；旧LT30冻结任务不能直接用修改后源码重跑。

## 耗时、资源与进度

估计双臂训练加两个300条原生诊断约15–30分钟；依据LT30每臂20轮约32分钟，按本轮4轮折算并为新损失/补齐和诊断留余量。未实测完整epoch，机器负载和换页会增加耗时。CPU，无GPU；建议空闲内存8GB以上，保守预留10–12GB，磁盘预留1GB。两臂顺序执行，不后台并行。

每轮输出`epoch n/4`、loss、自由滚动ADE/FDE、teacher0.2；到4/4后仍有best恢复、val校准和绘图，等待最终完成/report路径。初期可能正在解压，不启动重复作业。

## 有先后依赖的PowerShell命令

在同一PowerShell窗口依次执行，任何异常立即停止后续阶段。

### 1. 身份、环境与输出保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt31Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt31Handoff = 'runs/prepared/lt31-terminal-hold-20261027/handoff.json'
if (-not (Test-Path -LiteralPath $lt31Python)) { throw '指定conda Python不存在' }
if ((Get-FileHash -LiteralPath $lt31Handoff -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'a0be0a5aa14c34e99f6debfd3fb63fe81c8fc52b0c6c8c60c426e2cbbd47b71c') { throw 'LT31 handoff漂移' }
$lt31Frozen = Get-Content -LiteralPath $lt31Handoff -Raw -Encoding UTF8 | ConvertFrom-Json
function Assert-LT31Frozen {
    foreach ($group in @($lt31Frozen.code, $lt31Frozen.inputs, $lt31Frozen.configs)) {
        foreach ($entry in $group.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "冻结文件漂移：$($entry.Name)" }
        }
    }
}
Assert-LT31Frozen
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
@'
import json, sys
from pathlib import Path
import torch, numpy, yaml, matplotlib
frozen = json.loads(Path('runs/prepared/lt31-terminal-hold-20261027/handoff.json').read_text(encoding='utf-8'))
actual = dict(python=sys.version.split()[0],torch=torch.__version__,numpy=numpy.__version__,yaml=yaml.__version__,matplotlib=matplotlib.__version__)
assert actual == frozen['versions'], actual
assert Path(sys.executable).resolve() == Path(frozen['python']).resolve()
assert torch.get_num_threads() == 1
print('LT31 environment passed', actual)
'@ | & $lt31Python -X utf8 -
if ($LASTEXITCODE -ne 0) { throw 'LT31环境检查失败' }
$lt31Root = $lt31Frozen.run_root
if (Test-Path -LiteralPath $lt31Root) { throw 'LT31正式根目录已存在；保留并反馈Agent，勿覆盖或自行续跑' }
New-Item -ItemType Directory -Path $lt31Root | Out-Null
Copy-Item -LiteralPath $lt31Handoff -Destination (Join-Path $lt31Root 'handoff.json')
```

### 2. 顺序训练control与hold

```powershell
$lt31Exits = [ordered]@{}
foreach ($lt31Name in @('control','hold')) {
    Assert-LT31Frozen
    $lt31Arm = $lt31Frozen.arms.PSObject.Properties[$lt31Name].Value
    & $lt31Python -u -X utf8 scripts/train_model.py --config $lt31Arm.config 2>&1 | Tee-Object -FilePath (Join-Path $lt31Root "$lt31Name-train.console.log")
    $lt31TrainExit = $LASTEXITCODE
    $lt31Exits["$lt31Name-train"] = $lt31TrainExit
    if ($lt31TrainExit -ne 0) { throw "LT31 $lt31Name 训练失败；停止后续阶段" }
    $lt31Report = Get-Content -LiteralPath (Join-Path $lt31Arm.output 'report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($lt31Report.status -ne 'completed' -or $lt31Report.history.train_loss.Count -ne 4 -or $lt31Report.data.train_samples -ne 2918 -or $lt31Report.data.val_samples -ne 300) { throw "LT31 $lt31Name 完成状态/轮数/分母不符" }
}
```

### 3. 两臂原生last的完整300条诊断

```powershell
foreach ($lt31Name in @('control','hold')) {
    Assert-LT31Frozen
    $lt31Arm = $lt31Frozen.arms.PSObject.Properties[$lt31Name].Value
    & $lt31Python -u -X utf8 scripts/analyze_predictions.py --data $lt31Arm.val --checkpoint (Join-Path $lt31Arm.output 'last.pt') --output (Join-Path $lt31Root "analysis-$lt31Name-native") --batch-size 8 --device cpu 2>&1 | Tee-Object -FilePath (Join-Path $lt31Root "$lt31Name-analysis.console.log")
    $lt31AnalysisExit = $LASTEXITCODE
    $lt31Exits["$lt31Name-analysis"] = $lt31AnalysisExit
    if ($lt31AnalysisExit -ne 0) { throw "LT31 $lt31Name 原生诊断失败；停止" }
}
```

### 4. 保存预声明阶段筛选与产物摘要

以下静态筛选即使false也正常保存，表示方法未过本阶段，不是脚本异常。它不会启动下一任务。

```powershell
Assert-LT31Frozen
@'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
reports = {name: read(root/f'analysis-{name}-native/report.json') for name in ['control','hold']}
rows = {name: read(root/f'analysis-{name}-native/sample_metrics.json') for name in ['control','hold']}
for name in reports:
    assert reports[name]['epoch'] == 3 and reports[name]['model_config']['stop_threshold'] == .5
    assert reports[name]['overall']['samples'] == reports[name]['overall']['emitted_endpoint_joint_evaluated'] == 300
    assert reports[name]['overall']['terminal_status'] == 'observed'
    assert [r['index'] for r in rows[name]] == list(range(300))
assert [r['task_id'] for r in rows['control']] == [r['task_id'] for r in rows['hold']]
baseline = read(Path('runs/validation/live-20261018-170806/full/E5/report.json'))
success = {e['meta']['dataset_index'] for e in baseline['episodes'] if e['success']}
assert len(success) == 205
for e in baseline['episodes']:
    assert rows['hold'][e['meta']['dataset_index']]['task_id'] == e['meta']['task_id']
old_inputs = {n: sum(r['emitted_endpoint_joint_passed'] for r in rows[n] if r['index'] in success) for n in rows}
a, b = reports['control']['overall'], reports['hold']['overall']
parent = read(Path('runs/verification/emitted-terminal-precheck-20261027/lt30-per-view-metrics.json'))['overall']
parent_old = read(Path('runs/verification/lt30-independent-review-20261027/report.json'))['arms']['per-view']['old_E5_success_input_geometry_only']['endpoint_joint']
checks = dict(
    endpoint_joint_increased=b['emitted_endpoint_joint_passed'] > a['emitted_endpoint_joint_passed'],
    endpoint_mean_distance_decreased=b['emitted_endpoint_position_error_mean_m'] < a['emitted_endpoint_position_error_mean_m'],
    prefix_ade_not_worse=b['ade_m'] <= a['ade_m'], prefix_fde_not_worse=b['fde_m'] <= a['fde_m'],
    support_not_worse=b['emitted_any_joint_passed'] >= a['emitted_any_joint_passed'],
    old_success_inputs_not_worse=old_inputs['hold'] >= old_inputs['control'],
    parent_endpoint_joint_not_worse=b['emitted_endpoint_joint_passed'] >= parent['emitted_endpoint_joint_passed'],
    parent_endpoint_distance_not_worse=b['emitted_endpoint_position_error_mean_m'] <= parent['emitted_endpoint_position_error_mean_m'],
    parent_prefix_ade_not_worse=b['ade_m'] <= parent['ade_m'], parent_prefix_fde_not_worse=b['fde_m'] <= parent['fde_m'],
    parent_support_not_worse=b['emitted_any_joint_passed'] >= parent['emitted_any_joint_passed'],
    parent_old_success_inputs_not_worse=old_inputs['hold'] >= parent_old)
result = dict(status='completed', checks=checks, stage_screen_passed=all(checks.values()),
    old_success_input_static_joint=old_inputs, parent_old_success_input_static_joint=parent_old,
    scope='static stage screening only; not closed-loop success or old capability preservation', production_qualified=False)
(root/'stage_screening.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
'@ | & $lt31Python -X utf8 - $lt31Root
if ($LASTEXITCODE -ne 0) { throw 'LT31阶段筛选数据核验异常；保留产物并反馈' }
$lt31Stage = Get-Content -LiteralPath (Join-Path $lt31Root 'stage_screening.json') -Raw -Encoding UTF8 | ConvertFrom-Json
[ordered]@{task='LT-20261027-31';status='completed';phase='training_and_static_screening_only';exits=$lt31Exits;stage_screen_passed=$lt31Stage.stage_screen_passed;production_qualified=$false} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $lt31Root 'completion.json') -Encoding UTF8
$lt31Artifacts = [ordered]@{}
Get-ChildItem -LiteralPath $lt31Root -File -Recurse | ForEach-Object {
    $relative = $_.FullName.Substring($lt31Root.Length + 1).Replace('\','/')
    $lt31Artifacts[$relative] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$lt31Artifacts | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $lt31Root 'artifacts.json') -Encoding UTF8
Write-Host 'LT31执行完成，等待Agent独立复核；不继续训练或运行闭环'
```

## 输出、异常、恢复与保护

正式根目录`runs/training/lt31-terminal-hold-20261027/`。两臂各有last/best/deployment.pt、report/history、阈值校准及曲线；两份analysis各有report.json、**全部300条sample_metrics.json**及分组/叠加图；根有四日志、handoff、stage_screening、completion和artifacts。训练根report使用校准best，主比较读取原生analysis。

正常完成是两臂4轮/1460新更新、2918/300分母、两个原生诊断300条、四入口退出0及上述文件齐全。stage_screen_passed是预声明静态筛选结论，任一项false不扩大同配方；true也只允许后续另立闭环计划，不能晋升。

身份漂移、非有限、内存不足、异常/非零退出或Ctrl+C均停止，保留目录、日志及已有last/best。不得删目录原地重跑、擅自resume_from、换batch/seed/剂量/损失权重或覆盖产物。先反馈Agent；仅分析失败无需重训，某臂已完成也不要重复执行它。恢复或重跑须另行诊断、冻结新配置和新目录，不自动续跑。

禁止覆盖任何源数据、LT28失败现场、LT29/30正式结果、LT30权重、source-before源码快照、短预检、handoff或已有正式/失败目录。旧LT30冻结代码已变化，应使用归档快照复现，不放宽旧SHA检查。

## 完成反馈与Agent验收

回复“LT31执行完毕”或异常及最后阶段。保留根handoff/completion/stage_screening/artifacts、四日志、两臂report/history及三个checkpoint、两份原生report/sample_metrics和图像，原地保存即可。

Agent将核对身份/退出/分母、同初始权重、Adam从0到1460、last epoch3/原生0.5、有限损失和参数、逐样本任务一致性及SHA，再独立复算阶段判定和场景/机动/噪声回归。新增阈值是本次训练假设的筛选标准，不改变主线固定300条成功≥210/碰撞≤30、原205成功逐例保持、完整车体安全及五模块证据要求。静态筛选中“原成功输入”只是这205条任务起点的预测指标，完整闭环保持仍未测。

## 执行结果

- 身份：handoff SHA `a0be0a5a…b71c`与计划一致；167代码/11输入/2配置零漂移（运行前与每臂/每分析前各断言一次）；环境单线程检查通过。
- 训练：两臂均`completed`、各**4轮/1460新更新**、2918训练+300验证；均经`initialize_from`从LT30 `per-view/last.pt`（SHA `56eadf99…`，source_epoch 19）只加载权重，Adam从0起步；`last.pt`为epoch 3、原生阈值0.5、权重有限。
- 原生300条（analysis-*-native）：

| 指标 | control（hold_weight=0） | hold（0.25） | LT30父模型 |
|---|---|---|---|
| ADE | **0.1736 m** | 0.3403 m | 0.2593 m |
| FDE | **0.1449 m** | 0.3150 m | 0.3187 m |
| 航向 MAE | 0.1817 rad | 0.2181 rad | 0.2082 rad |
| 实际末点联合通过 | **84（28.0%）** | 14（4.67%） | 20（6.67%） |
| 停止前联合支撑 | **253** | 15 | 227 |
| 末点位置误差均值 | **0.812 m** | 1.552 m | 1.695 m |
| 有后缀样本/点数 | 226 / 2251 | 14 / 115 | 269 / 5986 |
| stop_found_rate | 1.000 | 0.9967 | 1.000 |

- 阶段筛选（事先声明，12项）：仅`parent_endpoint_distance_not_worse`（1.552≤1.695）与`parent_prefix_fde_not_worse`（0.3150≤0.3187）为真，其余10项为假——相对同剂量control端点联合数未增加（84→14）、末点均距未下降（0.812→1.552）、ADE/FDE均变差、支撑数下降（253→15）、原205成功输入的静态联合数下降（**61→13**，父模型为16）。故`stage_screen_passed=false`。
- 独立复核：自`sample_metrics.json`逐样本重算12项检查与文件一致；control/hold索引均为0..299、task_id逐条相同且与基线205个成功任务身份一致；`artifacts.json` 39项SHA零缺失零漂移（仅清单自身未登记）；`completion.json`四入口exit=0、`production_qualified=false`。
- 判读：**终点保持假设未获支持**。hold臂的"有后缀样本"从226骤降到14，说明该目标主要让模型提前缩短发出轨迹、而不是把末点对齐到专家终点；端点联合与支撑同时崩塌、原成功输入静态联合从61降到13。control臂相对父模型的改善（ADE 0.2593→0.1736、端点联合20→84）来自额外4轮同数据微调，不是被测干预。按计划：不放宽阈值、不增加剂量、不搜索权重、不进入闭环；保留负结果，下一步需重新审视后缀监督与教师补齐的设计，而不是继续调`terminal_hold_weight`。
- 边界：静态筛选只反映205个原成功任务起点的预测指标，闭环碰撞/余量/到达与旧能力保持仍未测；源数据、LT28–LT30产物、LT30权重与冻结配置未改动；未闭环、未部署、未add/commit/push。

## 完成反馈后的独立复核

167代码/11输入/2配置、39项产物SHA一致，四入口日志完成，last均epoch3/Adam1460/原生0.5且权重有限；同LT30权重初始化、优化器重置。重新推理各300条并逐样本/五维分组重算，12项筛选与原报告一致，stage_screen_passed仍false。

补充定位：hold比专家长度提前285条，260条在原始全horizon首次联合达标点前结束发出；完整未发出320点有275条曾联合达标，最后一点均距1.736m（control 7.887m）。这细化了执行摘要的判读：末端保持坐标部分学到，发出轨迹的停止决策严重提前，不能仅由后缀样本数减少断言坐标保持没有学到。固定输出交叉hold坐标/control停止长度得到166个静态末点联合匹配，仅作归因，不是可晋升候选。控制器、碰撞与真实到达均未测。

决定：该hold配方停止扩大，不加剂量/搜权重；control的静态改善保留为待真实闭环检查候选，不自动晋升或启动下一任务。报告：[独立复核](../../runs/verification/lt31-independent-review-20261027/review.md)。
