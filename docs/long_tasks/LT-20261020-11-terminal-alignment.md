# LT-20261020-11：训练前部署停点与真值末点对齐审计

## 状态与目的

**已运行完成（2026-10-20 12:14，Agent 代跑并已核验）：2400 条全量审计完成，三类计数 1144/920/336、早停 692/晚停 228、真值索引达标 1760、无达标点 336（S4 173、S6 153）。** 独立审计入口、22项相关测试及六初态smoke已完成。保持R0、原阈值、损失、MPC与所有容差不变。

本轮已确认父模型存在终点位姿损失权重1、终点yaw权重10、尾部对齐0.5及累计停止监督，不能重复加同一损失当改进。原train2400标签末点全部满足0.3m/10°。但训练终点损失监督真值长度索引，部署却在停止头首次越过阈值的索引截断，必须分开量化。

六初态短检：#1298/#1559停止79/131点，真值90/150点；真值索引预测可联合达标，部署停点却不合格。另三条完整预测没有联合达标点，不能只校准阈值。本审计区分优先调查停止/轨迹哪条链路，**不据初态开环结果直接确定训练权重或宣布闭环改善**。

## 固定输入和唯一审计范围

仅原train `data/task_dataset/tracked_pivot_v8_3000/train.npz`2400任务，原顺序全覆盖。它是父模型训练域的原始任务池，不包含v40恢复池518条，不是独立测试集，也不是闭环访问状态。不能外推S8恢复状态或300条主成绩。

配置：`runs/diagnostics/terminal-alignment-train-20261020-v1/config.json`，SHA256：

`57092907eab527c28ace9e42c9e93119ccd3871251fc8640cd2a46d10ffcb641`

冻结5输入（train、manifest、R0、LT10 config/report）、99执行源及运行环境；旧97源逐项未改。模型SHA `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`，train SHA `8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f`。

固定max_horizon320、dt0.2、batch_size8、CPU单线程，无teacher forcing，自由滚动。阈值读取父模型约0.26且冻结，不扫描阈值，不读val/selection作推理，不改模型元数据或重新感知输入。使用归档BEV及现有局部goal/state转换；浮点批量推理不声称与每次单样本部署逐位相同。

## 输出与互斥分类

每个任务保存dataset_index/task_id/scene、真实长度、部署停止长度、停点误差、真值末点索引处的预测误差、标签末点误差、完整320点中首末联合达标索引与数量。达标同时要求位置≤0.3m及周期航向误差≤10°，不判断几何/控制安全。

三类分母合计必须等于2400，全体和按scene各自统计：

1. deployment_joint_valid：部署选中的点联合达标。
2. stop_misaligned_with_joint_point：部署点不合格，但完整自由滚动中存在联合达标点；另分early/late/inside_gap。停止后的点只作离线诊断，不是可在线使用的oracle。
3. no_joint_point：完整自由滚动中没有联合达标点。

类别1不等于MPC能安全到达；类别2不证明只改停止头就能闭环救回；类别3不证明训练标签错误。无停止交叉时按现有predict回退最后一个点，仍单独计stop_not_found。

## 已完成验证

22项聚合测试通过（terminal_alignment_audit、angular_training_loss、model_variants、prediction_analysis、stop_calibration），含新增6项：三类/early-late-gap、环绕角与阈值等号、真值索引与部署索引不同、有限值/shape/前缀mask、分组/索引、身份漂移与覆盖拒绝。实现前缺模块失败已观察；一次测试命令误列不存在test_training_data导致聚合失败，移除该错误模块后完整22项通过，不把失败命令记为通过。

六初态smoke2.248秒：分类1/2/3，两个错停均early，真值索引预测3条合格、标签末点6条合格；都是S4机制样本，不代表全train分布。NPZ/逐样本摘要及从保存预测重算统计一致，详见[预检复核](../../runs/diagnostics/terminal-alignment-train-20261020-v1/preflight/agent-audit.md)。不再重新prepare或smoke。

## 环境、资源与进度

唯一Python为`D:\conda\envs\endtoend-parking\python.exe`（3.12.13/NumPy2.5.2/torch2.13.0+cpu）。预计1～10分钟：六条2.248秒包含加载，正式300个batch不能按样本简单线性承诺。建议4GB可用内存、200MB磁盘，均为估计，不是实测峰值。运行中勿改代码或并行训练。

每25个batch打印进度，最终显示COMPLETED及分类统计；预测期间没有逐任务日志属于正常。无自动恢复/重试，异常保留输出反馈，不删目录再跑、不篡改hash。

## PowerShell：同一窗口按顺序执行

### 1. 只读身份预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$alignmentPython = 'D:\conda\envs\endtoend-parking\python.exe'
$alignmentRoot = 'D:\program\endtoend Parking\runs\diagnostics\terminal-alignment-train-20261020-v1'
$alignmentConfig = Join-Path $alignmentRoot 'config.json'
$alignmentHash = '57092907eab527c28ace9e42c9e93119ccd3871251fc8640cd2a46d10ffcb641'
if ((Get-FileHash -LiteralPath $alignmentConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $alignmentHash) { throw '配置漂移，停止反馈' }
& $alignmentPython -u -B scripts/audit_terminal_alignment.py --config $alignmentConfig --config-sha256 $alignmentHash --check-only
if ($LASTEXITCODE -ne 0) { throw '身份预检失败，不继续' }
```

应显示INPUT/CODE/RUNTIME VERIFIED; NO INFERENCE STARTED。

### 2. 正式原train2400条审计

```powershell
if (Test-Path -LiteralPath (Join-Path $alignmentRoot 'results')) { throw '已有正式输出，禁止覆盖/重跑，先反馈' }
$alignmentLog = Join-Path $alignmentRoot 'console.log'
if (Test-Path -LiteralPath $alignmentLog) { throw '已有日志，禁止覆盖，先反馈' }
& $alignmentPython -u -B scripts/audit_terminal_alignment.py --config $alignmentConfig --config-sha256 $alignmentHash 2>&1 | Tee-Object -FilePath $alignmentLog
if ($LASTEXITCODE -ne 0) { throw '失败/中断，保留所有产物，不自动重试' }
```

完成后results/report.json应status=completed、smoke=false、summary.samples=2400，三类计数合计2400，分scene分母总和2400。产物：identity.json、predictions.npz（自由滚动points/logits/target/mask/局部goal/原索引）、samples.json、report.json及SHA摘要。异常保存status.json，不换样本或补挑来源；反馈console末尾与status。旧LT10结果不覆盖；新增执行源使旧配置全集检查拒绝重跑是保护，不改旧hash绕过。

## 回传后的内容把控

核对全身份/样本覆盖/摘要，从NPZ重算类别与分组、检查无交叉/早停/晚停/无联合点的场景分布，再提出一个可检验训练改法。不按单个比例直接自动开训，也不只调阈值：若没有联合点，阈值选择无解；若有联合点，仍须几何和控制兼容性检查。

训练设计继续以原默认MPC为当前基线；若拟更换最终执行器必须单独验证，不静默用精修教师替代兼容证据。#1298/#1455原MPC下仍未通过的恢复标签不得被此次开环报告洗成通过。70%/10%主验收未变。

本任务已完成，命令不再重跑或prepare，不要求运行任何训练。历史回传口令为“训练域终点对齐审计完成”。

## 独立复核

见[agent-audit.md](../../runs/diagnostics/terminal-alignment-train-20261020-v1/results/agent-audit.md)：5输入/99源/环境、2400完整索引/原数据局部转换与NPZ/逐样本及独立公式重算通过。920错停中147条的合格点只在真值长度之外，336无联合点中179条位置/航向分别合格但时刻错开。初次MPC窗口比较920/920不因截断到首个联合点而变化，不能把错停数直接当可修复碰撞数。停止头冻结对照仅为后续待设计候选，不自动开训、不修改标签到模型oracle点。

## 运行结果（2026-10-20 12:14，Agent 代跑）

- 只读预检通过；正式运行 16.2 s、300 个 batch，`status=completed`、`smoke=false`、`samples=2400`；产物 identity/predictions.npz/samples.json/report 齐全，无 `status.json`。
- 三类互斥：`deployment_joint_valid` 1144（47.7%）、`stop_misaligned_with_joint_point` 920（38.3%）、`no_joint_point` 336（14.0%），合计 2400；停点方向 early 692 / late 228；`stop_not_found=0`。
- 间隔独立复核修正：early中位5点/p90 16/最大115，late中位5/p90 10/最大46；旧版混用0基索引与1基长度造成一点误差，原始预测/主要分类不变，不据短间隔宣称闭环可救回。
- 索引口径：真值末点索引处预测联合达标 1760/2400（73.3%），标签末点 2400/2400；剩余 640 条中 336 条整段无达标点、304 条“有达标点但不在真值索引处”。
- 分场景：S4 36/221/173、S6 30/157/153 退化最重（两场景占全部无达标点的 97%）；S1/S3/S5/S8 无“无达标点”样本，S8 85 条中 79 条为早停错位。
- 结论：停止链路是主要矛盾之一（38.3% 可进窗口但停错，其中 75% 早停），但 14% 整段无达标点、另有 304 条真值索引超差，说明轨迹/近端监督也有份。纯离线诊断，不训练、不调阈值、不改部署；主成绩 68.3%/25.7% 未变。明细见 `runs/diagnostics/terminal-alignment-train-20261020-v1/results/summary.md`。
