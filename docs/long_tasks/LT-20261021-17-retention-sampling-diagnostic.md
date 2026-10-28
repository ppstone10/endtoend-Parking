# LT-20261021-17 · 保旧覆盖与采样诊断完整化

状态：**已运行完成（2026-10-21 16:42–16:48，Agent 代跑并已核验）**。数据门禁与训练前基线门禁均通过，8 模型/192 更新 + 7 探针完成；四臂在两个 seed 上都没有同时满足预声明条件，裁决 `no_qualified_retention_signal_do_not_expand`。脚本、固定配置、50项定向测试及只读precheck已齐全；是诊断补全，不是已验证有效的长训配方。不得自动部署。

## 1. 目的与变量

完成预声明的两因素四臂：保持池覆盖（原/增强）×保持采样（原源均衡/分层时间计划），判断能否在保留LT16 S8收益的同时减少S3/S4/S6退化。共同固定恢复角色序列、模型/Adam起点、24次更新、lr/损失/阈值/MPC；增强池同时补任务和时间帧，不单独归因某一子因素。

| 臂 | 保持池 | 保持采样 |
|---|---|---|
| flat_old | LT16原池 | 原源均衡，从全局240继续 |
| balanced_old | LT16原池 | 场景→任务/机动/距离→源→有限时间覆盖 |
| flat_augmented | 原池＋成功补源/补时间帧 | 原源均衡，从全局240继续 |
| balanced_augmented | 同增强池 | 同分层时间计划 |

四臂各顺序seed20261020/20261021，同seed共享同一LT16联合240点及Adam240。每角色2帧/步、原角色损失、grad clip1、lr1e-5，各24更新至264，共8个模型/192更新；不是epoch训练，不接上次中断checkpoint继续加步数。顺序seed不是独立预训练初始化。

## 2. 为什么由用户运行

初次短检30.720秒，因S6四候选无成功正标签而停训练；一次有界扩源后第二次在180.023秒结束，只完整5组合，第6已训练但评测未完。Agent没有扩时续跑。已有部分结果均不胜未更新联合点，不能由此判断尚未跑的增强池/分层臂。

本任务独立新目录完整重跑同一训练剂量，可能重复部分计算；这是工程完整化，不是因负结果盲目加训练。原6探针缺第一seed的原能力退化对象，正式版在训练前作可测性门禁，并预声明额外基线候选；不是事后改晋升门槛。

原短检证据：[摘要与限制](../../runs/diagnostics/retention-sampling-preflight-20261021-v2/summary.md)、[原始报告](../../runs/diagnostics/retention-sampling-preflight-20261021-v2/report.json)。

## 3. 环境、身份与固定输入

工作目录`D:\program\endtoend Parking`；唯一Python为`D:\conda\envs\endtoend-parking\python.exe`，Python3.12.13/torch2.13.0+cpu/numpy2.5.2，CPU单线程。禁止C:\Python314，无新依赖/GPU要求。

唯一正式目录：`runs/diagnostics/retention-sampling-diagnostic-20261021-v2`。

已准备[config.json](../../runs/diagnostics/retention-sampling-diagnostic-20261021-v2/config.json)，SHA256：

`f4d08a8fed191d22dff443116e13dc70247fe099c128b27347a5173491864db0`

`identity.json`固定配置；114执行源（原LT16 112＋本轮2文件）、6输入/环境、前置完成单元及原短检config/report SHA均冻结。precheck逐项核验旧存档源，不改写旧identity；新增模块会使旧LT16全目录集合相等检查失配，不用旧入口重新跑LT16。原112文件未修改。

R0/train2400/manifest/val/selection/R2及SHA沿用LT16；R2只作输入身份，不拟合。val/selection仅查来源隔离，不训练、不作探针。两起点为LT16`broad_novel-{seed}/step0240/state.pt`，路径与全部输入SHA见config。

原E5实际BEV/target通道不变，网络H320/dt0.2/stop阈值0.26000000000000006，原MPC dt0.1/H10/replan10/max600、控制seed=数据索引、接触0/规划0.2；不加安全盾、在线教师或参数搜索。每回合完整，不缩短步数凑完成。

诊断v1目录是最终互斥修正前的准备配置，**不要运行**；两个短检目录均是历史证据，不覆盖/续跑。只使用本节v2目录。

## 4. 数据与训练前门禁

先重建LT16原390帧联合库，并按完整输入/标签去重。

- 旧S3#1196、S6#2149先逐元素复现LT16 R0状态/控制，成功才补全时间帧，不冒充新来源。
- 新S4-T2前进候选`[1452,1400,1438,1409]`。
- 新S6-T5前进候选`[2291,2269,2253,2258,2237,2263,2230,2274,2286,2308,2312,2256]`。原前4失败仍保留，一次扩到12，不再扩张/换源。
- 18采集来源全部运行，只有R0完整成功才克隆其软轨迹/stop/实际消费mask，失败轨迹不作正标签。两空单元各至少1新成功源，否则完整采集后输出`completed_data_gate_failed`，不训练。

短检实际取得#1400与#2256两个新成功源，113有效新增输入/标签、8旧重复拒绝、增强503帧；完整运行需核验是否复现，不能为凑数换来源。按task/source与LT16所有声明任务、val/selection隔离；旧2源的时间补全是明确例外，仅复用训练回放。

## 5. 基线与探针：先确认存在可修复对象

原6条不变：`[1199,1399,2184,2280,2377,2397]`，不作本轮梯度监督。实际短检R0仅1成功，第一seed联合点没有R0原成功丢失，不能直接用它验证双seed保旧修复。

额外16个基线候选在配置中已固定，排除原来源/探针/补源/保留集：

- S3-T4倒车：`[1194,1184,1171,1187]`；
- S4-T2前进：`[1415,1428,1398,1434]`；
- S6-T3倒车：`[2188,2159,2175,2201]`；
- S6-T5前进：`[2257,2285,2307,2290]`。

训练前对原6及额外16都跑R0、两未更新联合点；先原6再额外16的固定顺序，给每seed取首个“R0成功/联合失败”的S3/S4/S6证据。两证据并集去重，已有原6中的不再添加，最终共同评估6～8条。选择只依未更新基线，不看新模型，不替换候选；这是一组基线失败定向挑战，不是无偏总体收益估计。

若任一seed没有退化证据，或没有S8原成功可保留，输出`completed_baseline_gate_failed`并停训练。不能因为数据门禁过就对没有修复对象的集合盲训。

## 6. 运行命令

配置已prepare，**不用再次prepare，不用再跑短检**。先只读预检：

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/preflight_retention_sampling.py --output 'runs/diagnostics/retention-sampling-diagnostic-20261021-v2' --complete-diagnostic --precheck
if ($LASTEXITCODE -ne 0) { throw '诊断身份预检失败，停止并反馈' }
```

看到`PREPARED IDENTITY VERIFIED; NO ROLLOUT STARTED`后，执行一次完整诊断：

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/preflight_retention_sampling.py --output 'runs/diagnostics/retention-sampling-diagnostic-20261021-v2' --complete-diagnostic --run-prepared 2>&1 | Tee-Object -FilePath 'runs/diagnostics/retention-sampling-diagnostic-20261021-v2/console.log' -Append
if ($LASTEXITCODE -ne 0) { throw '诊断未完整结束，反馈report.json及日志尾部，不重置状态重跑' }
```

`completed_data_gate_failed`或`completed_baseline_gate_failed`会正常退出0，是实验门禁结果，不强行启动训练。不要同时跑第二份进程、不要逐臂改参数或只挑最好seed。

## 7. 时间、资源与产物

预计6–10分钟，基于180秒只完成5组合、另加48个基线候选回合及未完成组合估算，实际取决于600步振荡任务/CPU负载，不保证。活动预算**900秒（15分钟）**，不是追加训练步数。预检耗时为只读校验，不计作正式模型训练；无自动继续或预算放宽。

建议内存2–6GB、磁盘预留1–2GB，峰值未实测；8个含Adam检查点约数十MB，另有数据缓存/完整轨迹。stdout为COLLECT、BASELINE（分模型/原探针/候选阶段）、FIT、FINISHED；阶段可能数十秒无新输出。

正式目录输出：

- `config.json/identity.json`冻结身份；`report.json`保存阶段、来源、门禁、基线候选、选中的退化证据、四臂结果及最终判断。
- `data.pt`为原池与新增训练监督；`collect-*/R0.npz`为真实完整采集。
- `baseline/*.npz`为R0及两联合点的原6/候选/拟合基线；模型均未更新。
- `<arm>-<seed>/state.pt`为264步模型/Adam，`report.json`含实际采样身份/时间帧、loss、共同探针及2拟合监测；NPZ为状态/控制。
- `artifacts_sha256`只在完整结束后生成，不含不断追加的console/root报告/锁；未生成不能称产物已完整验证。torch只加载本项目已哈希的可信本地产物。

## 8. 裁决与停止恢复

每seed候选相对未更新联合点须：探针成功净增≥1、碰撞不增、不丢其任何原成功，至少救回1条R0原成功/联合失败的关键场景任务，且S8至少有1条原成功并全部保留。两seed同时满足；再对同24步`flat_old`，逐seed成功不低/碰撞不高/无其成功丢失，两seed合计至少多1成功。

符合才报告诊断信号，可另准备训练计划；只过拟合或一个seed、控制臂也同样改善，都不够。`allowed_to_prepare_training_plan`控制后续已有效方案训练，不代表当前诊断完整化已证明有效。主≥70%/≤10%始终另验，`main_protocol_passed=false`，不自动部署/继续主300。

本版不支持断点续跑。Ctrl+C后等进程结束，保留报告与所有partial checkpoint；不要从中途Adam接着训。已有执行状态的目录拒绝再次run-prepared，互斥失败不改写原状态。若中断/预算耗尽/磁盘或身份错误，先反馈；Agent审核后才能给新目录的新准备/预算，不能把report改回prepared、删除目录或改identity绕过。崩溃留下run.lock时保留并反馈PID，不擅自删锁或同时启动新进程。

禁止覆盖LT15/LT16、R0/R2、train/val/selection、两个短检与其部分模型；不清失败记录换好结果，不执行git commit/push。

## 9. 完成反馈与Agent核验

反馈“LT17保旧采样诊断完成”及主目录。保留report/config/identity、console末60行、全部checkpoint/NPZ/data；门禁失败或超时也反馈，不必搬大文件。

Agent随后须核验来源正标签与原两R0源复现、数据去重/隔离、基线证据选择确在训练前、两seed同Adam起点/264步/恢复采样完全一致、6～8探针共同分母、完整8组合及S8保留。前5已完成组合在原6探针及模型张量上的可复现性也应核查，不能把漂移当新收益。全部数据仍属R0训练域中的本次适配隔离，挑战证据按失败定向选择，不能替代新的正式独立确认。结论无论正负都统一分析，不以一臂结果结束或自动串行长训。

## 运行结果（2026-10-21 16:42–16:48，Agent 代跑）

- precheck 通过；一次 `--run-prepared` 完成，316.9 s（约 5.3 分钟，900 秒预算内），日志末行 `FINISHED no_qualified_retention_signal_do_not_expand`；`stage=matched_fit_and_probe`、`branch_failures=[]`。
- 门禁：采集 18 源全跑，有效新增 113（追加 121、重复 8），增强池 503 帧；两空单元各得新成功源（#1400 S4、#2256 S6）→ `data_gate_passed=true`。训练前基线：每 seed 各取首个“R0 成功/联合失败”证据（seed1 #1428、seed2 #2184），S8 原成功 #2377/#2397 两 seed 都在；`baseline_gate.passed=true`，最终 7 条共同探针。
- 四臂（相对未更新联合点，7 条共同探针）：
  - flat_old：seed1 −2/+1（丢 1199、2184、2377）；seed2 −2/+2（丢 1399、1428）。
  - balanced_old：seed1 −2/+1（丢 1199、2184）；seed2 −1/+2（丢 2377、1428）。
  - flat_augmented：seed1 −1/0（丢 1199、2184）；seed2 **0/0 且零丢失**。
  - balanced_augmented：seed1 0/0（丢 1199）；seed2 0/+1（丢 1428）。
  相对控制臂（flat_old）两增强臂为 +1～+2 成功 / −1～−2 碰撞。
- 判定：四臂全部 `passed=false`——增强池把旧成功丢失从 2～3 条压到 0～1 条，但没有任何一臂在**两个 seed**上同时取得“成功净增≥1、碰撞不增、零丢失、救回关键场景、S8 全保留”，故 `qualified=[]`、`allowed_to_prepare_training_plan=false`、`main_protocol_passed=false`。
- 结论：保旧覆盖有效、恢复收益未兑现；不扩大训练、不自动串行主 300。明细见 `runs/diagnostics/retention-sampling-diagnostic-20261021-v2/summary.md`。

## 反馈后的独立核验（2026-10-21）

[独立核验](../../runs/diagnostics/retention-sampling-diagnostic-20261021-v2/independent-review.md)已完成：冻结源/179产物SHA、原390帧/113新增、来源隔离、基线选择、8份实际采样与Adam264、原短检6份模型/Adam及5个完整组合原轨迹均核验；160份NPZ/54092转移运动学、终点误差及末段接触符合记录，8回合checkpoint完整实跑逐元素复现。未修改执行代码、未启动新训练或主300。

当前四臂均不合格的判断不变，但上述运行记录“增强池旧成功丢失压到0～1”不适用于增强原采样第一seed（仍丢2）；增强分层虽两seed各仅丢1，也没有净成功增加。新增发现是两个增强分层模型实际仅消费2/3个新增输入，旧S3/S6补的67个时间内点均未采到；不能据此认定采样已饱和。下一优先是在不增加24更新预算的前提下，先保障关键新增帧真正进入梯度，再按原双seed净闭环收益门槛短检；尚未实施或证明有效，不生成已有效训练计划。
