# LT-20261022-18：S6成功策略教师迁移与旧能力保持

状态：**已运行完成（2026-10-22 15:26–15:28，Agent 代跑并已核验）**。四模型各 28 更新、44 回合完成；两个 policy 学生都在 S6 #2149 严格恢复，但同输入配对净成功 0/−2、碰撞 +1/+2，且两 seed 都丢掉两条 S8 原成功，`qualification.passed=false`，裁决 `no_qualified_native_transfer_do_not_expand`。关联任务T-145305、[Spec](../../spec/s6_policy_transfer.md)。Agent已完成代码、定向测试与真实零更新预检。

## 目的与唯一主要变量

检验成功恢复策略的离线points/soft-stop能否迁移为C240学生的原生闭环收益。匹配两臂仅改变53个新增输入的目标：self_targets采用各自未校准C240自目标；policy_targets采用已验证的非候选成功网络教师目标。共同输入/mask、旧监督、顺序、优化器、学习率及更新数一致。共同mask来自成功策略实际消费，不保证处于C240原首停支持内；自目标臂是反事实控制，不是额外成功教师。

## 固定输入、代码与参数

- 工作目录`D:\program\endtoend Parking`；仅`D:\conda\envs\endtoend-parking\python.exe`（Python3.12.13、torch2.13.0+cpu、numpy2.5.2）。CPU单线程，无GPU。
- 冻结配置`runs/prepared/s6-policy-transfer-preflight-20261022-v3.json`，SHA256=`cd1d25e0cb8bbbcf191a7ef0f7b95eedb22a62ee420936547a9e8f981627e2c1`；其中登记139代码、606输入/产物文件及完整两顺序/两臂批次身份。工作树未提交，文件SHA是本实验版本依据；修改任一冻结文件会拒绝运行，不能手工改hash或重建配置绕过。
- 未校准起点：`effective-replay-20261021-v2/units/fit-broad_novel-20261020/attempt-066c99e4c7b2/step0240/state.pt`与`fit-broad_novel-20261021/attempt-a012fc45c5eb/step0240/state.pt`（均在`runs/diagnostics/`）；精确路径和SHA在配置中。
- 成功银行`runs/diagnostics/s6-chain-learning-preflight-20261022-v2/success-teacher-bank/bank.pt`，SHA256=`f4fc58d3beca9d2cddb616290a8b092c3b5adfcc62ade2d5c870376b6ca13095`；教师checkpoint SHA=`5d8e127c6cec63cefe00d143b052b4f37e28a2ef24300c5b4a38ca8ad3996cd1`。仅训练源#2149的53个clean非零新帧；训练/val/selection文件和旧390+113库均冻结，7挑战不回灌。
- 每学生Adam240→268，28更新、lr1e-5、clip1。每批2旧anchor+2旧recovery+2新增recovery，旧24批循环补4批；两顺序各56新槽，53输入全覆盖，112旧槽保持。behavior_loss不变，旧保持权重1/2、旧恢复/新恢复各1/4。序列20按时间、21固定置换；不改学生bias、stop_threshold、输出截断或MPC。
- 四模型，每模型固定7已见挑战+4训练源，共44个600步上限完整原MPC回合。无超参搜索/额外教师/数据聚合/自动重试。总计算预算300秒（协作式检查，单次底层操作/落盘可能使墙钟略超）。

## 已完成的短时预检

`tests.test_s6_policy_transfer`及相关三模块共31测试通过，覆盖标签配对、全53曝光、旧槽不变、真实小模型Adam268、漂移/超时失败关闭、禁止覆盖、JSON键及四模型CLI编排；编排测试使用mock，不代表实际四模型完成。真实输入核验＋四种首批前向/反向28.575秒完成，loss/梯度有限，**零optimizer.step、零闭环回合**，两起点Adam仍240。证据：[预检报告](../../runs/diagnostics/s6-policy-transfer-precheck-20261022-v1/report.json)。最终v3身份预检另已通过，补齐相对旧三个查询的去重；标签/梯度计算没有改变。早期冻结配置v1/v2已过期，不使用。

## 有先后依赖的PowerShell命令

在同一PowerShell窗口依次执行。先检查环境、冻结配置和输出占用；任何报错均停止。

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$transferPython = 'D:\conda\envs\endtoend-parking\python.exe'
$transferConfig = 'runs/prepared/s6-policy-transfer-preflight-20261022-v3.json'
$transferOutput = 'runs/diagnostics/s6-policy-transfer-preflight-20261022-v1'
$transferConsole = 'runs/diagnostics/s6-policy-transfer-preflight-20261022-v1.console.log'
if (-not (Test-Path -LiteralPath $transferPython)) { throw '指定conda Python不存在' }
if ((Get-FileHash -LiteralPath $transferConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'cd1d25e0cb8bbbcf191a7ef0f7b95eedb22a62ee420936547a9e8f981627e2c1') { throw '冻结配置漂移' }
if ((Test-Path -LiteralPath $transferOutput) -or (Test-Path -LiteralPath $transferConsole)) { throw '已有产物，停止；保留旧目录并联系Agent复核' }
& $transferPython scripts/preflight_s6_policy_transfer.py --config $transferConfig --output $transferOutput 2>&1 | Tee-Object -FilePath $transferConsole
$transferExit = $LASTEXITCODE
if ($transferExit -ne 0) { throw "本实验退出码 $transferExit；不要自动重试" }
Get-Content -LiteralPath "$transferOutput/report.json" -Tail 30
```

## 耗时、资源与正常进度

估计约3–5分钟，CPU单线程，内存暂估4–8GB，新增磁盘预留1GB；依据前轮四个24步模型+44回合114.725秒、本次梯度样本量约1.75倍及真实首批预检，未实测完整任务，缓存/机器负载/回合长度会改变耗时。固定300秒后失败关闭，不延长预算追完；耗时不确定，因此由用户执行。

控制台依次出现四条`FIT ... updates=28`，每个模型先训练再评测；中途无逐回合输出时可查看`report.json`的stage及已保存子报告，不另启动作业。`status=completed`、`stage=decision`且四臂/顺序齐全，表示执行完整；`qualification.passed`才是方法是否达本阶段门槛。主验收和自动部署始终false，不能只看FINISHED就认定改进。

## 输出、停止、恢复与保护

输出目录如上；根目录config/report，四个`<arm>-<seed>/state.pt`与report、44条回合记录；控制台日志放在同级`.console.log`，防止提前创建输出目录。根报告记录实际更新数；每checkpoint应为268，28条loss和28实际批次，教师槽56/53 unique。

异常、非有限、身份漂移、超时或Ctrl+C均停止本次，不自动续跑；保留整个目录、控制台及已完成模型。强杀可能留run.lock，不删锁后原地续跑。先反馈Agent诊断；若经复核需要重跑，只在新v2输出/日志完整从C240重跑，不复用部分Adam或追加剂量。不得覆盖任何`effective-replay-*`、`retention-*`、`s6-chain-*`、成功教师银行、预检报告、冻结配置或本次失败目录，不重新生成银行/校准bias。

## 完成后反馈与Agent验收

回复完成/异常和上述输出路径；保留根report/config、四checkpoint与子报告、全部回合NPZ和同级console.log。不需要粘贴大数组。

Agent先核验配置SHA/139源/606文件、身份与退出状态、日志尾部、四模型Adam268/真实批次、44回合分母与产物SHA，再分析结果。联合门槛沿用S6-RECOVERY-003：相对历史C240/flat_old/消费保障保持门槛全部通过，同输入self控制逐顺序不丢成功、成功不低/碰撞不高且合计净成功至少+1；两个policy学生#2149均从起点全程严格到达，四训练源C240旧成功全部保住。历史24步控制是资格参考，不能用于解释28步标签干预的因果效应。按需从checkpoint独立重放决定性回合，不只凭汇总或loss宣布达标。

任一组合未完只报不完整；完整但门槛失败不扩大同配方。通过也只允许另行准备后续计划，7已见挑战不是独立确认，原主成绩68.3%/25.7%和≥70%/≤10%主联合验收仍不变。

## 运行结果（2026-10-22 15:26–15:28，Agent 代跑）

- 命令块一次性通过环境/冻结配置/输出占用检查；运行 109.3 s（300 s 预算内），日志末行 `FINISHED no_qualified_native_transfer_do_not_expand`；`status=completed`、`stage=decision`，四模型 `actual_updates` 均为 28，四目录 `state.pt`/`report.json`/11 份回合 NPZ 与根 `artifacts_sha256` 齐全。
- **同输入配对（policy 相对 self，7 条已见挑战）**：seed20261020 为 **0/+1**、丢 #2377/#2397（S8 −2，S3/S6 各 +1）；seed20261021 为 **−2/+2**、丢 #1399/#2377/#2397（S8 −2、S4 −1、S3 +1）。两 seed 都丢掉两条 S8 原成功，`incremental_passed=false`。
- **拟合源门禁**：两 seed 的 S6 #2149 都严格恢复 ✅；但 seed20261020 丢 #1400（原拟合成功未全保留），故 `fit_passed=false`。
- `original_gate`（沿用 LT17 基线门槛）仍 false；总 `qualification.passed=false`、`allowed_to_prepare_training_plan=false`、`main_protocol_passed=false`、无自动部署。
- 结论：教师目标可学（#2149 两 seed 严格到达），但没有转为净收益，代价是 S8 两条原成功回退；属“完整但门槛失败”，不追加剂量、不换教师、不扩大。明细见 `runs/diagnostics/s6-policy-transfer-preflight-20261022-v1/summary.md`。
