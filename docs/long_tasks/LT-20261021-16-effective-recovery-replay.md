# LT-20261021-16 · 有效恢复监督 × 广覆盖行为回放

状态：**已运行完成（2026-10-21 12:52–13:40，Agent 代跑并已核验）**。数据门禁通过（62 条有效新恢复标签 + 240 帧新回放、旧 27 帧反例仍 0 新增），158 单元/24 检查点齐全；无方向通过发现门槛，诊断确认里 `broad_novel-240` 在两套各 4 组合上出现联合收益但丢 3～4 条旧成功（要求 ≤2），裁决 `no_qualified_direction`。独立入口、冻结配置、39项定向测试及70.905秒最终冒烟均已通过。它已有诊断性闭环收益，但不符合晋升标准，不是主目标已达标的训练配方。

用户反馈完成后的[独立复核与下一方向](../../runs/diagnostics/effective-replay-20261021-v2/independent-review.md)：1728评测回合/402210转移核验、8回合完整checkpoint重放均一致；62教师后缀10435转移全段接触/规划几何通过。识别监督与实际采样覆盖缺口，下一步先验证保旧，不重跑本计划。原摘要/下面运行记录中的88/27帧及60/12步混淆，以独立复核第7节勘误为准；原运行证据不回写。

## 1. 为什么做这轮，什么才算进展

LT15的“新增”每seed27帧实际全重复，保持池又仅3个近距干净T1。新版本修正数据有效性后，一次运行完整比较两因素，而不是试完一个方向就停：

| 臂 | 保持池 | 恢复池 |
|---|---|---|
| old_old | LT15原保持56帧/3源 | LT15原恢复32帧/2源 |
| broad_old | 原保持＋分层R0成功回放 | 原恢复 |
| old_novel | 原保持 | 原恢复＋有效新增查询 |
| broad_novel | 原保持＋分层R0成功回放 | 原恢复＋有效新增查询 |

对照分别回答扩大保持覆盖能否减少旧成功损失、新恢复覆盖能否减少闭环碰撞，以及二者能否同时成立。新恢复因素同时扩任务、状态及输入质量，不能把其收益只归因DAgger某一子变量。

先完整采集所有预固定来源，汇总拒绝原因；数据不过线不得启动四臂训练。四臂均无发现收益仍按预固定排序复核前二方向，但不能倒推发现过线。只有完整纯网络闭环成功增加、碰撞减少且旧能力回归受控，才是阶段收益；教师成功、loss下降或两步smoke均不是。

主目标仍为原E5无干预成功率≥70%、碰撞率≤10%；当前主300仍205成功/77碰撞/18振荡，未部署任何新模型。

## 2. 前置条件、冻结身份与禁止项

- 工作目录`D:\program\endtoend Parking`；指定`D:\conda\envs\endtoend-parking\python.exe`，Python3.12.13/torch2.13.0+cpu/numpy2.5.2；CPU单线程，不用C:\Python314。
- LT15完整产物必须保留在原路径；本入口逐项核验旧109份源、6输入、10份旧R0数组及5个必需完成单元（collect、两dense_joint、两aggregate-labels）。旧配置/报告/identity SHA硬固定，不回写任何LT15源或产物。
- 新配置：[config.json](../../runs/diagnostics/effective-replay-20261021-v2/config.json)，SHA256为`368718182c5f8fbfef43ef68ea278ae87583d069245dd78606b3ed56e65cb088`；112份当前执行源及输入/环境/分组均冻结。
- R0、原train/val/selection/manifest与LT15一致。val和selection只核验来源隔离，不训练、不选臂、不确认；R2只作固定输入身份，不参与本轮拟合。
- 不改架构、阈值0.26000000000000006、原MPC dt0.1/H10/replan10/max600、原BEV通道、接触余量0/规划余量0.2。网络H320/dt0.2；候选评测只有网络→原MPC，没有在线专家/安全盾。
- 运行时不要编辑执行Python、输入、已完成单元或配置，不要重写identity。新增实验文件会使旧LT15全目录集合检查出现版本差异；不要用旧入口重跑或修改旧identity。本版本用旧存档清单逐项核验，源修改仍拒绝。
- 预备的`effective-replay-20261021-v1`仅为元数据修正前的配置，**不运行**；v2为唯一正式目录。旧smoke-v1/v2保留但最终证据只用smoke-v3，均不得充当正式结果。

## 3. 预固定数据与新增门禁

分组先按元数据哈希、task/source身份确定，不读结果选源：

1. 新60确认：S4/S8各15，S1/S2/S3/S5/S6各6；排除旧FIT5、PROBE5、LT15确认60及val/selection身份。
2. 新恢复12源：S4 `[1620,1542,1611,1486,1576,1336]`；S8 `[2375,2370,2398,2338,2378,2351]`；与上述组隔离，失败不换源。
3. 回放94候选：按scene/task_type/maneuver/distance_tier实际47格各2源，排除以上所有组。只采R0完整真实成功回合的均匀最多4帧，失败不补选；完整清单见config。

三查询策略固定R0、LT15两顺序seed的dense_joint240。每策略在每恢复源实际闭环后，从非零、规划安全、每10步的推理上下文按时间均匀取最多3个，最多36策略-来源单元、108查询起点。允许早期失余量之后重新进入安全集合的状态，但查询起点必须安全、整个教师后缀严格成功，不能用此前不安全前缀冒充全回合安全。

S4持续教师只试一次；S8归档全局参考加既有至多2次有界离线yaw补偿，参数不搜索。只在实际到达/零接触/零规划侵入时采教师首帧标签。位姿、BEV、state必须复现该步网络输入；goal几何一致后监督输入取网络原值，重算R0 stop缓存。教师后续帧只作执行证据，不冒充学习器查询。

按完整float32输入与points/stop/mask计算SHA。追加数、唯一数、相对旧库新增输入/标签对、真正采纳数分开统计；同输入重复或异标签拒绝。BEV/state全同、goal仅2e-5内舍入差异也保守拒绝，避免旧教师局部goal算式差异制造假新增。每轮重新复现LT15两组27帧有效新增0的反例。

正式训练联合门禁：

- 广回放至少24个**新成功任务**，覆盖S1–S6、至少4种task_type、forward/reverse、near/medium/long。
- 新恢复S4/S8各至少2个不同来源任务，且均是非零实际网络查询；旧库/轮内去重后计算。
- 无未关闭的采集单元ValueError。严格教师不通过是正常拒绝，不等同代码错误。

任何一项失败：所有预声明来源仍尝试完，输出`completed_data_gate_failed`及覆盖/教师原因，**不做训练和确认**。这不是用户运行失败，也不能扩大步数绕过。未覆盖S7（原train没有）；S9不触碰。所有确认都属于R0原训练域中的本次隔离复核，不称模型未见泛化。

## 4. 数据通过后的完整探索

四臂×顺序seed20261020/20261021，从同R0重新开始；固定lr1e-5、原角色损失、每角色2帧/步、角色内按源轮换、grad clip1。不是换初始化seed，不能解释为两独立预训练模型。

各240次更新，共1920次，全部保存12/60/240共24份含Adam的检查点。每点跑原FIT5＋新恢复12＝17条拟合任务，以及固定PROBE5；共同缓存指标用同一broad_novel池，不比较不同轮换批次的loss。

60和240点在运行前已声明可参加方向排序，12仅诊断。每方向必须同臂/同时点的两顺序seed，不允许各seed独挑最好点。最多8方向中取2个：两seed发现均满足净成功≥1、净碰撞≤−1、无旧成功丢失者优先，再按既有净收益/损失排序，最后名字固定破同分。无方向发现过线也继续前二诊断确认，不据确认重训或换臂。

确认分两套：LT15旧60仅回归，新60作本次隔离确认；每套R0和两方向×两顺序seed×两控制seed完整配对。控制RNG种子为任务索引＋0/10000。最多1200个确认回合，不用smoke缩短回合代替。两套的四组合都须各满足成功净增≥3、碰撞净减≥3、丢旧成功≤2、任一场景成功净减≤1，且发现先过线，才标`stage_candidate_for_review`。本轮`main_protocol_passed`始终false；不自动部署或接着跑主300。

## 5. 运行命令（只运行v2）

正式目录已经prepare，**不用重新prepare，也不用再跑smoke**。先只读预检：

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/explore_effective_replay.py precheck --output 'runs/diagnostics/effective-replay-20261021-v2'
if ($LASTEXITCODE -ne 0) { throw '身份预检失败，停止并反馈，不启动run' }
```

看到`IDENTITY AND ARTIFACTS VERIFIED`后，一条run完成所有声明分支（训练有数据准入）：

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/explore_effective_replay.py run --output 'runs/diagnostics/effective-replay-20261021-v2' 2>&1 | Tee-Object -FilePath 'runs/diagnostics/effective-replay-20261021-v2/console.log' -Append
if ($LASTEXITCODE -ne 0) { throw '任务异常结束，请反馈report.json与日志尾部，不改配置重试' }
```

不要同时运行第二个进程，不逐臂人工选优或中途改超参数。数据门禁失败会正常退出0并打印`FINISHED no_training_review_coverage_and_teacher_failures`，请反馈而非强行启动训练。

## 6. 时间、资源与输出

预计完整45–85分钟；数据门禁未通过时会更早结束。估算基于LT15约25分钟、此次更多来源与两套确认，不是时间保证。总活动预算**90分钟**，恢复累计计入；正式最多94回放＋36查询单元＋8拟合单元＋20确认单元＝158单元。教师最多108起点，按S4一次/S8三次上限共216后缀；无参数扫描或无限补源。预算耗尽就是未完成，不清计时绕过。

建议内存预留2–6GB、磁盘2GB；峰值未实测。旧单检查点含Adam实测约2.81MiB，最终24份之外还有数据缓存/轨迹/失败attempt。无GPU要求。

目录`runs/diagnostics/effective-replay-20261021-v2/`：

- `config.json/identity.json`：不可回写的身份；`report.json`：有效新增、回放覆盖、来源尝试、四臂因素对照、排序和两套确认。
- `progress.json`：累计活动时间，约30秒更新；`console.log`：用户命令日志。正常stdout为BEGIN/END/CHECKPOINT，最终FINISHED；训练阶段可能数十秒无新输出。
- `units/<单元>/completed.json`：SHA完成清单；`attempt-*/payload.pt`：可信本地产物；NPZ为实际状态/控制/教师参考。只加载本项目已哈希的可信torch档案。
- `fit-*/attempt-*/step*/state.pt/report.json`：模型、Adam、真实闭环与共同缓存指标，诊断专用格式，不替换部署。

一般无收益/碰撞继续其余预声明臂；采集/拟合单元ValueError记账，身份漂移、超时、其他运行错误退出。`completed_with_branch_failures`不是完整比较，也不晋升；数据阶段单元错误会禁止训练。单个教师执行ValueError记录查询错误后继续其他预声明上下文，不采该标签。

## 7. 中断与恢复

Ctrl+C后等待进程退出；恢复先同一precheck，再同一run。完成单元SHA核验后跳过；未完成单元保留旧attempt并从该单元起点重算，**不从中途梯度步恢复**。确认按每套/模型/顺序seed/控制seed单元保存，不必重跑全部确认。

崩溃留下锁时，先读取锁PID并在任务管理器确认相应任务进程确实退出，不能判断就反馈。确认后仅删指定锁，再precheck/run：

```powershell
Get-Content -LiteralPath 'D:\program\endtoend Parking\runs\diagnostics\effective-replay-20261021-v2\run.lock'
# 人工核对该PID的本任务进程确已退出后，才执行下一行。
Remove-Item -LiteralPath 'D:\program\endtoend Parking\runs\diagnostics\effective-replay-20261021-v2\run.lock'
```

强制崩溃最多可能丢最后约30秒计时；90分钟耗尽、代码/输入漂移、缺文件、磁盘不足都反馈，不删progress、不改identity、不换目录扫好结果。不得改写R0/R2、train/val/selection、LT14/LT15、任何旧预检/已完成单元；失败attempt保留。

## 8. 已完成预检与结果反馈

最终[smoke-v3报告](../../runs/diagnostics/effective-replay-smoke-20261021-v3/report.json)70.905秒，22单元全部完成无分支失败；8份两步checkpoint的Adam步数/阈值与全部完成清单SHA已核验。报告SHA=`af662e90d7fa87c390b7f184b2921aa431f0e0523435c56612edd230de8544fb`。

实际新增：C240第一顺序seed的S4#1559 step320，经持续教师47步严格到达，仅首帧入库；S8#2330按均匀规则取10/110/210，只有step110在最多2次既有补偿后163步严格到达，仅首帧入库。两输入/标签均与旧库不同；20个教师续行帧不计作新增。被拒的10/210没有因接触为0就降低余量门槛。旧两组27帧仍各新增0。回放短检S1#0、S5#1720均实际成功。

本次缩小池只有2个新回放任务、S4/S8各1个新查询来源，**formal_data_gate_passed=false是预期**。smoke显式允许缩小集走两步集成，但`qualified_directions=[]/main_protocol_passed=false`；不得用它启动无门禁正式训练。最终正式236来源初始E5 BEV全与归档精确一致，v2只读precheck通过，尚无正式units。

39项定向单测通过：`python -m unittest tests.test_effective_recovery tests.test_behavior_exploration tests.test_executable_behavior tests.test_reference_compensation tests.test_behavior_preservation`，实际使用指定conda。覆盖27重复反例、ULP别名、异标签/批内重复、实际首帧、非零均匀上下文、task/source隔离、覆盖门禁、因素池、配置步数及既有锁/恢复/优化器/角色损失。

完成后回复“LT16有效恢复回放探索完成”，保留/提供主目录、report/config/identity/progress、console末60行及全部units，无须手动搬大文件。即使数据门禁失败也反馈。Agent下一步核验真实新增和源分布、四臂同预算、实际轨迹/检查点与两套配对，给出支持/反对哪个假设的证据；不能把负结果泛化成架构无解，也不只凭拟合数字继续加训。

## 运行结果（2026-10-21 12:52–13:40，Agent 代跑）

- precheck 通过；run 一次完成 2900.6 s（约 48.3 分钟，90 分钟预算内），日志末行 `FINISHED no_qualified_direction`；`status=completed`、`smoke=false`、`branch_failures=[]`；158 单元（回放 94 + 查询 36 + 拟合 8 + 确认 20）、24 份 `state.pt`。
- 数据门禁 **通过**（`formal_data_gate_passed=true`）：新恢复标签追加 62 / 唯一 62 / 新 62 / 重复 0 / 冲突 0；广回放追加 240 / 新 240；LT15 两 seed 27 帧反例追加 88 全重复、新 0；覆盖 24+ 新成功任务、六场景、4 task_type、双 maneuver、三距离档全过。
- 四臂 × 两 seed × 60/240：全部 `discovery_passed=false`。拟合集仅两处正收益（回放×新恢复 240：+4/−4、+8/−8；恢复×广回放 240：+3/−3、+5/−5），但未迁移到发现集；其余 0/0 或负。
- 排序前二诊断确认：`old_old-60` 结果混杂（旧集 +2/−3、0/−1、−1/0、+1/−2；新集 +5/−5、0/+1、+4/−5、+5/−5）；**`broad_novel-240` 在两套各 4 组合全部联合收益**（旧集 +8/−10、+5/−6、+10/−10、+9/−10；新集 +4/−5、+6/−7、+11/−12、+10/−12），但每条丢 3～4 条旧成功（要求 ≤2）、场景净减超 1（如 S4 −2、S6 −2，收益主要来自 S8 +8），故 `confirmation_passed=false`。
- 判定：`no_qualified_direction`、`qualified_directions=[]`、`main_protocol_passed=false`、无自动部署。数据侧“有效新增”瓶颈已解决；下一步应针对“保留收益同时把旧成功丢失压到 ≤2”设计新计划。明细见 `runs/diagnostics/effective-replay-20261021-v2/summary.md`。
