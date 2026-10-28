# LT-20261022-19：S8保持来源完整闭环核验

状态：**已完成**（2026-10-22 17:2x，用户明示授权Agent代跑该长任务的命令块）。18回合、20.389秒；两臂9源均0成功/9碰撞，缓存C240同源9/9成功。结果见文末「执行结果」。入口、冻结配置、零回合precheck、编译和17项定向测试此前已完成。本轮已执行60.312秒来源审计、89.489秒学习短检、43.768秒独立核验及定向测试；继续串行这批评测预计会越过AGENTS累计5分钟边界，因此原本交用户执行。

## 目的与唯一主要变量

区分“保持查询loss下降但对应完整训练任务也失败”和“完整训练任务保持住、未迁移至挑战”。只改变模型检查点：原LT18 policy与新S8保持学生，在完全相同的9个已固定训练来源上比较。**不训练、不调权、不查询教师、不生成新梯度标签。** 两顺序代表各自策略/来源组合，不能当作9个不同任务（物理任务有重叠）。不含7挑战/val/selection。

## 已有证据与输入版本

- 工作目录`D:\program\endtoend Parking`；仅`D:\conda\envs\endtoend-parking\python.exe`（endtoend-parking、Python3.12.13、torch2.13.0+cpu、numpy2.5.2），CPU单线程。
- 原LT18完整四模型/44回合；新短检完整双模型/22回合。新学生#2149为423/478步严格到达，挑战成功/碰撞仍4/3、3/4，S8未恢复；保持查询上的漂移loss降低约24%/36%，不能当作来源任务成功。
- 指定来源：20261020为2338/2351/2375；20261021为2330/2338/2351/2370/2378/2398。它们由前轮实际S8梯度源、C240完整无接触成功和至少3个clean非零窗口筛出，在本次候选评测前已冻结。所有9个来源的C240全回合仍有规划余量侵入，不能标为全程严格教师；27个选中窗口自身干净。
- 原policy检查点：`runs/diagnostics/s6-policy-transfer-preflight-20261022-v1/policy_targets-<seed>/state.pt`；新保持学生：`runs/diagnostics/s8-preservation-learning-preflight-20261022-v1/<seed>/state.pt`。四者均Adam268，原阈值/输出/MPC不变。脚本只读权重，输出不写checkpoint。
- 旧C240来源结果缓存：`runs/diagnostics/s8-preservation-audit-20261022-v1/report.json`，不会冒充本次重跑。
- 冻结配置：`runs/prepared/s8-preservation-source-validation-20261022-v1.json`，SHA256=`792f42ee8c6597b4cb7bf265b2fb2141ed23ffeef50b21ec6ebc3d38f37752dc`，登记144代码和704输入/产物文件。当前工作树未提交，文件SHA是本次版本依据；配置同时冻结实际检查点、根报告和学生子报告，不手改hash或重建配置绕过漂移。
- 验收契约：[S6-TRANSFER-007](../../spec/s6_policy_transfer.md)；本轮调查与短检结果见[摘要](../../runs/diagnostics/s8-preservation-learning-preflight-20261022-v1/summary.md)。

## 按顺序复制执行的PowerShell命令

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$sourcePython = 'D:\conda\envs\endtoend-parking\python.exe'
$sourceConfig = 'runs/prepared/s8-preservation-source-validation-20261022-v1.json'
$sourceOutput = 'runs/diagnostics/s8-preservation-source-validation-20261022-v1'
$sourceConsole = 'runs/diagnostics/s8-preservation-source-validation-20261022-v1.console.log'
if (-not (Test-Path -LiteralPath $sourcePython)) { throw '指定conda Python不存在' }
if ((Get-FileHash -LiteralPath $sourceConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne '792f42ee8c6597b4cb7bf265b2fb2141ed23ffeef50b21ec6ebc3d38f37752dc') { throw '冻结配置漂移，停止' }
if ((Test-Path -LiteralPath $sourceOutput) -or (Test-Path -LiteralPath $sourceConsole)) { throw '已有产物，停止并保留旧目录' }
& $sourcePython scripts/validate_s8_preservation_sources.py --config $sourceConfig --output $sourceOutput 2>&1 | Tee-Object -FilePath $sourceConsole
$sourceExit = $LASTEXITCODE
if ($sourceExit -ne 0) { throw "核验退出码 $sourceExit；不要自动重试" }
Get-Content -LiteralPath "$sourceOutput/report.json" -Tail 35
```

## 预计耗时、资源和正常进度

估计30–90秒，固定总预算120秒（协作式检查，底层操作/落盘可使墙钟略超），依据本轮13来源+指标审计60秒、8关键回合约16秒；尚未实测本次18回合，加载、完整回合长度和机器负载有不确定性。CPU单线程，暂估4–8GB内存、磁盘预留0.5GB，无GPU。

控制台四条`EVALUATE`依次为LT18_policy两顺序3/6回合、preservation_student两顺序3/6回合；最终`FINISHED source_retention_diagnostic_only_no_promotion`。`status=completed`、`stage=decision`、`complete_episodes=18`才算执行完整。`training_updates`、`allowed_to_prepare_training_plan`、`main_protocol_passed`、`automatic_deployment`保持0/false；任何结果都不会自动放行。

## 输出、异常停止、恢复和禁止覆盖

- 输出根config/report，四个`<arm>-<seed>/source-<index>-control0.npz`共18个；根报告包含两臂每来源结果、缓存C240和分顺序配对；同级console.log。没有训练日志或新增checkpoint。
- 异常/非有限/身份漂移/超时/Ctrl+C立即停止，保留目录和console，不能删除run.lock后原地续跑或自动补缺。强杀可能留下running报告或锁，反馈Agent诊断。
- 如需重跑，须核验原因后指定不存在的新v2输出/日志，从头完整核验，不在旧目录补回合。不得覆盖LT18、来源审计、保持短检、银行/模型、冻结配置、独立核验报告或本次失败产物；不重训、不重采教师、不改变权重0.25/阈值/600步/0.2余量。

## 反馈文件和Agent后续验收

回复完成/异常和输出目录；保留根report/config、18个NPZ、同级console。Agent先复核144代码/704冻结文件、配置/产物SHA、日志尾部/退出状态、18分母及状态/命令长度，再分析每来源成功/碰撞/严格余量相对缓存C240和LT18 policy的变化；必要时短时独立重放决定性回合。

若完整训练来源也丢失成功，下一关口是局部保持窗/目标与完整任务之间的约束缺口；若来源保持但挑战未恢复，下一关口是跨任务覆盖/迁移。仅凭查询loss下降或一条成功不能区分两者，也不能推断唯一梯度冲突或架构容量不足。无论哪种结果，都不自动扩大现有配方、晋升候选或串行启动下一长任务，主成绩68.3%/25.7%不变。

## 执行结果（2026-10-22）

- 执行：`status=completed`、`stage=decision`、`complete_episodes=18`、`elapsed_seconds=20.3896`；`training_updates=0`，`allowed_to_prepare_training_plan`/`main_protocol_passed`/`automatic_deployment` 均 false；`decision=source_retention_diagnostic_only_no_promotion`；控制台5行，四条`EVALUATE`依次3/6/3/6回合。
- 身份：配置SHA256仍为`792f42ee…7752dc`；脚本内对144代码+704冻结文件及运行环境全量核验通过；4个检查点、18个NPZ与根`config.json`独立重算SHA与登记值全部一致（23项、0不符）；`report.json` SHA256=`3dc03602…83d3cf`。
- 每臂9源（20261020的2338/2351/2375，20261021的2330/2338/2351/2370/2378/2398）各执行一次完整原MPC回合：
  - 缓存C240（只读复用005）：9/9成功、0碰撞，均值289.7步、末端位置误差0.155、余量违例49.2步；严格余量到达仍为0。
  - LT18 policy：0成功、9碰撞，均值71.7步、末端位置误差9.793、余量违例33.7步。
  - 保持学生：0成功、9碰撞，均值124.7步、末端位置误差6.993、余量违例56.6步。
- 相对缓存C240：两臂均丢失全部9源成功（lost_success_indices为全部9个），碰撞各+9，分顺序配对`success_delta`为−3/−6。
- 相对同源LT18 policy：成功Δ0、碰撞Δ0；学生末端位置误差8/9更小（2375 9.938→9.777、2330 8.202→5.094、2378 9.219→6.103、2398 12.528→2.470、2338 7.453→1.036），步数7/9更长（2338 117→221、2398 6→205、2330 130→191），规划余量违例步数均值33.7→56.6。
- 判读：**完整训练来源上的成功失败在LT18 policy上已经成立**，不是保持学习新引入的回归；保持短检的查询loss下降（约−24%/−36%）不构成完整任务保持，两者必须分开读。按计划分支，下一关口是“局部保持窗/目标与完整任务之间的约束缺口”，不是跨任务覆盖/迁移；也不据此推断唯一梯度冲突或架构容量不足。
- 边界：本轮未训练、未改权重/阈值/600步预算/0.2余量，未重采教师，未覆盖LT18、来源审计、保持短检、模型银行或冻结配置；主成绩68.3%/25.7%与全部门禁不变。

## 独立复核补充

144代码/704冻结文件/19登记产物、18回合分母与配对指标重算通过，6关键回合状态/控制/指标逐元素复现。缓存C240的9成功均有规划余量侵入，不能称全程严格教师。两个模型均0成功/9碰撞，说明局部查询loss下降尚未保住其训练来源完整任务；保持学生只有1/9在首查询时刻前碰撞，不能统一归为窗口太晚。起始C240窗口4/9干净、5/9侵入，下一方法须先验证起点/学生访问状态上的安全可执行监督，不直接补入不安全入口或扫系数。证据：[独立结果摘要](../../runs/diagnostics/s8-preservation-source-validation-20261022-v1/summary.md)。
