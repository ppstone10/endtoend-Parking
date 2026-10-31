# LT-20261029-40 · 全零运动补全的125任务保持验证

**状态：用户正式执行完整完成，Agent复核完成；联合验收未通过，候选拒绝晋升。以下命令保留为历史记录，禁止覆盖重跑既有目录。**

## 正式结果与决定

[独立复核报告](../../runs/verification/lt40-independent-review-20261029/review.md)：162源、原模型/数据/旧产物及配置身份一致，run/inspect退出均0，125唯一任务/250回合/752文件完整。父102/125，候选104/125，恢复292/296且保留父臂全部成功；两臂零风险。历史120成功仍缺18（父臂同样失败），原生93仅保住90；不得用2个新增抵消。

候选43088次compute中1次117.407ms（133/step519，未触发补全），违反100ms硬门槛。完整inspect核869130个H10标量转移、8783个网络前向；Agent另有界独立核全量文件/动力学/到达/时延与配对前缀，抽查44完整网络前向及66安全H10，范围详见报告，未重复长审计。

剩余21例均600步非零运动振荡，零保持/补全触发均0；20例未进入位置阈值，S3十例最近仍距目标约6米以上。停止扩大本组合，转向固定参考与正常重预测的收敛归因，不追加剂量；训练/候选/生产资格仍false，未自动运行下一任务。

## 目的与唯一主要变量

LT39已在八个训练任务上取得网络固定/滚动及教师各8/8，未丢旧成功且零风险。本轮检验其能否在既有失败与成功任务中保持：复用LT34预先冻结的验证集22回退、8恢复、5残余失败、93原生成功，四组保留重叠关系，按task_id并集**125任务**。

同一权重、输入、正常重预测、控制seed和预算下，每任务配对LT38父控制器（parent）与LT39候选（candidate），共**250回合**。唯一算法变量仍是整段H10全零后的固定8条运动候选补全；不调参数、不训练、不重新生成教师、不做专家替换或安全前缀准入。不会把当前控制收益当作原生首停已学会。

历史93原生成功、22回退及8恢复的成功并集为120任务，候选必须全部保持；另须保留新父臂的全部成功。历史LT32/33的control_seed=0，新两臂seed=原验证集索引，历史仅作逐任务成功保持约束，不宣称历史执行逐值复现。规范：[RESCUE-KEEP-001～004](../../spec/zero_motion_retention.md)。

## 冻结输入、模型与代码

- 工作目录：`D:\program\endtoend Parking`；环境：`D:\conda\envs\endtoend-parking\python.exe`（Python3.12），CPU线程1。
- prepared：`runs/prepared/zero-motion-retention-20261029-v1/config.json`；SHA256 **443ee0150933fc0c2a9f3e9db58f8899ca3677a98106d2c4aaad66208a708115**。
- 冻结162源：LT39原160源+`scripts/zero_motion_retention_capture.py`、`scripts/validate_zero_motion_retention.py`。不改旧源码、默认工厂或旧结果；源变化必须反馈，不能重写摘要绕过。
- 验证源：`runs/verification/surround-target-rebuild-lineage-20261027/val/val.npz`，300任务，SHA `b8250edeb24f966374cb1900a101c6239723888183a37c62046214d14885cc38`；manifest、LT32结果与配置、LT33复核、LT34原清单及LT39全部正式产物均冻结核验。
- 子集缓存：上述prepared的`cohort.npz`，SHA `749e903d4c8200e3d2acc046e9826777172a35beeb1953e61e9a54c502cc580c`。保留原索引、task_id、场景和全部组成员；125来自验证集，不能用训练源的同编号替代。是已被多次评估的条件回归集合，不称独立未见测试。
- 原权重：`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA `359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c`；无更新，无新训练检查点。
- 原生stop0.5/H320/dt0.2；全部network_rolling，每10步正常重预测，不截教师长度、不修补网络坐标。
- MPC H10/dt0.1/80×5/原代价与权重/原时间进度/每回合600步，seed=验证集原dataset_index；限幅2m/s、1rad/s，原6×3m矩形、0.2m余量和原车位位置/角度双阈值保持。
- candidate全零时额外检查`v=[-2,-1,-0.5,-0.2,0.2,0.5,1,2]`、omega=0的8条固定H10，完整安全且原cost严格下降才替换；补全及最终标量复核计入控制耗时。parent无该补全。
- 安全几何来自simulator_ground_truth；两臂均保留既有安全终端转向，计入各自controller_stats，不能称纯网络或底盘安全资格。
- run和inspect各有14400秒（4小时）总预算；无resume、自动重试、自动下一阶段。

## 已完成的前置测试

- 指定conda单线程下94项聚合测试通过，1.166秒：`-m unittest tests.test_zero_motion_retention tests.test_zero_hold_motion_rescue tests.test_zero_hold_motion_study tests.test_ungated_refinement tests.test_ungated_refinement_study tests.test_runtime_refinement tests.test_runtime_refinement_study tests.test_feasible_cem_mpc tests.test_feasible_cem_study tests.test_strict_footprint_runtime -q`。
- 新2脚本/1测试py_compile、定向Spec检查及下列3段PowerShell命令语法解析通过；覆盖重叠去重、125/250/752分母、历史120保持、配对成功保持、新增不抵回退、初态风险、任何单次超限、无增量、异常incomplete、防覆盖、SHA漂移、首次干预前原始预测和实际控制一致、零步不安全初态的负结果及伪造拒绝。
- [有界短检](../../runs/verification/zero-motion-retention-20261029/review.md)：固定验证索引0/183，两臂各12步，48实际步、5.001秒，零风险、补全干预0次；控制p50/p95/max=8.322/13.536/14.900ms。
- inspect通过14登记文件、480个选中H10标量转移、48次冻结求解器重放、两任务配对前缀及全部8次网络完整前向重算。短检formal=false、未到达不算正式失败或成功；没有用这48步声称补全脱困，干预行为沿用LT39正式与定向测试证据。
- 本轮完全沿用既有冻结、采集和审计方法，无新算法或外部依赖，无需重复方法调研。正式收益仍未验证。

## 预计耗时与资源

run约15–90分钟，inspect另15–90分钟，合计约30分钟–3小时；依据LT39的4135实际步/105秒，以及本轮48步/5秒（含固定开销）估算，250×600最多150000实际步。父臂可能持续零保持到600步，完整原始输出、压缩、逐步求解器重放和网络重算均有额外成本；机器负载与实际轨迹使估算不确定，两阶段各4小时耗尽即incomplete，不自动续跑。

CPU单线程、无GPU。建议可用内存4GB、磁盘8GB；48步短检约1.59MB，按最坏150000步线性外推约5GB，并为场景/长参考差异留空间。控制<=100ms是本轮实测门槛，不是预先保证，也不包括感知和网络整链路。

## PowerShell命令：前一步成功后再继续

### 1. 环境、身份及输出保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt40Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt40Prepared = 'runs/prepared/zero-motion-retention-20261029-v1'
$lt40Root = 'runs/diagnostics/lt40-zero-motion-retention-20261029'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt40Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '443ee0150933fc0c2a9f3e9db58f8899ca3677a98106d2c4aaad66208a708115') { throw 'LT40配置漂移，停止并反馈' }
& $lt40Python -X utf8 scripts/validate_zero_motion_retention.py --phase precheck --prepared $lt40Prepared
if ($LASTEXITCODE -ne 0) { throw '源码/数据/旧结果/模型身份漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt40Root) { throw '已有LT40目录，保留并反馈，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt40Root | Out-Null
```

### 2. 正式250回合

```powershell
& $lt40Python -u -X utf8 scripts/validate_zero_motion_retention.py --phase run --prepared $lt40Prepared --output "$lt40Root/results" 2>&1 | Tee-Object -FilePath "$lt40Root/console.log"
$lt40RunExit = $LASTEXITCODE
[ordered]@{task='LT40';stage='run';exit_code=$lt40RunExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt40Root/exit.json" -Encoding UTF8
if ($lt40RunExit -ne 0) { throw 'LT40未完整完成，保留现场并反馈，不重试' }
```

按冻结125原索引升序，每任务parent后candidate，各一条BEGIN和完成行，共250回合。完成行有success/collision/margin/steps/rescue。配对执行即使出现指标回退仍会完成既定分母；这只为收集拒绝原因，不更改候选或追加任务。正式status=completed且formal=true才是完整结果；exit0不能单独表示验收通过。

### 3. 产物与实际执行独立审计

```powershell
& $lt40Python -u -X utf8 scripts/validate_zero_motion_retention.py --phase inspect --prepared $lt40Prepared --output "$lt40Root/results" 2>&1 | Tee-Object -FilePath "$lt40Root/inspect.log"
$lt40InspectExit = $LASTEXITCODE
[ordered]@{task='LT40';stage='inspect';exit_code=$lt40InspectExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt40Root/inspect-exit.json" -Encoding UTF8
if ($lt40InspectExit -ne 0) { throw 'LT40证据核验未通过，保留并反馈，不重试' }
```

inspect每回合输出AUDIT累计转移/预测重算数，最后status=verified、tasks=125、episodes=250、artifacts=752；核冻结身份、实际转移/完整H10/父重放/候选原cost/计数、全部原生停止及坐标、全部保存网络输入前向、首次干预前配对一致。审计计时不混入实际控制计时。

## 产物、异常、恢复及禁止覆盖

反馈完整目录`runs/diagnostics/lt40-zero-motion-retention-20261029`，包括console.log、exit.json、inspect.log、inspect-exit.json及results全部内容。

- results登记config.json/report.json、250份逐回合JSON、250份执行NPZ、250份完整预测NPZ，共752份；另artifacts.json不自登记。无新权重或训练检查点。
- report包含两臂125分母、逐任务结果、四组与分场景指标、丢失/新增索引、历史120保持、初态风险和逐回合完整p50/p95/max/超限；JSON保留每次compute_ms及补全/终端计数，Agent复核时汇总全量控制分布。
- 安全停止、600步超时、规划失败均保留为负结果；初态已侵入也不得移除任务。无法验证的安全异常、非有限值、身份漂移、文件/转移/求解器/预测不一致或预算耗尽：incomplete，停止并反馈。
- Ctrl+C或非零退出后禁止直接重跑、续跑、补文件、拼接或改SHA；先由Agent诊断，确需修复时新版本prepared和新结果目录，再人工执行。
- 禁止覆盖LT31/34–39权重、配置、正式结果、所有已有prepared/短检以及当前LT40任何已有目录。未创建正式目录前退出可反馈；即使仅创建空目录也不自行删目录重来。
- 无依赖/迁移/默认工厂变更；停止使用新入口即可回退。不自动启动训练、全300、下一阶段或部署。

## Agent后续验收与明确退出

1. 核任务身份、源码/配置/权重/数据及SHA、run/inspect退出0、completed/formal=true、125唯一task_id、250回合与752文件完整性，再独立读证据。
2. 候选必须保住历史成功并集120任务，以及新父臂的全部成功；四组分别展示，不以新增抵丢失。任一丢失立即拒绝该候选，不改速度集合、轮数、seed或阈值延续同一配方。
3. 候选125任务的碰撞、0.2m余量违规及初态风险均为0，所有candidate compute<=100ms；任一风险/超限拒绝。parent风险/耗时也完整报告，不能用其较差来豁免候选硬门槛。
4. 保持与安全/时延均通过后，至少1个同条件parent失败由candidate恢复才有配对增量；否则`retention_passed_no_increment`，不得声称扩大后的新增收益。另列五残余失败实际恢复，不挑结果拼开关。
5. 通过仅`ready_for_next_validation=true`；candidate_qualified、lt34_training_unlocked、production_qualified仍false。依据失败任务的实际原生参考/重规划状态/控制选择证据决定下一转向，不自动恢复训练或追加剂量。
