# LT-20261029-39 · 全零保持时补全安全运动候选

**状态：用户正式执行完成，Agent独立复核通过；八任务执行门槛通过，仅准备下一项验证，不解锁训练或生产。以下命令保留作历史记录，禁止覆盖重跑既有目录。**

## 正式结果与决定

[独立复核报告](../../runs/verification/lt39-independent-review-20261029/review.md)：LT39身份/冻结160源、退出0/completed/formal=true、8唯一任务/24回合/66登记文件全部核验。固定网络、滚动网络、固定教师均8/8，碰撞及0.2m余量违规均0；同时保留LT36、LT37 combined、LT38全部旧成功。相对LT38恢复固定2315、滚动860、教师860/2315。

4135次实际控制和完整H10（41350个转移）、冻结父求解器重放及补全候选审计通过；全部203份保存网络输入重新前向，H320坐标和stop logits逐值一致。109次全零触发、872条额外候选检查、24次实际替换；未干预18回合与父执行保持。控制p50/p95/max为8.269/13.732/74.639ms，超100ms为0。

本轮把局部候选遗漏见证兑现为完整闭环增量，但权重未更新、8条起点预测仅3条原生末点联合达标，安全检查依赖仿真真值几何。训练源八任务通过不代表泛化、网络停止监督修复或整链路实时资格；LT34仍暂停。下一步仅准备按任务去重的旧成功保持/失败恢复验证，不自动启动。

## 目的与唯一主要变量

LT38确认部分状态有安全且原代价更低的非零H10，CEM却选全零保持。本轮只在父求解器选中**整段全零H10**时，增加8条固定常速、零角速度候选检查；安全且原cost严格更低才替换。非零父输出、终端转向、原参考进度及整个CEM搜索保持。

参考[Nav2 DWB](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/dwb_controller/)速度候选生成/仿真/评分分工与[iCEM](https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf)执行已评估候选的原则。仅做最小补全，不引入ROS、完整DWA/iCEM、相关噪声或外部实现。固定直行/倒车不能替代转弯规划，也不保证全程可达。

同八任务×网络固定/网络滚动/固定教师，共24新回合；同时复用LT36、LT37 combined、LT38的冻结完整结果，比较成功并集，不挑任务拼开关或用新增抵消丢失。规范：[MPC-RESCUE-001～005](../../spec/zero_hold_motion_rescue.md)。不训练，不修改stop或标签。

## 冻结输入、环境与参数

- 工作目录：`D:\program\endtoend Parking`；conda `endtoend-parking`、Python3.12，`D:\conda\envs\endtoend-parking\python.exe`，CPU线程1。
- 新prepared：`runs/prepared/zero-motion-rescue-20261029-v1/config.json`；SHA256 **c99e5054bc5e2367b531c2781aaebd442b88632ae59830eecd705db56c3fa12c**。
- 冻结160源：原LT38的157源+新增3个scripts，原源码/产物保持；包括三基线身份、旧模型/数据/教师和固定12项保存输入短检，禁止改摘要绕过。
- 父控制契约为LT38 ungated，prepared SHA `3a146dbcf58096d7c109d59cf53b12a7b45b515a3ffb320958e65380f0d78bb6`。三基线为`runs/diagnostics/lt36-feasible-cem-20261029/results`、`lt37-runtime-refinement-20261029/results`的combined，以及`lt38-ungated-refinement-20261029/results`。
- 父权重`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA `359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c`；无模型更新或新训练检查点。
- 原2918条训练源中的固定八索引`[0,433,860,1290,1720,1975,2315,2]`，按task_id去重，读取原cohort；教师为原LT34 `teacher-<index>.npz`。不是未见测试集，不重新生成标签或扩大源。
- 原生H320/dt0.2/stop0.5，固定只预测一次、滚动每10步正常重预测，无教师长度截断、专家回退或参考修补。
- MPC原时间进度、H10/dt0.1/80×5/原代价与权重、seed=原索引、限幅2m/s与1rad/s、每回合600步、原车位双阈值不变；原分布保护、候选择优和安全终端证书保留。
- 固定补全集合顺序：`v=[-2,-1,-0.5,-0.2,0.2,0.5,1,2]`m/s，omega=0，保持10步；原400随机样本及RNG不增。**每触发一次额外最多8条确定性H10候选检查**，如替换再做最终标量复核；不是总计算预算不变。安全但不降原cost仍保持零；不引入“强制前进”或新目标位置项。
- 原6×3m矩形/0.2m余量/扫掠分辨率，所有实际输出用原标量完整H10复核；几何来自simulator_ground_truth，不是纯网络或真实底盘安全资格。
- run总预算7200秒；控制计时含补全与最终验证，排除感知/网络/离线审计；无resume、自动重试或下一阶段。

## 实际完成的短检

- 指定conda下81项聚合测试通过，1.163秒：`-m unittest tests.test_zero_hold_motion_rescue tests.test_zero_hold_motion_study tests.test_ungated_refinement tests.test_ungated_refinement_study tests.test_runtime_refinement tests.test_runtime_refinement_study tests.test_feasible_cem_mpc tests.test_feasible_cem_study tests.test_strict_footprint_runtime -q`。
- 新3脚本/2测试py_compile通过；定向Spec检查通过。覆盖严格降cost/平局不替换、完整后段风险、首动作零但未来非零不触发、终端保持、坏初态、实际补全热启动重放/计数、伪造证据拒绝、三基线并集/超时/防覆盖。
- [保存输入预检](../../runs/verification/zero-motion-rescue-20261029/probes-v1.json)：固定12输入来自2任务/4配对分支，不是12个独立任务。9项原全零保持均被策略替换为安全、原cost严格更低的非零H10；另3项原非零输出逐值不变，全部CEM代信息/RNG一致。仅未执行的保存状态反事实，不证明原起点任务恢复。
- [入口短检](../../runs/verification/zero-motion-rescue-20261029/smoke-v1/report.json)：#0三分支各12步，共36实际步，5.534秒完成、formal=false；零风险、最大控制调用15.243ms。此段前缀补全干预0次，不冒称实际脱困证据。
- inspect核验10登记文件、360个选中H10标量转移、36次父求解器重放、原始输出与首次干预前前缀。实际干预后的热启动/审计在两步定向测试覆盖，正式仍需实测。

## 预计耗时与资源

run约3–15分钟，inspect另约1–8分钟，合计约5–25分钟；依据LT38的5855步/153秒、新入口36步/5.534秒、逐状态父重放及最多额外8条标量H10检查估算。实际轨迹、零保持触发频率、文件压缩和机器负载会改变耗时，估计不确定；最坏24×600步，run预算2小时，耗尽记incomplete。

CPU单线程、无GPU；建议空闲内存4GB、磁盘2GB。上一轮有单次103.8ms，本轮不取消100ms门槛，也不承诺新增检查后必达实时。

## PowerShell命令：前一步成功后再继续

### 1. 环境、身份和目录保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt39Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt39Prepared = 'runs/prepared/zero-motion-rescue-20261029-v1'
$lt39Root = 'runs/diagnostics/lt39-zero-motion-rescue-20261029'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt39Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'c99e5054bc5e2367b531c2781aaebd442b88632ae59830eecd705db56c3fa12c') { throw 'LT39配置漂移' }
& $lt39Python -X utf8 scripts/compare_zero_hold_motion.py --phase precheck --prepared $lt39Prepared
if ($LASTEXITCODE -ne 0) { throw '代码/源/旧基线/保存输入身份漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt39Root) { throw '已有目录，保留并反馈，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt39Root | Out-Null
```

### 2. 正式24回合

```powershell
& $lt39Python -u -X utf8 scripts/compare_zero_hold_motion.py --phase run --prepared $lt39Prepared --output "$lt39Root/results" 2>&1 | Tee-Object -FilePath "$lt39Root/console.log"
$lt39Exit = $LASTEXITCODE
[ordered]@{task='LT39';stage='run';exit_code=$lt39Exit} | ConvertTo-Json | Set-Content -LiteralPath "$lt39Root/exit.json" -Encoding UTF8
if ($lt39Exit -ne 0) { throw '未完整完成，保留现场并反馈，不重试' }
```

按原八索引顺序，每源network_fixed/network_rolling/teacher_fixed各一条BEGIN与完成行，共24条；显示success/collision/margin/steps/rescues/p95。超时、振荡、零保持和safety_stop是有效负结果，不因未达标重跑。只有全部完成才status=completed，指标失败仍可正常退出0。

### 3. 完整实际执行与补全审计

```powershell
& $lt39Python -X utf8 scripts/compare_zero_hold_motion.py --phase inspect --prepared $lt39Prepared --output "$lt39Root/results" 2>&1 | Tee-Object -FilePath "$lt39Root/inspect.log"
if ($LASTEXITCODE -ne 0) { throw '身份/父求解器/固定候选/实际输出核验失败，停止并反馈' }
```

inspect用实际热启动重放冻结父求解器、独立标量复核所有补全候选与原cost、实际选中H10/转移、原停止/坐标转换、第一次补全前的LT38实际前缀。父重放用于证明未改CEM，不声称独立新优化器。补全使后续输入变化是预期下游影响，不强行替换为旧输入。

## 产物、异常与回退

反馈完整目录`runs/diagnostics/lt39-zero-motion-rescue-20261029`，至少保留console.log、exit.json、inspect.log及results全部内容。

results有config/report/artifacts、24份执行JSON和NPZ、16份完整网络预测NPZ，共**66份登记文件**；artifacts.json不自登记。JSON额外记录父输出、固定候选安全/cost/选择及补全检查/干预计数；无新权重或训练检查点。

- Ctrl+C、预算耗尽、非有限、检查器故障、父重放/补全/实际H10审计不一致：记incomplete并退出，不跳过/续跑/自动重试。
- 父无安全控制仍走原safety_stop，不借补全吞掉异常或输出危险首动作。
- 必须修复时另冻prepared与结果目录，重新短检，不改摘要、拼接或覆盖已完成/不完整结果。
- 禁止覆盖LT31/34–38所有权重/配置/产物、LT39 prepared v1/probes-v1/smoke-v1和任何已有正式结果。
- 无依赖/迁移/默认工厂变更；停用新CLI即回退。正式由用户执行，不自动训练/部署/125/300或启动下一长任务。

## 验收和明确退出条件

1. 核LT39/160源/原权重数据/退出0/completed、8唯一任务/24回合/66文件及全部SHA。核原生预测、父重放、固定候选与实际H10/转移，不能只看成功汇总。
2. 同时按task_id对照LT36、LT37 combined、LT38。任何旧成功丢失或风险立即拒绝，新增不抵消；成功并集要求固定8/8、教师8/8、滚动至少保留原7成功，零碰撞/余量。
3. 所有观测MPC.compute<=100ms；报告补全触发/候选检查/干预与完整p50/p95/max/超限比例，不把控制器计时当整链路实时资格。
4. 教师仍失败：停止本次固定运动补全，转控制/可行推进契约；教师恢复但固定网络仍失败：核原生发出终点/轨迹与控制契约，不宣称模型已学会停止；固定全过但滚动回退：转访问状态覆盖/重规划连续性。
5. 无论失败在哪，不加轮数、权重、速度集合或seed延续同一具体修正，不按任务挑控制开关。通过只允许准备下一项验证，不自动解锁LT34、全池/125/300或生产。
