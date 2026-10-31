# LT-20261029-38 · 撤销硬进度门控的单变量修正

**状态：用户执行完成，Agent独立复核完成。简单撤门控方案因教师回退、2315未恢复及单次时延超限拒绝，不扩大或训练。**

## 正式结果与退出决定

24回合153.219秒完成、退出0，157源/8唯一训练任务/66文件通过。Agent重做inspect核验5855实际步/58550个选中H10转移与5855次冻结父求解器重放，256次完整网络预测独立重算一致，零碰撞/余量。详见[正式复核报告](../../runs/verification/lt38-independent-review-20261029/review.md)。

- 网络固定相对LT37从5→7/8，恢复860、1290，2315仍失败；相对LT36同为7/8但丢2315换860，不能抵消。
- 网络滚动保持7/8和原7成功；教师从8→6/8，丢860、2315。门控修复与原严格关口均失败，860网络固定恢复不足以放行。
- 全5855次MPC调用有1次103.7985ms（433教师step123），原任意超限规则未过；不忽略离群点或用重跑取消结果。
- 2315合格教师与网络固定的600步控制/实际状态逐值相同，两者2911/3000代无可行样本、450步全零保持。12项保存状态/参考输入与108条预设常速H10检查（仅2任务、4配对分支），其中9项零保持输入均找到安全、原cost更低且靠近目标的非零序列，证明至少这些输入存在候选覆盖不足；尚未执行恢复。
- 停止简单开/关门控的这套修复，优先处理可行推进候选覆盖与执行进度契约，不加剂量延续、不挑任务拼最优开关，不自动训练或开启下一长任务。

下方命令保留为执行记录，已有prepared与正式产物禁止覆盖。旧LT34、125/300及部署保持暂停。

## 目的与唯一变量

相对LT37 combined仅关闭progress位姿硬门控，恢复原MPC时间推进/原小窗纠偏；分布保护、候选择优、安全终端转向、批量检查及标量最终H10验证全部保留。直接复用项目既有方法，不引入新的进度算法或外部依赖。

要回答：LT37固定参考丢失1290/2315，是否能够通过这一个改动恢复，并保留滚动7/8和教师8/8。不能预设关闭门控必成功，不扫参数、seed或挑最佳配方。

新跑同八任务×网络固定/网络滚动/教师固定共24回合；对照复用LT36以及LT37 combined的完整冻结产物。8个按task_id去重的训练任务不是未见测试集，不扩展到2918源或125/300。

规范：[REF-UNGATE-001～004](../../spec/ungated_reference_progress.md)。无训练、专家回退、参考修补或生产默认变更。

## 冻结身份、参数和环境

- 工作目录：`D:\program\endtoend Parking`；conda `endtoend-parking`，Python3.12，`D:\conda\envs\endtoend-parking\python.exe`，CPU线程1。
- 新配置：`runs/prepared/ungated-refinement-20261029-v1/config.json`；SHA256 **3a146dbcf58096d7c109d59cf53b12a7b45b515a3ffb320958e65380f0d78bb6**。
- 配置冻结157执行源（LT37原154+新增3个scripts）、全部旧模型/数据/教师、LT36和LT37产物身份、当前源码的保存状态短检。禁止手改摘要绕过precheck。
- LT37配置：`runs/prepared/runtime-refinement-20261029-v1/config.json`，SHA `74d83b81b766457d2e29502c456ce5a32b76e279fb184351edb9a89d5c29d0f3`；基线`runs/diagnostics/lt37-runtime-refinement-20261029/results`。
- LT36配置：`runs/prepared/feasible-cem-20261029-v3/config.json`，SHA `ef300846382c6f495600d69f925ed72091ca3bf6c3c9925d7ee08a4744ac8107`；原基线引用保持。
- 父权重：`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA `359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c`。不改网络，不产生模型检查点。
- 固定八索引：`[0,433,860,1290,1720,1975,2315,2]`，来自2918条训练源的原cohort；教师仍原LT34 `teacher-<index>.npz`，不重新生成标签。
- 网络原生stop0.5/H320/dt0.2，不用教师长度；固定只预测一次、滚动每10步正常重预测。后续输入随执行变化属于本变量的下游影响，只要求首次预测/首控制与LT37一致，不强制不同状态使用相同后续预测。
- MPC H10/dt0.1/80×5/原权重/原限幅2m/s与1rad/s、seed=原索引、上限600步、原车位双阈值。分布alpha0.1、标准差下限0.1初始值、精英<2不拟合，候选及终端逻辑均沿用LT37。
- 原6×3m车体、0.2m余量和原扫掠分辨率；使用simulator_ground_truth几何，终端转向记干预，不当作网络原生停止收益。
- 正式run预算7200秒，无resume或自动重试；观测时延仅MPC.compute，包含控制安全检查和终端模式，不含感知、网络和离线审计重放。

## 已完成有界预检

- 指定conda下67项聚合测试通过，0.945秒：`-m unittest tests.test_ungated_refinement tests.test_ungated_refinement_study tests.test_runtime_refinement tests.test_runtime_refinement_study tests.test_feasible_cem_mpc tests.test_feasible_cem_study tests.test_strict_footprint_runtime -q`。
- 新3脚本/2测试py_compile与REF-UNGATE定向Spec检查通过。
- [三个保存状态短检](../../runs/verification/ungated-refinement-20261029/probes-v1.json)：860 step150进度54→55、1290 step300进度152→156、2315 step500进度106→107；门控开启均停留原索引。相同新窗口下父求解器输出/所有代/代价/证书/RNG精确一致，最终H10安全。
- **860、2315这两个保存状态的新首动作仍为零。** 因此只确认索引不再被门控锁住，不能据此声称动力学困境或原任务已恢复；完整从起点运行可能访问不同状态，需要实测。
- [smoke-v1](../../runs/verification/ungated-refinement-20261029/smoke-v1/report.json)：#0三分支各12步，共36实际步，2.584秒完成，formal=false；零碰撞/余量，最大控制调用6.906ms。inspect核10份登记文件、360个选中H10标量转移、36次冻结父求解器重放及原参考/停止/首次前缀，全部一致。

## 耗时与资源估计

正式run约3–15分钟，inspect另约1–8分钟，合计约5–25分钟；估算依据LT37的7466步/134秒、新入口36步/2.584秒，并额外对每个实际状态重放一次冻结父求解器，inspect再做一次。路径改变可能导致最多24×600步，机器负载、JSON和压缩也会改变耗时，估计不确定；正式run上限2小时，耗尽即incomplete。

CPU单线程、无GPU，建议空闲内存4GB、磁盘2GB。inspect重放是离线审计，不计入控制实时数据；不得将其耗时隐藏在MPC.compute指标内。

## PowerShell命令：依次运行

### 1. 环境、身份和目录保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt38Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt38Prepared = 'runs/prepared/ungated-refinement-20261029-v1'
$lt38Root = 'runs/diagnostics/lt38-ungated-refinement-20261029'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt38Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '3a146dbcf58096d7c109d59cf53b12a7b45b515a3ffb320958e65380f0d78bb6') { throw 'LT38配置漂移' }
& $lt38Python -X utf8 scripts/compare_ungated_refinement.py --phase precheck --prepared $lt38Prepared
if ($LASTEXITCODE -ne 0) { throw '代码/数据/旧基线/短检身份漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt38Root) { throw '已有目录，保留并反馈，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt38Root | Out-Null
```

### 2. 正式24回合

```powershell
& $lt38Python -u -X utf8 scripts/compare_ungated_refinement.py --phase run --prepared $lt38Prepared --output "$lt38Root/results" 2>&1 | Tee-Object -FilePath "$lt38Root/console.log"
$lt38Exit = $LASTEXITCODE
[ordered]@{task='LT38';stage='run';exit_code=$lt38Exit} | ConvertTo-Json | Set-Content -LiteralPath "$lt38Root/exit.json" -Encoding UTF8
if ($lt38Exit -ne 0) { throw '未完整完成，保留现场并反馈，不重试' }
```

日志按八索引顺序，每任务出现network_fixed、network_rolling、teacher_fixed的BEGIN和完成行，共24条完成行；显示到达/碰撞/余量/步数/p95。600步失败、振荡、safety_stop和零保持均是有效负结果，不因未达标重跑。全部回合完成才status=completed，指标失败也可能正常退出0。

### 3. 独立实际执行与唯一变量核验

```powershell
& $lt38Python -X utf8 scripts/compare_ungated_refinement.py --phase inspect --prepared $lt38Prepared --output "$lt38Root/results" 2>&1 | Tee-Object -FilePath "$lt38Root/inspect.log"
if ($LASTEXITCODE -ne 0) { throw '身份/实际执行/冻结求解器/原生预测核验失败，停止并反馈' }
```

逐个实际状态重建原时间进度/参考窗口，以保存状态和RNG重放冻结LT37 combined父求解器，检查所有代统计/mask/精英/候选选择/终端证书及实际输出一致；这是唯一变量检查，不声称另一套独立优化器。原标量判据另核全部选中H10/真实转移，原停止长度和坐标转换另核。

## 产物、异常、恢复与回退

反馈完整目录`runs/diagnostics/lt38-ungated-refinement-20261029`，保留console.log、exit.json、inspect.log与results全部内容。

results有config/report/artifacts、24执行JSON与NPZ、16完整网络预测NPZ，共**66份登记文件**；artifacts.json不自登记。没有训练检查点。

- Ctrl+C、预算耗尽、非有限值、检查器故障、原进度/父求解器或实际H10不一致：记incomplete并退出，不跳过、不续跑、不自动重试。
- 无安全序列走原safety_stop，禁止危险首动作；坏初态不得伪装为安全保持。
- 身份漂移先反馈。必要修复后新prepared/结果目录重新短检，不修改摘要、拼接不完整回合或覆盖原结果。
- 禁止覆盖LT31/34/35/36/37所有权重/配置/结果，及LT38 prepared v1、probes-v1、smoke-v1、任何已有正式产物。
- 无新依赖/迁移或生产工厂注册；停用新CLI即回退原运行方式，不改变默认系统，不自动训练/部署或启动下一长任务。

## 后续验收与两级退出条件

1. 先核LT38/157源/原权重数据/退出0/completed、8唯一任务/24回合/66文件及完整SHA。检查全部H10/真实转移、原生停止和父求解器重放，禁止仅看汇总。
2. 同时对LT36和LT37逐任务比较三分支，任何旧成功丢失或风险均拒绝，新增不抵消丢失。
3. **门控修正关口**：恢复1290、2315固定成功，保留LT36固定7/8的所有旧成功、LT37滚动7/8及教师8/8，零碰撞/余量、所有观测控制调用<=100ms。任一失败，排除本次简单撤门控方案，转参考/安全推进契约，不加剂量延续。
4. **原LT37严格关口**：在上项全部通过基础上，还需860网络固定恢复。若仅上项通过，报告“门控回退已修、860网络仍未解决”，不能放大成整体候选通过。即使上项全部通过，也不证明原生轨迹坐标/首停已全合格或全池泛化。
5. 完整固定恢复但滚动回退，转访问状态覆盖/重规划连续性；860仍高比例零动作或坏终点，分别核实际发出参考和可行推进，不加终点损失冒充控制修复。
6. 不管结果如何，本计划不自动解锁LT34、125/300或生产。只有原严格关口通过才可准备下一项验证，由用户另行执行。
