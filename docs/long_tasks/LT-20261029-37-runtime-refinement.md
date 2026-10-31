# LT-20261029-37 · 批量安全检查与执行收敛修正

**状态：用户执行完成，Agent独立复核完成。时延、教师与滚动局部收益通过，组合因固定参考回退拒绝，不扩大或训练。**

## 正式结果与退出决定

30回合133.625秒完成，退出0；154源身份、8唯一训练任务及80登记文件通过。Agent重做inspect核验7466实际控制步/74660个选中H10转移；299次完整原生网络预测独立重算、21保存状态8400候选标量复核一致，两条fast全600步执行数组/mask/精英与LT36一致。详见[正式复核报告](../../runs/verification/lt37-independent-review-20261029/review.md)。

- combined网络滚动6→7/8，新增1290、旧成功保持；教师7→8/8，恢复860；均零碰撞/余量。
- combined网络固定7→5/8，丢1290、2315，无新增；860固定仍失败。新增不抵消丢失，组合按原门槛拒绝。
- 全7466次观测控制调用均<100ms，combined合并p95/max为18.667/86.264ms，仅覆盖MPC.compute，不能代替整链路实时资格。
- 单因素：860教师variance 200步、selection 102步成功，progress仍600步失败；1290滚动terminal-only前538步与LT36相同，之后6步安全转向、544步完成。局部证据不外推全八或挑最佳臂替代combined。
- 保存执行显示固定进度长时间锁定、零保持和远处往复；尚无固定回退的完整单因素证据，不唯一归因于progress。停止扩大该组合，下一步隔离进度/参考推进契约，不加配方剂量，不自动训练或启动下一正式任务。

下方命令保留为执行记录；本次prepared/正式产物禁止覆盖重跑。旧LT34与125/300保持暂停。

## 目的、主要变量与对照

针对LT36三个已复现局限：标量安全检查过慢；860稀疏精英导致方差坍缩、参考进度与真实跟踪脱节；1290滚动分支位置合格但航向不合格。优化入口独立于旧冻结源码和生产默认，不训练。

本计划包含相互分开的实验变量，不把组合收益归因于某一个开关。`fast`只批量化原安全判据，必须精确重现LT36；四个单因素预设各只在fast上增加一项改动；`combined`同时启用四项。参数预先固定，不扫描权重/seed，不在看完结果后挑臂。

| 预设 | 相对fast的唯一行为改动 | 正式分母 |
|---|---|---|
| combined | 四项组合，验证整体候选 | 同八任务×固定网络/滚动网络/固定教师，24回合 |
| fast | 无行为改动，仅等价加速 | 860固定教师、1290滚动网络，2回合 |
| variance | 精英不足2不拟合；分布平滑与标准差下限 | 860固定教师，1回合 |
| selection | 比较安全均值、最佳安全样本和安全热启动的原代价 | 860固定教师，1回合 |
| progress | 接近当前参考位置和航向才单调推进 | 860固定教师，1回合 |
| terminal | 真目标位置合格后，以整段旋转安全证书对齐航向 | 1290滚动网络，1回合 |

共30个新回合、8个按task_id去重的训练任务。LT36原24回合直接复用；单因素只支持对应见证任务的局部归因，不能证明全八有效。固定教师仅用于控制契约验证，不是网络运行时回退。

规范：[RUN-REFINE-001～006](../../spec/runtime_refinement.md)。方法参考[iCEM](https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf)的分布平滑/执行已评估候选、[Nav2 MPPI](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/)的批量计算与最终轨迹验证，以及[Rotation Shim](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/configuring_rotation_shim_controller/)的目标航向接管。没有引入ROS或外部实现。进度门控是本项目待验证的方案，不预设保证收敛。

## 冻结输入、代码与参数

- 工作目录：`D:\program\endtoend Parking`；Python：`D:\conda\envs\endtoend-parking\python.exe`，conda `endtoend-parking`、Python3.12、CPU线程1。
- 配置：`runs/prepared/runtime-refinement-20261029-v1/config.json`；SHA256 **74d83b81b766457d2e29502c456ce5a32b76e279fb184351edb9a89d5c29d0f3**。
- 执行源154个：LT36原149源+新增5个scripts；LT34/35/36源码和产物不改。precheck核旧数据、模型、配置、全部登记基线产物，以及新源和19状态等价短检身份，禁止手改摘要。
- LT36配置：`runs/prepared/feasible-cem-20261029-v3/config.json`；SHA `ef300846382c6f495600d69f925ed72091ca3bf6c3c9925d7ee08a4744ac8107`。基线：`runs/diagnostics/lt36-feasible-cem-20261029/results`。
- 原父权重：`runs/training/lt31-terminal-hold-20261027/control/last.pt`；SHA `359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c`，不生成新模型或训练检查点。
- 原2918条训练源中的固定八任务：`[0,433,860,1290,1720,1975,2315,2]`；读取LT34已准备cohort，不重新生成数据。教师为原`runs/training/lt34-emitted-joint-20261028/teacher/teacher-<index>.npz`。
- 保持原H320/dt0.2/stop0.5自由解码，不按教师长度截断；滚动每10步正常重预测，固定网络只预测一次。MPC H10/dt0.1/pop80/5代/原代价/限幅2m/s和1rad/s、seed=原数据索引、每回合最多600步、原车位容差。
- 原6×3m矩形、0.2m余量和原扫掠采样分辨率；最终实际H10仍由原标量检查器完整复核。几何来自simulator_ground_truth，不声称纯网络/纯传感器或真实底盘安全资格。
- 分布保护：精英<2保持旧分布；否则alpha0.1混合旧分布，标准差下限初始值的0.1（v0.1、omega0.08），不增加采样。
- 进度门槛位置0.15m/航向0.1rad，每步最多原catchup_window4；终端采用原目标及原容差，只有整段最短转角和实际H10安全才接管，证书拒绝则保留基础控制。
- 正式预算7200秒，无resume/自动重试/自动续阶段。时延仅计`MPC.compute`，包含安全检查与终端接管，不含感知/网络及审计。

## 已完成预检与资源估计

- 指定conda下54项聚合测试通过（0.734秒）：`-m unittest tests.test_runtime_refinement tests.test_runtime_refinement_study tests.test_feasible_cem_mpc tests.test_feasible_cem_study tests.test_strict_footprint_runtime -q`。
- [19状态等价重放](../../runs/verification/runtime-refinement-20261029/probes-v2.json)：每代mask/精英、实际序列和RNG精确一致；无profile的调用中位95.61ms→11.20ms，批量最大13.04ms。只覆盖这19个调用，不是全回合时延证明。
- [组合入口短检](../../runs/verification/runtime-refinement-20261029/smoke-v1/report.json)：#0三分支各12步，共36步；1.610秒完成、formal=false；零碰撞/余量，三分支p95约6.81–6.97ms、最大7.06ms。inspect独立核验10份登记产物及360个选中H10转移、全部原生预测和教师参考。
- [保存状态终端短检](../../runs/verification/runtime-refinement-20261029/terminal-saved-state-v1/scope.json)：以1290滚动step537之后的存档状态替换起点，6步满足原双阈值，零碰撞/余量；最大8.08ms。它不计入正式30回合，不证明原起点恢复。step569完整旋转余量证书仍被拒绝。
- 正式预计5–20分钟，CPU单线程、无GPU，建议空闲内存4GB、磁盘2GB。依据LT36实际5834步约586秒、19状态安全检查约8.5倍加速及最坏30×600步折算，并考虑推理、轨迹变化、JSON/压缩/审计和机器负载；估计存在较大不确定性。总预算2小时，耗尽记incomplete，不加时。

## PowerShell命令：前一步成功后再执行下一步

### 1. 环境、身份和目录保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt37Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt37Prepared = 'runs/prepared/runtime-refinement-20261029-v1'
$lt37Root = 'runs/diagnostics/lt37-runtime-refinement-20261029'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt37Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '74d83b81b766457d2e29502c456ce5a32b76e279fb184351edb9a89d5c29d0f3') { throw 'LT37配置漂移' }
& $lt37Python -X utf8 scripts/compare_runtime_refinement.py --phase precheck --prepared $lt37Prepared
if ($LASTEXITCODE -ne 0) { throw '代码/旧证据/短检身份不一致，停止并反馈' }
if (Test-Path -LiteralPath $lt37Root) { throw '已有目录，保留并反馈，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt37Root | Out-Null
```

### 2. 正式30回合

```powershell
& $lt37Python -u -X utf8 scripts/compare_runtime_refinement.py --phase run --prepared $lt37Prepared --output "$lt37Root/results" 2>&1 | Tee-Object -FilePath "$lt37Root/console.log"
$lt37Exit = $LASTEXITCODE
[ordered]@{task='LT37';stage='run';exit_code=$lt37Exit} | ConvertTo-Json | Set-Content -LiteralPath "$lt37Root/exit.json" -Encoding UTF8
if ($lt37Exit -ne 0) { throw '未完整完成，保留日志并反馈，不重试' }
```

日志先按八任务原顺序出现combined三个分支，共24条完成行；然后860教师fast/variance/selection/progress、1290滚动fast/terminal，共6条。每行显示success/collision/margin/steps/p95_ms。正常负结果包括超时、振荡、safety_stop或零保持，不因未到达重跑。仅全部30回合完成才有report.status=completed；指标不通过仍可正常退出0。

### 3. 独立产物与执行核验

```powershell
& $lt37Python -X utf8 scripts/compare_runtime_refinement.py --phase inspect --prepared $lt37Prepared --output "$lt37Root/results" 2>&1 | Tee-Object -FilePath "$lt37Root/inspect.log"
if ($LASTEXITCODE -ne 0) { throw '身份/执行/预测/门槛复核失败，停止并反馈' }
```

inspect属于本次人工长任务的一部分：逐个选中H10与实际转移用原标量判据重算；重建RNG/精英统计/原代价/进度门控和终端证书；核原生停止长度与全部发出坐标。两条fast完整回合须与LT36全部执行数组、每代mask/精英精确一致，任何差异中止资格。一般候选mask仍是运行账本，不声称独立重采样了其余所有候选。

## 产物和反馈

反馈完整目录`runs/diagnostics/lt37-runtime-refinement-20261029`，至少保留console.log、exit.json、inspect.log，以及results全部内容。

results含config.json/report.json/artifacts.json，按profile子目录保存30回合执行JSON/NPZ，以及18个网络回合的完整原生预测NPZ；**80份登记文件**，artifacts.json不自登记。没有模型检查点。配置、任务和完整SHA集合先验验收，不能只反馈汇总成功数。

## 异常停止、恢复与禁止覆盖

- Ctrl+C、预算耗尽、非有限值、检查器故障、输出重算或fast等价不一致：记incomplete并退出，保存现场；不跳任务、不续跑、不替换seed、不自行调阈值。
- 无安全控制由原执行器记safety_stop，禁止执行危险首动作。坏初态不能伪装成安全保持。
- 代码或身份漂移先反馈；修复后另建prepared版本和运行目录，重新做短检。不修改旧SHA、不拼接不完整结果。
- 禁止覆盖LT31/34/35/36的权重、配置和结果，以及LT37 v1 prepared、probes-v1/v2、smoke-v1、terminal-saved-state-v1及任何已有正式结果。
- 回退是停止使用新CLI，默认运行路径始终保留；不部署、不自动训练或执行下一阶段。

## Agent后续验收与退出决定

1. 核LT37身份/154源/退出0/completed、8唯一任务/30回合/80登记文件、两条fast精确等价及完整检查点文件集合。
2. combined逐任务与LT36对应分支比较：须恢复860网络固定和教师固定，保留LT36固定/滚动/教师所有旧成功，零碰撞、零余量；丢失不由新增抵消。教师须严格8/8。
3. combined所有MPC.compute观测均<=100ms；报告p50/p95/max与超限比例，不把控制器时延冒称整条感知执行链满足100ms。
4. 消融只给860/1290局部证据，记录保护次数、停留/进度、动作符号翻转、终端干预及拒绝；不选最佳预设代替预先固定combined。
5. 任意旧成功丢失或风险，拒绝候选；教师/860仍失败，停止扩大该具体修正，转控制/监督契约；完整固定成功但滚动失效，转访问状态覆盖和重规划连续性，不加终点损失；仅时延未过则回到性能边界，不减安全标准。
6. 全部通过也只允许准备下一项验证，不能直接解锁LT34训练、125/300批评测或部署。GT终端接管不能声称网络已学会停止；仍须区分原生解码、控制辅助与真实滚动收益。
