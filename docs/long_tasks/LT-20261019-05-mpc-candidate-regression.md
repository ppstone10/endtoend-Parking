# LT-20261019-05：离线MPC候选保留69条双臂回归

## 状态与目的

已运行完成（2026-10-19 16:18–16:20，Agent 代跑并已核验）：原臂逐条复现历史 36/27/6，试验臂 34/25/10，四项门槛全不满足，`decision=stop_no_expansion_or_deployment`；本页命令不再重复执行。这不是训练，不更新正式MPC、模型或部署。

用户已允许调整“控制器必须完全不变”的试验边界，但只用于离线对照。唯一主要变量为求解器返回解的选择：原臂返回原CEM最终精英均值；试验臂保留原搜索，然后用原代价比较最终均值/移位热启动/全零序列，确定性择优。同代价优先原均值。额外评估3条控制序列，不改变种群、迭代、拟合或采样RNG消耗，但不声称总计算量完全相同。

不读取障碍或安全检查器、不修改参考、不使用专家接管、不降低余量。控制器原文件及默认入口不变，实验模块通过专用脚本进程内显式注入并finally恢复。

方法只采用候选比较思想：[iCEM原论文](https://proceedings.mlr.press/v155/pinneri21a.html)、[MBRL-Lib实现](https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/planning/trajectory_opt.py)提供采样记忆/最佳解与均值返回的对照依据；本实现不是完整iCEM，无新依赖或外部代码拷贝。

## 已有短证据与风险

38项相关测试通过，覆盖静止零代价、同RNG搜索、代价不劣、限幅、前进/转向原测试、隔离恢复、冻结/覆盖拒绝和联合门槛。先前85个执行源摘要仍原样匹配，原MPC未改。

真实短预检为S4 #164/S8 #292，双臂各60步，总计算1.58秒；均未到达、无接触。原臂均timeout，试验臂均被分类为oscillation，仍有约3m行驶，不能说已停住或已改善。这是明确风险，完整回归可能失败。短预检只验证入口、输出和注入，不通过正式门槛；不要重跑已有preflight。

低代价不保证安全或到达；零序列候选可能带来停滞、方向切换和成功率回归。正式联合门槛不能仅用碰撞下降替代。

## 冻结来源与参数

根目录：`runs/validation/mpc-candidate-ab-20261019-v1/`。

配置 `config.json` SHA256：`45246333b17039c3a289420721773c13629ec39ea8f17209dccc9bf721686ec8`。

配置冻结5输入、86执行代码文件及Python/NumPy/torch环境，运行前后自动核验。

| 输入 | SHA256 |
|---|---|
| 已知69条protocol.json | `63c3566fa33367853938d26569ec94a4b2e1aa5ebdf64e3606e80c68c34b25d1` |
| 原val.npz | `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963` |
| 原manifest.json | `424fcb68cc8257707a2568c8204bf27f8db2c75c16ddca9a0752557da7ebd077` |
| R0 dep-epoch0007.pt | `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8` |
| 历史原模型完整E5报告 | `87ad68d81e3dfd97853df2aaf01da612f6f70fe8d523ec49949fcf2709723a9c` |

各输入绝对路径见配置；原69协议位于`runs/validation/independent-acceptance/p2-regression/protocol.json`。仅固定该69索引，不补挑、不将其称为独立选型/最终成绩，不用val标签训练。无S9。

固定R0权重与父停止阈值约0.26、原实际E5感知口径、车辆/目标容差、seed0（每回合加索引）、max600/replan10、MPC dt0.1/horizon10及所有原权重/采样参数；碰撞余量0，规划标准不降低。原臂69条应精确复现历史36成功/27碰撞/6振荡及逐条步数/终点误差，任何不一致在试验臂开始前停止。

新增实验模块会令旧诊断的“代码文件全集”摘要多一项；旧85文件内容未变，历史结果仍有效。不要修改旧配置或旧hash绕过；仅使用这份新冻结身份。

## 环境、耗时及资源

工作目录 `D:\program\endtoend Parking`，指定conda Python3.12.13，NumPy2.5.2、torch2.13.0+cpu；CPU单线程，无GPU、无新增安装。

预计3～10分钟：以先前单组69条约1分钟及本次短预检为参考，完整轨迹长度、额外代价评估和CPU负载可能显著影响。建议4GB可用内存、1GB磁盘（非实测峰值）；不要同时训练/修改代码。正式两组属于多轮批量评测，按既定长任务流程由用户运行，不由Agent自动执行。

## PowerShell运行（同一窗口）

### 1. 身份预检，不启动回合

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$trialPython = 'D:\conda\envs\endtoend-parking\python.exe'
$trialRoot = 'D:\program\endtoend Parking\runs\validation\mpc-candidate-ab-20261019-v1'
$trialConfig = Join-Path $trialRoot 'config.json'
$trialHash = '45246333b17039c3a289420721773c13629ec39ea8f17209dccc9bf721686ec8'
if ((Get-FileHash -LiteralPath $trialConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $trialHash) { throw '配置摘要不匹配，停止并反馈' }
& $trialPython -u -B scripts/compare_mpc_solvers.py --config $trialConfig --config-sha256 $trialHash --check-only
if ($LASTEXITCODE -ne 0) { throw '代码/输入/环境校验失败，不继续' }
```

预期`INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`。不要运行prepare或smoke覆盖现有目录。

### 2. 串行原臂69条，再试验臂69条

```powershell
if (Test-Path -LiteralPath (Join-Path $trialRoot 'results')) { throw '已有正式目录，禁止覆盖或重跑，请反馈现有产物' }
$trialLog = Join-Path $trialRoot 'console.log'
if (Test-Path -LiteralPath $trialLog) { throw '已有运行日志，请反馈，不覆盖' }
& $trialPython -u -B scripts/compare_mpc_solvers.py --config $trialConfig --config-sha256 $trialHash 2>&1 | Tee-Object -FilePath $trialLog
if ($LASTEXITCODE -ne 0) { throw '运行失败或中断，保留全部产物并反馈，不自动重试' }
```

进度依次为`ARM original BEGIN`、`original [n/69]`、`ORIGINAL 69 HISTORICAL RESULTS MATCHED`、`ARM candidate BEGIN`、`candidate [n/69]`、`COMPLETE`。原臂不匹配即停止，不把后续试验结果与不可信基线比较。

## 产物、异常与恢复

根目录下`console.log`保存运行日志；新`results/identity.json`保存完整身份，`original.json`与`candidate.json`保存各69条的指标、身份/容差、周期统计和显式offline_solver_trial标识，`report.json`保存completed及联合决策。

异常可能留下`results/status.json`与已完成臂报告。每臂结束才写该臂完整报告，中途回合只在日志中有进度，没有逐回合检查点；硬终止可能无status。脚本使用新目录原子创建拒绝覆盖/并发，没有自动恢复参数。Ctrl+C后保留整个目录，反馈日志和status；由Agent核验后另开新配置/目录确定需重跑的完整臂，不删除/覆盖部分结果、不手改hash、不自动串跑下一阶段。

## 预声明验收与反馈

首先验证身份、两臂各69条、无任务复原失败、相同任务/容差/参数，以及原臂历史逐条复现。

试验臂必须同时满足：成功≥39、碰撞≤24、保留原36成功中的≥33，各场景原成功损失≤1。任一失败则停止本候选，不扩大、不部署、不为了达标改阈值/改分母。振荡、路径长度、耗时和运行时开销也须人工审核，不能只看自动gate。

全部满足仍只获得“可另拟独立300条选型计划”的资格，不代表整体≥70%/≤10%，也不自动开训或推广到生产。仍需后续原300条联合验收；新控制器相应对比基线需公平复测。

完成后回复“MPC双臂69条完成”，提供上述目录；异常直接反馈日志末尾及status。Agent将独立核验逐场景回归、原成功丢失、原/试验控制器身份及联合门槛，再决定是否继续。

## 运行结果（2026-10-19 16:18–16:20，Agent 代跑）

- 只读预检通过；正式运行 151.2 s（原臂约 71 s、试验臂约 80 s）。两臂各 69/69、复原失败为空、`status=completed`，身份与 `offline_solver_trial` 标识正确。
- 原臂在试验臂开始前通过逐条历史比对（36/27/6 精确复现），基线可信；试验臂为 **34 成功 / 25 碰撞 / 10 振荡**，保留原成功 **29/36**，场景毛丢失 S1 1、S4 3、S6 3。
- 四项门槛：成功 ≥39 ✗（34）、碰撞 ≤24 ✗（25）、保留 ≥33 ✗（29）、每场景丢失 ≤1 ✗（S4/S6 各 3）。脚本输出 `gate_passed=false`、`decision=stop_no_expansion_or_deployment`。
- 转移：5 条碰撞转成功（S3 两条、S4 两条、S5 一条），同时 5 条原成功转碰撞（S1 一条、S4 三条、S6 一条），另有 2 条原成功、2 条原碰撞拖成 600 步停滞 —— 与 §“已有短证据与风险”预警的零序列停滞一致。
- 分场景净变化：S3 +2、S5 +1、S6 −3、S1 −1、S4 −1（原成功毛丢失 3），S2/S8 不变。
- 开销：路径长度 10.688→10.595 m，停车时间 20.54→22.04 s，逐回合网络推理均值两臂均为 36.0 ms；试验臂每次求解额外 3 条代价序列，实际计算量更高（未量化）。
- 结论：停止该候选，不扩大、不部署、不改阈值/分母；明细见 `runs/validation/mpc-candidate-ab-20261019-v1/results/summary.md`。若继续，应限制零序列/停滞候选并先解释 S6 整体转向停滞，另交新计划。
