# LT-20261020-09：S4连续恢复教师可执行验证

## 状态与目标

已运行完成（2026-10-20 10:22，Agent 代跑并已核验）：六源完成，4 个恢复源中 2 个严格恢复（#1559、#1441）、2 个失败（#1298 到达但 mv 3；#1455 于 replan@50 起始位姿冲突提前结束），两条成功对照均保持成功，判定 `separate_mechanism_review_no_training`；本页命令不再重复执行。只作训练前置可执行审查，不训练、不生成训练集、不部署，不把教师表现当学习器表现。

LT08专家一次虽救回#1455，却使两个原成功对照退化；六个S4专家首10步均安全，后续网络反馈才出现危险。这里的唯一执行变量是“正常网络更新”与“持续专家重规划”，原MPC/热启动/RNG/车辆/容差/预算均不变。持续专家仍消费正常感知/网络推理作旁路记录，但不选其输出；每次参考都来自同状态既有专家。该教师仅离线可用，不作为部署接管策略。

## 另一路S8已停止，不混入正式任务

本轮先实现并短检了不读障碍物的几何航向代价FootprintYawMPC：四角误差减去中心平移后的航向项为`2r²(1−cosΔyaw)`，`r²=(L²+W²)/4`，按原位置权重/终点权重计价。原搜索预算与返回规则不变，不加安全盾。

[Autoware官方算法](https://autowarefoundation.github.io/autoware_universe/main/control/autoware_mpc_lateral_controller/model_predictive_control_algorithm/)区分位置/航向代价；[acados官方文档](https://docs.acados.org/python_interface/index.html)说明非线性输出残差平方代价。这些仅支持结构，角点公式是本项目独立推导，不是推荐参数或效果证明。没有引入这些库或复制第三方代码，无新依赖/许可证负担。

预声明#2348短检中，新代价的正常参考余量损失仍4步、专家参考仍2步，与原MPC相同，未解决既知反例。因此停止此候选，不跑其余三S8源、不扫gain/权重，不把缺失结果算通过。原MPC不改。S8后续需另定方案，本计划**只运行S4**。

## 固定来源与身份

| 顺序 | train索引 | 上下文步 | 用途 | 两分支各自最大剩余步数 |
|---:|---:|---:|---|---:|
| 1 | 1298 | 210 | 规划回退恢复 | 390 |
| 2 | 1455 | 180 | 规划回退恢复 | 420 |
| 3 | 1559 | 310 | 规划回退恢复 | 290 |
| 4 | 1441 | 220 | 接触回退恢复 | 380 |
| 5 | 1334 | 60 | 原成功对照（原轨迹曾失规划余量） | 540 |
| 6 | 1680 | 110 | 原成功对照 | 490 |

全部为LT08已有训练任务，属于事后机制样本，不是独立总体评测；不得按新结果补选。源回合完整复现LT08，再从同车辆状态/MPC热启动与RNG/感知快照分叉。正常网络完整后缀须先在1e-9内复现，持续教师首段还须与LT08专家一次前10步相符；若专家规划早期失败，记录`matched=null`与实际比较步数/失败原因，不伪造通过或跳过来源。

只使用最终目录 `runs/diagnostics/recovery-compatibility-20261020-v3/`。配置 `config.json` SHA256：

`dcf2dd9be560a0c97c406ad0c2ba81bcc7d66177da555a68fbbb5cc8a49f2917`

冻结47输入、94执行代码文件及环境：LT08原31输入、LT08配置/正式报告、六源各自LT08 report/NPZ共12份，以及S8停止证据v1 config/preflight报告两份。旧91执行源逐项未变。精确task_id/绝对路径/摘要均在新config中，不手改。

关键身份：R0模型`d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`；原2400任务train `8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f`；LT08 config `14b46ab57beb0ddc61da863572a6891da0a1181c20a32ec867a2a4d224a4659f`；LT08正式report `31c5fa4ec5eb3678a21c31569ae46714a83861accca75e2909bb0efba6255fc5`。

原实际E5感知/父阈值约0.26保持；初态BEV逐源匹配原train。seed0+train索引，max600、replan10、dt0.1/horizon10、6×3m车辆，接触余量0/规划余量0.2m不降低，原到达容差不放宽。保留两个原成功轨迹及输入摘要，不导出专家监督来覆盖它们。

### 版本与禁止重跑

v1的53输入/94源配置与6.741秒两源短检保留为S8停止证据，旧SHA `0a59b605bdadc0f146ab8d1103ed0c6c04126e2f15bc8f3c194eedf58dbcd5f8`，不得覆盖；停止报告SHA `5b9c9d63543cf205ad912fe9bd05f14dd802a99d5f51f3d13c68124668a02003`。v2仅完成范围收敛准备，随后发现早期专家规划失败需要记账而非一律中断，补充该测试后使用新v3；v2不执行。

新执行代码全集改变，旧入口可能拒绝旧清单重跑，这是保护机制；不要改旧hash绕过。仅使用本计划v3，不再次prepare或smoke。

## 已有验证与边界

53项聚合测试通过，覆盖角点公式/周期性、stage和terminal新增项、gain0精确复现原MPC命令和RNG、正gain随机流消耗一致、热启动/进度复制隔离、持续教师跨多次重规划、正常完整后缀、失败记账、正式范围排除S8、身份及覆盖拒绝。

已有v1真实短检：S4 #1298源225步精确复现，从step210正常分支15步后碰撞，持续专家20步/两次重规划内无接触无余量损失但未到达；不是全程恢复证明。S8 #2348四分支各10步未到达无接触，新代价余量未改善。两源总6.741秒。v3仅调整正式范围及早期失败记账，控制器/执行核心与该短检相同；复用物理预检证据，定向测试覆盖新增逻辑，不能把v1说成v3全程运行。

## 环境和资源

工作目录 `D:\program\endtoend Parking`，Python唯一使用 `D:\conda\envs\endtoend-parking\python.exe`，3.12.13/NumPy2.5.2/torch2.13.0+cpu，CPU单线程，无新安装/GPU。运行期间不编辑执行代码/并行训练。

预计1～10分钟：LT08一次替换11源40.2秒，本轮六源但每10步专家重规划，搜索耗时不稳定；不能按短检线性承诺。建议4GB可用内存、1GB磁盘，非实测峰值。按正式批量长任务处理，由用户运行。

## PowerShell（同一窗口，依次执行）

### 1. 只读身份预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$recoveryPython = 'D:\conda\envs\endtoend-parking\python.exe'
$recoveryRoot = 'D:\program\endtoend Parking\runs\diagnostics\recovery-compatibility-20261020-v3'
$recoveryConfig = Join-Path $recoveryRoot 'config.json'
$recoveryHash = 'dcf2dd9be560a0c97c406ad0c2ba81bcc7d66177da555a68fbbb5cc8a49f2917'
if ((Get-FileHash -LiteralPath $recoveryConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $recoveryHash) { throw '配置摘要变化，停止反馈' }
& $recoveryPython -u -B scripts/validate_recovery_compatibility.py --config $recoveryConfig --config-sha256 $recoveryHash --check-only
if ($LASTEXITCODE -ne 0) { throw '输入/源码/环境校验失败，不继续' }
```

预期 `INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`。

### 2. 正式S4六源

```powershell
if (Test-Path -LiteralPath (Join-Path $recoveryRoot 'results')) { throw '已有正式输出，禁止覆盖或重跑，先反馈' }
$recoveryLog = Join-Path $recoveryRoot 'console.log'
if (Test-Path -LiteralPath $recoveryLog) { throw '已有日志，禁止覆盖，先反馈' }
& $recoveryPython -u -B scripts/validate_recovery_compatibility.py --config $recoveryConfig --config-sha256 $recoveryHash 2>&1 | Tee-Object -FilePath $recoveryLog
if ($LASTEXITCODE -ne 0) { throw '失败/中断，保留所有产物，不自动重试' }
```

每源进度`S4 COLLECT BEGIN`→`original_network BEGIN`→`original_expert_continuous BEGIN`→`COMPLETE cohort=S4`。教师分支可能多次搜索，单行停留一段时间不等于死循环。整批完成以results/report.json `status=completed`、`smoke=false`、tasks=6、s4_recovery_sources=4、s4_success_controls=2为准；**本次不应有S8或footprint_yaw分支**。

## 产物、停止与回传

根目录console.log；results/identity.json完整身份；逐源task-NNNN/report.json记录源/正常后缀、教师首段复现、到达/碰撞/余量/规划失败与专家尝试/应用次数；trajectories.npz保存原源/两分支状态命令和上下文；两个*-audit.json记录实际窗口、所选控制、rollout/原代价及几何；teacher-references.json保存每次实际教师状态、全局参考与dt。所有数组/审计/教师参考文件均有SHA256。

教师参考只是诊断产物，缺BEV/训练schema和入库流程，不能直接交Trainer；原train/模型没有被写入。正常回合/后缀漂移、初态BEV不符、身份变化立即停。专家途中无解记录为失败，继续保留该来源，不以新状态/任务替换。

异常status.json和已完成源保留；Ctrl+C保留产物，强杀可能没有status.json。无自动resume/retry，不删除results、不换源/预算/容差/余量，不修改hash，反馈后另定剩余任务新目录。

完成后回复“S4连续恢复验证完成”；异常时附控制台末尾和status.json。

## Agent回传后的判断

先核验47输入/94源、六任务分母/文件摘要、原源和正常完整后缀、真实推进/审计/教师参考与应用次数，再分析四个事件的严格恢复和两个对照的教师结果。

只有四恢复源均完整到达、零接触/零规划余量损失，才将这些状态序列列为后续数据设计候选；否则保留失败并明确短板，不降低门槛。两个成功源的原记录保持不变，教师对照结果单列，不能称学生成功保留通过；后续训练仍需原样本池与独立回归验证。无论结果都不自动训练、扩大或部署，不以六个事后机制源声称70%/10%达标。

## 运行结果（2026-10-20 10:22，Agent 代跑）

- 只读预检通过；正式运行 25.4 s，`status=completed`、`smoke=false`、`tasks=6`、`s4_recovery_sources=4`、`s4_success_controls=2`，无 S8/footprint_yaw 分支；6/6 起始 BEV 与正常后缀复现，6/6 教师首段与 LT08 前 10 步相符。
- 恢复源：#1559 严格恢复（42 步、mv 0）、#1441 严格恢复（50 步、mv 0）；#1298 到达但 mv 3（非严格）；#1455 教师在第 50 步重规划因“起始位姿与障碍物冲突”提前结束（6 次尝试/5 次应用）。`s4_teacher_strict_recoveries=2`。
- 成功对照：#1334 教师 125 步 mv 0（原网络 62 步 mv 5，余量反而更好）、#1680 教师 181 步 mv 0（原网络 101 步 mv 0）；两条都保持成功，`s4_control_teacher_successes=2`，原轨迹未被改写。
- 结论：`decision=separate_mechanism_review_no_training`。持续教师能让 2/4 个状态严格恢复且不破坏成功对照，但“到达≠带余量”和“中途起始位姿冲突”两类失败使四源全通过的门槛未满足；不列入数据设计候选、不训练、不扩大。明细见 `runs/diagnostics/recovery-compatibility-20261020-v3/results/summary.md`。
