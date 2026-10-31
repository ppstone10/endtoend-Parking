# LT35 · 固定八训练任务的教师／网络固定参考／网络重规划对照

状态：**已完成并独立复核**。教师8/8到达、0碰撞；网络固定参考7/8到达、1碰撞；正常重规划4/8到达、4碰撞。重规划丢失固定成功3例，另1例固定网络参考执行已失败。证据见[LT35独立复核](../../runs/verification/lt35-independent-review-20261028/review.md)。以下命令保留追溯，不重跑；未训练、不改LT34门槛、不自动扩大125/300任务。

规范：[EMIT-DIAG-001–004](../../spec/emitted_execution_diagnosis.md)。任务T-213216。

正式运行11.179秒、exit0/completed，50产物及146代码身份通过；5.455秒离线独立重算2185网络控制步/全部H10并从输入逐值复现122次完整预测，八组初始输出/前10步配对一致。余量侵入分别为教师19、网络固定43、滚动101控制步，不能混同实际碰撞。#1720教师/固定均严格安全但滚动碰撞，是重规划相关回退的清晰对照；#1290教师严格通过但首次网络参考已危险，说明也有独立的初始输出/控制兼容性缺口。尚未确定更细根因，不自动加训练剂量。

## 目的与唯一主要变量

在同一批八个任务、同一父网络与原MPC下，分开观察固定网络首次参考与正常重规划的结果。唯一主动改变是**是否重新感知并更新网络参考**；教师回合作为已核验的任务执行对照，直接复用LT34原始产物，不重复跑教师。

正式分母为**8个独立训练任务**。共三分支24个回合结果，其中8教师已存在，本次新增16个网络回合。32监督帧、115教师沿程输入不扩充任务分母。300验证任务与后续125任务集合均不参与本次执行或训练。

## 冻结输入与固定参数

- 工作目录`D:\program\endtoend Parking`，Python只用`D:\conda\envs\endtoend-parking\python.exe`。原conda Python3.12.13、torch2.13.0+cpu、numpy2.5.2，CPU单线程。
- 配置`runs/prepared/emitted-execution-diagnosis-20261028-v1/config.json`，SHA256 **`0b5fe08e72e8d696493ba92c90e9cd3392cfa1d46a80b2625c2c5af0075c76dd`**，冻结146执行源码。工作树未提交，以文件SHA为版本；任何漂移停止，不编辑hash绕过。
- 原LT34配置SHA `86cbf4c55664de4cd51a9c4098d4edf2b7ec26a7302a269c654244917d3cad6f`保持；precheck仍验证其144源码/7输入、八源缓存与manifest。本次只新增scripts与tests，不修改旧冻结源码。
- 固定训练索引顺序`[0,433,860,1290,1720,1975,2315,2]`，七场景/正倒车；task_id与LT34逐项相同，不能删掉860/2315或按结果换样本。
- 网络固定为`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA `359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c`。无更新、无优化器、无模型替换或阈值校准。
- 教师来自`runs/training/lt34-emitted-joint-20261028/teacher`；artifacts.json SHA `7a0f2180f8a903c4d27c6104cc590b87f31c579512ed45492c5c9a102da9a8d8`，10产物逐项验证。八教师到达8/8、碰撞0/8、余量侵入任务2/8及19步，保留原结论；允许诊断对照不等于授权拟合。
- 原生sensor→network→mpc_vehicle、safety_mode=none，无专家替换/接管/安全前缀；net-v1/image/per_view_v2/semantic/H320/dt0.2/stop0.5。MPC dt0.1/H10、seed=原索引（control_seed0）、每10控制步重规划、最多600步、车辆6×3m及各车位原tol_pos/tol_yaw保持。
- 固定分支只调用一次公开model.predict，按原生stop0.5截断后的参考转成全局坐标，整个回合原样复用；不随当前车位姿重新变换、不按教师长度截断。滚动分支按原周期重新感知/预测。
- 每次公开predict只解码一次，用只读decoder hook保存该次完整H320坐标/logits及输入；不混用别的轨迹或停止头。两个网络分支首次输入/完整预测/logits/发出轨迹及重规划前共享最多10控制步必须一致，否则停止归因。

## 已完成短检与预计资源

- 指定conda下20项聚合测试通过（2.770秒）：`python -m unittest tests.test_emitted_execution_diagnosis tests.test_emitted_output_joint tests.test_emitted_output_study -q`。覆盖固定参考缓存、滚动重新预测、公开输出及参数不变、异常hook移除、共享前缀/缺失分母拒绝、分类与余量/碰撞分开；先有缺实现导入失败证据。
- 单任务#0[短检报告](../../runs/diagnostics/emitted-execution-smoke-20261028-v1/report.json)1.763秒：复用教师80步；网络固定80步、网络滚动102步，两者均到达/无碰撞/无余量侵入，首次原生输出和前10步一致。182个控制步的选中H10首步与实际推进逐步复现，8份产物inspect通过。短检formal=false，不是8例正式成绩。
- 预计正式16网络回合约1–5分钟。依据首任务两分支182步及完整记录1.763秒，最坏16×600=9600控制步折算并为不同场景几何、压缩与机器负载留余量；尚未测全八，不保证线性。正式批量按项目规则由用户执行，硬上限1800秒，耗尽即incomplete，不自动加时/续跑。
- CPU，无GPU；建议空闲内存4GB、磁盘2GB。原始BEV/全H320预测、H10及几何审计会增加压缩与存储成本。两分支和任务均顺序运行，不后台并发。

## PowerShell命令：依次执行

### 1. 环境、身份与输出保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt35Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt35Prepared = 'runs/prepared/emitted-execution-diagnosis-20261028-v1'
$lt35Root = 'runs/diagnostics/lt35-emitted-execution-20261028'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt35Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b5fe08e72e8d696493ba92c90e9cd3392cfa1d46a80b2625c2c5af0075c76dd') { throw 'LT35配置身份漂移' }
& $lt35Python -X utf8 scripts/diagnose_emitted_execution.py --phase precheck --prepared $lt35Prepared
if ($LASTEXITCODE -ne 0) { throw '身份/环境/教师证据不匹配，停止' }
if (Test-Path -LiteralPath $lt35Root) { throw '正式目录已存在；保留并反馈，不覆盖或自行续跑' }
New-Item -ItemType Directory -Path $lt35Root | Out-Null
```

### 2. 正式八任务配对诊断

```powershell
& $lt35Python -u -X utf8 scripts/diagnose_emitted_execution.py --phase run --prepared $lt35Prepared --output "$lt35Root/results" 2>&1 | Tee-Object -FilePath "$lt35Root/console.log"
$lt35Exit = $LASTEXITCODE
[ordered]@{task='LT35';stage='run';exit_code=$lt35Exit} | ConvertTo-Json | Set-Content -LiteralPath "$lt35Root/exit.json" -Encoding UTF8
if ($lt35Exit -ne 0) { throw 'LT35不完整，停止并反馈；不要自动重试' }
```

正常输出每分支`BEGIN`及success/collision/margin_steps/controls，共16条分支完成记录。网络发生碰撞或未到达是有效实验结果，不是需要重试的程序异常。全部8任务完成后report.status=completed，仍不授予候选/生产资格。

### 3. 完整性核验

```powershell
& $lt35Python -X utf8 scripts/diagnose_emitted_execution.py --phase inspect --prepared $lt35Prepared --output "$lt35Root/results" 2>&1 | Tee-Object -FilePath "$lt35Root/inspect.log"
if ($LASTEXITCODE -ne 0) { throw 'LT35完整性/配对核验失败，停止并反馈' }
```

## 产物与反馈

反馈`runs/diagnostics/lt35-emitted-execution-20261028`路径即可；保留：

- console.log、exit.json、inspect.log；
- results/config.json、report.json、artifacts.json；
- 16组`<原索引>-network_fixed/rolling.json`、同名执行npz、`-predictions.npz`（48文件）。连同config/report共50份登记产物；artifacts.json不自登记。

执行npz记录初态、实际前后状态、MPC窗口、选中控制序列/H10预测、参考和发出输入；predictions.npz记录每次实际解码的全H320坐标/logits/BEV/目标/运动状态及原生长度。教师原始8套npz不复制或覆盖，通过冻结来源引用复用。

## 异常、恢复与禁止覆盖

Ctrl+C、超时、配对前缀不一致或身份漂移均保留incomplete并停止；无resume、跳任务、自动重试、自动扩样或自动训练。技术异常先反馈Agent诊断，确需重跑时再准备新身份/新目录，不把异常变成改样本或改标准的理由。

禁止覆盖LT31模型、LT34数据/teacher/config、旧短检、新prepared或任何已有正式目录。两个网络分支均不使用专家回退，也不把缓存教师标成网络输出。原LT34训练暂停保持。

## Agent后续验收与决定

先核冻结代码/输入/退出状态、50产物完整性、8唯一task_id、同一父权重/初态/控制seed；重算实际运动学、真实碰撞与0.2m扫掠、原终点容差、原生首停/参考、H10首步与前10步配对，再讨论分支差异。

逐任务分别报告到达、碰撞、余量侵入任务数/步数，不把末点静态匹配当到达，也不把余量当碰撞。按以下模式定位环节：

| 同任务观察 | 有限结论与下一方向 |
|---|---|
| 教师本身未到达或碰撞 | 先查教师执行能力与控制契约 |
| 教师到达，网络固定与滚动均失败 | 固定网络参考执行已暴露问题；查参考形状/原生停止/几何与MPC兼容性，不能只归因重规划 |
| 网络固定到达，滚动失败 | 重规划相关缺口；查新输入、参考变化和跨周期连续性，不直接断言唯一是状态覆盖 |
| 网络固定失败，滚动到达 | 重规划能恢复固定参考失败，不能笼统称重规划有害 |
| 三者均到达 | 比较实际风险和逐步行为，不能以到达掩盖风险，也不能由8例宣布全数据解决 |

单任务可同时有多种风险，模式计数不是互斥根因的因果占比。根据八例证据决定下一项有限诊断；不自动运行125/300任务、不调整原损失、控制器或验收门槛。
