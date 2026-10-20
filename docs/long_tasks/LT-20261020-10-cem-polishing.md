# LT-20261020-10：原代价CEM精修六源对照

## 状态与目标

已运行完成（2026-10-20 11:33–11:36，Agent 代跑并已核验）：六源×四分支完成，精修教师 6/6 严格成功，但精修纯网络只保留 1 条原成功、碰撞 4→5，`gate_passed=false`、`decision=stop_no_expansion`；本页命令不再重复执行。不训练；默认MPC、模型和主300条成绩不变。

LT09两个首危险事件的参考/实际窗口安全而第一步执行越界。固定相同窗口、状态、限幅和原代价，精修可得到更低cost且事后安全的序列，说明有限CEM求解误差参与失败。但纯网络短检仍两条碰撞，不是唯一病因。本次只回答：精修能否使S4持续安全教师全程可执行，同时保留原网络成功？六个事后机制样本不代表总体成绩。

## 算法变量与成本

独立CEMPolishedMPC完整执行原CEM，再从其均值精修**完全相同的原cost**：双精度、`limits*tanh(z)`、L-BFGS最多60迭代/80次closure硬上限、history10、strong_wolfe、梯度容差1e-9/变化容差1e-12。原NumPy cost最终复核，有限且严格更优才接受，平局保留原解，非有限回退，意外异常停止。

不改参考时序、种群80/迭代5、权重、车辆、限幅、余量、阈值和到达标准；不加环境输入、安全盾、几何筛选或全零候选。原CEM每步随机消耗保持；精修改变真实状态及后续热启动/参考属实验结果。几何仅事后记录，不保证安全或全局最优。

每步额外最多80次可微cost/梯度评估及前后NumPy复核，**不是等算力对照**。保存每步原序列、前后cost、接受、次数和停止原因。只用现有torch，不安装SciPy/SCS或复制外部代码。[PyTorch LBFGS官方说明](https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS)用于接口选择（网页2.14，本机2.13接口已核对）；[SCS官方MPC示例](https://www.cvxgrp.org/scs/examples/python/mpc.html)用于区分目标与约束，不引入其库或障碍约束。

## 固定来源与四分支

| train索引 | 上下文步 | 用途 | 每分支剩余步数上限 |
|---:|---:|---|---:|
| 1298 | 210 | 恢复 | 390 |
| 1455 | 180 | 恢复 | 420 |
| 1559 | 310 | 恢复 | 290 |
| 1441 | 220 | 恢复 | 380 |
| 1334 | 60 | 原网络成功对照 | 540 |
| 1680 | 110 | 原网络成功对照 | 490 |

每源完整复现LT09原网络源回合，复制同车辆/MPC热启动与RNG/感知快照，依序运行original_network、original_expert_continuous、polished_network、polished_expert_continuous。原两臂完整状态/命令须在1e-9内复现LT09，原教师参考逐项相同。精修两臂首次实际窗口与未精修CEM序列匹配对应原臂。身份或复现不符即停，不换源。

持续教师每次仍消费正常感知/网络旁路推理，但选同状态既有专家输出；不是部署策略。max600/replan10/dt0.1/horizon10、seed=train索引；6×3m车辆、规划0.2m/接触0余量、原到达容差不变。实际E5 target_override=False，初态BEV逐源匹配train，不按checkpoint元数据切换感知模式。不读取val/selection轨迹作优化或训练。

## 冻结身份

目录`runs/diagnostics/cem-polishing-ab-20261020-v1/`，config.json SHA256：

`b66fcb7adda57629cb9454324d13c946f6aa8ef243e38dafaf7b5d57fcec96d1`

79输入、97执行源及运行环境冻结：LT09原47输入、LT09 config/report、六源各自report/NPZ/教师参考/两份控制审计。原94源未改。task_id/绝对路径/摘要见config，禁止手改、重新prepare或smoke。

模型SHA `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`；train SHA `8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f`；LT09 config SHA `dcf2dd9be560a0c97c406ad0c2ba81bcc7d66177da555a68fbbb5cc8a49f2917`。

## 已验证及边界

72项相关测试通过：原cost/梯度、禁用动作/RNG复现、硬预算/限幅/严格接受/回退/异常、上下文隔离/实际首步、联合门槛、冻结漂移与覆盖拒绝；包含原MPC、事件与连续恢复回归。

真实短检23.945秒，两源四分支各最多20步，原源/原两臂前缀/教师参考复现。教师原/精修均20步无接触无余量损失但未到达，未覆盖LT09相对46/48事件。纯网络原臂15/16步接触，精修14/17步接触，**不能称改善，#1298更早接触**。

另在LT09记录的原危险状态/窗口独立探针：#1298 cost0.144430→0.071183（69次评估），#1455 0.163950→0.062504（71次）；两个首步/完整1秒rollout事后均规划安全。不是强制新闭环回到这些状态，不是全程恢复证明。142步审计/NPZ摘要、命令/rollout/实际首步/原cost/接受/预算复核通过，累计4924次closure。详见[预检复核](../../runs/diagnostics/cem-polishing-ab-20261020-v1/preflight/agent-audit.md)。预检永远gate=false，不再重跑。

## 环境与资源

仅`D:\conda\envs\endtoend-parking\python.exe`，Python3.12.13/NumPy2.5.2/torch2.13.0+cpu，CPU单线程。运行中勿编辑执行源、并行训练或更换环境。

预计5～30分钟，仅预算参考：精修增加CPU成本，专家搜索耗时不定。建议4GB可用内存/1GB磁盘，非实测峰值。超时估计但console仍推进时不要另开进程；持续无进度或报错保留现场反馈。

## PowerShell：同一窗口依次执行

### 1. 只读预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$polishPython = 'D:\conda\envs\endtoend-parking\python.exe'
$polishRoot = 'D:\program\endtoend Parking\runs\diagnostics\cem-polishing-ab-20261020-v1'
$polishConfig = Join-Path $polishRoot 'config.json'
$polishHash = 'b66fcb7adda57629cb9454324d13c946f6aa8ef243e38dafaf7b5d57fcec96d1'
if ((Get-FileHash -LiteralPath $polishConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $polishHash) { throw '配置漂移，停止反馈' }
& $polishPython -u -B scripts/compare_cem_polishing.py --config $polishConfig --config-sha256 $polishHash --check-only
if ($LASTEXITCODE -ne 0) { throw '身份预检失败，不继续' }
```

应显示`INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`。

### 2. 正式六源四分支

```powershell
if (Test-Path -LiteralPath (Join-Path $polishRoot 'results')) { throw '已有正式输出，禁止覆盖或重跑，先反馈' }
$polishLog = Join-Path $polishRoot 'console.log'
if (Test-Path -LiteralPath $polishLog) { throw '已有日志，禁止覆盖，先反馈' }
& $polishPython -u -B scripts/compare_cem_polishing.py --config $polishConfig --config-sha256 $polishHash 2>&1 | Tee-Object -FilePath $polishLog
if ($LASTEXITCODE -ne 0) { throw '失败/中断，保留所有产物，不自动重试' }
```

每源SOURCE BEGIN→四分支BEGIN budget=...→COMPLETE smoke=False。教师/精修分支可能较慢。完成后results/report.json应为status=completed、smoke=false、tasks六项，四个variants各sources=6。gate_passed可真可假，完成不等于通过。

逐源保存NPZ、四份audit、两份教师references及摘要，根目录identity/report。失败保留status和已完成来源，无resume/重试，不删输出/换hash/补挑任务。不导出训练schema、不覆盖旧成功轨迹。

## 回传判断

原源/原臂复现为前提。仅当精修教师**六源全程严格成功**（四恢复+两对照，到达、无接触、零规划余量损失），且精修纯网络保留原两成功、碰撞数不超过原臂，才允许另写69条回归长任务计划；不自动执行。该门槛不是总体统计显著性。

未通过即停止扩大，分析失败的参考/窗口/执行/规划原因，不直接扫权重/预算或训练。即使通过，新控制器下教师可执行不等于原MPC标签可执行，主300条≥70%/≤10%仍须另验。

本任务已完成，不再重跑或重新prepare；历史回传口令为“CEM精修六源验证完成”。

## 独立复核补充

[正式独立复核](../../runs/diagnostics/cem-polishing-ab-20261020-v1/results/agent-audit.md)重新检查30回合2770转移、1475控制记录和109条教师参考，通过身份/完整性/原臂复现及汇总重算。联合门槛确实失败，不再扩大。#1334精修相对56步窗口已危险、63步首失余量；同一63步状态/窗口下原CEM首步安全，精修cost0.096479→0.055835却首步不安全。存在参考风险和局部降代价取舍，但网络反馈的独立作用尚未以固定参考隔离，不能把下方“压得更靠边”读成全部失败的已证单一原因。精修教师可执行不等于原MPC标签可执行，不直接训练。

## 运行结果（2026-10-20 11:33–11:36，Agent 代跑）

- 只读预检通过；正式运行 162.1 s，6/6 源四分支完成、`status=completed`、`smoke=false`、`tasks=6`、四分支各 `sources=6`；6/6 源前缀与起始 BEV 复现，6/6 的原两臂在 1e-9 内复现 LT09、教师参考逐项相同。
- 分支汇总：original_network 2 成功/4 碰撞/1 严格；original_expert_continuous 5/0/4；**polished_network 1/5/1**；**polished_expert_continuous 6/0/6**。
- 精修教师：四恢复源与两对照全部严格成功（#1298 57 步 mv0、#1455 57 步 mv0，修好了 LT09 的“到达但 mv3”与“replan@50 起始位姿冲突”）。精修网络：#1334 从成功 62 步变成碰撞 71 步（mv 8），#1680 仍严格成功；原两成功只保留 1 条、碰撞 4→5。
- 判定：`network_success_controls_retained=1`、`gate_passed=false`、`decision=stop_no_expansion`。同一求解器改动对教师链路完全有利、对纯网络链路有害，指向“原代价无余量项、精修会把贴边参考压得更靠边”；不写 69 条计划、不训练、不部署。明细见 `runs/diagnostics/cem-polishing-ab-20261020-v1/results/summary.md`。
