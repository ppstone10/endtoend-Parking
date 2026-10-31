# LT-20261029-41 · 固定原生参考与滚动重预测的收敛诊断

**状态：用户正式执行完整完成，Agent于2026-10-30复核完成；诊断分流10个固定恢复、11个固定仍失败，不授予候选或训练资格。以下命令保留为历史记录，禁止覆盖重跑既有目录。**

## 正式结果与决定

[独立复核](../../runs/verification/lt41-independent-review-20261030/review.md)：164源、原模型/任务、旧LT40证据及新86文件SHA一致，run/inspect退出均0。21新固定+21旧滚动复用，首次完整原生预测与首10步配对通过；固定恢复10个，仍失败11个，实际风险均0。新固定8832步，完整inspect重审新88320/旧126000个H10转移及1281个网络前向；Agent另独立核全部动力学/1281前向/1239窗口，标量几何抽查范围另列。

恢复：2、4、104、135、148、151、156、185、188、207，支持重规划相关分流。仍失败的87/183/186/189/191实际已到发出末点2.5～7.3cm内，但距真正目标1.9～6.1m；其余6个S3固定失败的参考均余量不合格。原生末点仅1/21合格，继续存在坐标与停止选择联合缺口，不能把控制恢复等同网络输出合格。

新固定有1次128.531ms（132/step217，未触发补全）；10个恢复均符合本轮安全/时延分流，整体无实时或生产资格。不按任务拼接固定/滚动成绩，不直接部署全局固定参考；LT40拒绝和训练暂停保持，下一步按三类证据分流，未自动启动新任务。

## 目的与唯一主要变量

LT40剩余21个失败均为运动振荡，且整段全零H10与补全触发为0；20个未进入位置阈值。本轮只检查：取消第一次预测之后的参考替换，能否在同起点完成任务。

固定全部21个失败，不按固定分支结果筛选：`[2,4,87,104,127,132,133,134,135,138,139,148,151,156,183,185,186,188,189,191,207]`。复用已完整核验的LT40对应candidate滚动执行，仅新增**21次network_fixed**；按task_id配对，**21唯一任务、21新固定回合+21复用滚动回合**，不能称42次新执行。

唯一干预：缓存第一次原生发出的全局轨迹，后续固定使用原对象。网络坐标、原生stop、控制器、原时间进度、warm start、索引seed、目标和安全标准均保持。每任务首次完整网络输出及首10个实际控制/状态/窗口须与旧滚动逐值一致。不使用教师、专家回退或轨迹修补。本轮是诊断，不实施固定参考生产策略。规范：[REF-DIAG-001～004](../../spec/fixed_reference_diagnosis.md)。

## 冻结输入、代码、模型与参数

- 工作目录`D:\program\endtoend Parking`；conda环境`D:\conda\envs\endtoend-parking\python.exe`（Python3.12），CPU线程1。
- prepared：`runs/prepared/fixed-reference-diagnosis-20261029-v1/config.json`；SHA256 **b66741d2a9e6d0f92f9fd063696ce0a924b0898510cfa99d3440146e6500a81a**。
- 164源冻结：LT40原162源+`scripts/fixed_reference_analysis.py`与`scripts/diagnose_fixed_reference.py`；旧源码/默认工厂不改，不增加依赖或训练入口。
- LT40 prepared SHA `443ee0150933fc0c2a9f3e9db58f8899ca3677a98106d2c4aaad66208a708115`。绑定LT40全部752文件SHA、report、UTF-16完整inspect日志与两份退出JSON，不从单个成功汇总生成新身份；历史结论仍是拒绝候选。
- 验证缓存及来源：`runs/prepared/zero-motion-retention-20261029-v1/cohort.npz`，SHA `749e903d4c8200e3d2acc046e9826777172a35beeb1953e61e9a54c502cc580c`；来自原300验证任务中的125子集，只取上述21个原索引，禁止使用训练集同编号。任务已多次条件评估，不称独立未见测试。
- 原模型`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA `359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c`，无更新或新训练检查点。
- 首次原生stop0.5/H320/dt0.2；控制器为冻结LT39 ZeroHoldMotionRescueController，MPC H10/dt0.1/80×5、原代价/时间进度/分布保护/安全终端及全零8候选补全、限幅2m/s与1rad/s、600步、seed=原验证集索引保持。
- 原6×3m矩形及0.2m扫掠余量、原车位位置与航向双阈值保持；依赖simulator_ground_truth几何。到达仍按位姿判定，未新增速度归零/停稳保持资格。
- 旧滚动输入：`runs/diagnostics/lt40-zero-motion-retention-20261029/results/<四位索引>-candidate.{json,npz}`及`-predictions.npz`。只读复用，不能复制后改成新执行，不能重跑滚动配对或换seed。
- run总预算7200秒，inspect另7200秒，smoke90秒；预算耗尽即incomplete，无resume或自动重试。

## 已完成的前置检查

- 指定conda单线程下39项定向聚合通过，0.612秒：`-m unittest tests.test_fixed_reference_diagnosis tests.test_zero_motion_retention tests.test_zero_hold_motion_rescue tests.test_zero_hold_motion_study -q`。
- 新2脚本/1测试py_compile、定向Spec检查、下列3段PowerShell命令语法解析及最终身份precheck通过。覆盖完整21分母/86文件、防覆盖/漂移/incomplete、首个H320/logits/输入/前10步伪造拒绝、风险或单次超限不算合格恢复、固定失败/全部恢复/混合分流、PowerShell BOM日志读取及同实际状态下旧续用/新初始化窗口对齐。
- [短检证据](../../runs/verification/fixed-reference-diagnosis-20261029/review.md)：固定验证索引2/133各12步，共24新实际步，9.981秒，零碰撞/余量风险，补全干预0；控制p50/p95/max为9.210/10.200/10.288ms。
- inspect通过10登记文件、240个新选中H10标量转移、24次冻结父求解器重放、全部2次网络完整前向、与LT40首次原生预测和首10实际步一致。
- 短检同时对两条旧滚动记录的全部118次重规划窗口作离线同状态比较并核实际新窗口；#2位置窗口RMS最大0.271m、8次规划几何下降，#133最大0.879m、1次规划几何下降。只是保存状态下的参考观察，不是新控制执行或失败因果结论。
- 短检不重新长审计旧600步控制，`reused_rolling_h10_transitions_reaudited=0`明确报告；正式inspect才完整重放21条旧滚动及全部网络输入。formal=false、分类truncated_unclassified，不能据12步判断任务恢复。
- 完全沿用现有采集/审计与reference_update_stability窗口度量，无新算法或外部依赖，无需重复方法调研。

## 预计耗时与资源

run约3–15分钟，inspect另5–20分钟，合计约10–35分钟；估计依据LT40的86913实际步/2538秒、最多21×600=12600新固定步，以及旧21×600步重放和1260个旧网络输入重算。固定是否早到达、JSON/NPZ压缩、752旧文件哈希和机器负载会改变耗时，估算不确定；两阶段各2小时上限，不能自动续跑。

CPU单线程，无GPU；建议可用内存4GB、额外磁盘2GB。短检24步及两份窗口诊断约1.02MB；正式长控制/窗口记录会增长，旧LT40约2.61GB继续只读保留。控制<=100ms只用于分流合格恢复，不声称整链路实时或最坏耗时保证。

## PowerShell命令：前一步成功后再继续

### 1. 环境、身份及新目录保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt41Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt41Prepared = 'runs/prepared/fixed-reference-diagnosis-20261029-v1'
$lt41Root = 'runs/diagnostics/lt41-fixed-reference-diagnosis-20261029'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt41Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'b66741d2a9e6d0f92f9fd063696ce0a924b0898510cfa99d3440146e6500a81a') { throw 'LT41配置漂移，停止并反馈' }
& $lt41Python -X utf8 scripts/diagnose_fixed_reference.py --phase precheck --prepared $lt41Prepared
if ($LASTEXITCODE -ne 0) { throw '源码/权重/旧产物/数据身份漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt41Root) { throw '已有LT41目录，保留并反馈，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt41Root | Out-Null
```

### 2. 正式21个新固定回合

```powershell
& $lt41Python -u -X utf8 scripts/diagnose_fixed_reference.py --phase run --prepared $lt41Prepared --output "$lt41Root/results" 2>&1 | Tee-Object -FilePath "$lt41Root/console.log"
$lt41RunExit = $LASTEXITCODE
[ordered]@{task='LT41';stage='run';exit_code=$lt41RunExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt41Root/exit.json" -Encoding UTF8
if ($lt41RunExit -ne 0) { throw 'LT41未完整完成，保留现场并反馈，不重试' }
```

按冻结顺序每任务显示FIXED BEGIN及success/collision/margin/steps/class，共21回合。完整后status=completed、formal=true；固定仍失败是有效诊断结果，不因负结果重复运行。exit0不代表候选可用或训练解锁。

### 3. 新固定与旧滚动的完整证据审计

```powershell
& $lt41Python -u -X utf8 scripts/diagnose_fixed_reference.py --phase inspect --prepared $lt41Prepared --output "$lt41Root/results" 2>&1 | Tee-Object -FilePath "$lt41Root/inspect.log"
$lt41InspectExit = $LASTEXITCODE
[ordered]@{task='LT41';stage='inspect';exit_code=$lt41InspectExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt41Root/inspect-exit.json" -Encoding UTF8
if ($lt41InspectExit -ne 0) { throw 'LT41完整证据审计失败，保留并反馈，不重试' }
```

每任务AUDIT累计输出new_H10/reused_H10/predictions。最终status=verified、formal=true、tasks=21、new_fixed_episodes=21、reused_rolling_episodes=21、artifacts=86。

新固定完整H10转移数按实际步数×10；旧滚动重审应为**126000**个H10转移（21×600×10）；全部网络前向应为**1281**次（新21+旧21×60）。核原生停止、坐标、父求解器/补全与终端计数、首10步前缀、逐任务最近位置/联合到达、全窗口对齐与诊断重算。旧审计复用的回合不算本轮新执行。

## 产物、异常、恢复与禁止覆盖

反馈完整目录`runs/diagnostics/lt41-fixed-reference-diagnosis-20261029`：console.log、exit.json、inspect.log、inspect-exit.json及results全部内容。

- results有config/report、21份固定执行JSON、21份执行NPZ、21份完整预测NPZ、21份diagnosis.json，共86登记文件；另artifacts.json不自登记。旧滚动以冻结路径/SHA绑定，不复制、不改写。无新权重/训练检查点。
- diagnosis记录原始参考质量、实际收敛/速度反转/终端尝试/控制时延、首个实际命令分歧、同状态窗口变化与几何；跳变或参考末点不能单独证明根因。
- 超时/振荡/未到达为负结果保留。非有限、初始输出或首10步不符、身份/文件/实际H10/求解器/网络/窗口审计失败、异常安全停止或预算耗尽：incomplete，停止并反馈，不换任务/拼接/续跑。
- Ctrl+C、任一非零退出后先反馈产物。确需修复时先诊断，再新版本prepared与结果目录人工重跑；禁止修改摘要、补文件或覆盖旧结果。空目录也先反馈，不自行删掉绕过保护。
- 禁止覆盖LT31、LT34–40所有权重/配置/产物，以及任何LT41 prepared/短检/正式已有目录。无依赖、迁移或默认工厂变更，停用新入口即可回退。

## Agent后续验收和明确退出

1. 核身份、退出、日志、164源/原模型/旧752产物/配置SHA、21唯一任务/21新固定+21复用滚动、86文件及完整审计计数；完整性通过后才分析结果。
2. 固定参考原生预测和首10步必须严格配对；不成立则无效/incomplete，不解释为模型或重规划失败。
3. 固定成功且零风险、所有控制<=100ms：列入`rolling_related_indices`，支持取消后续参考替换能够恢复该任务，转重规划连续性/访问状态覆盖；不直接发布长期固定参考策略。
4. 固定也失败：列入`fixed_also_failed_indices`，先核原生参考—控制契约和可执行监督。全部21固定也失败，则停止重规划优先假设；不能只凭危险参考、末点不准、安全H10或振荡断言唯一原因。
5. 固定到达但控制超限/风险：单列`fixed_arrival_unqualified_indices`，不算合格恢复。混合结果按任务分流，窗口几何和跳变只辅助定位，不用观察量当执行反事实。
6. 本轮无论结果如何，LT40拒绝结论及training_unlocked/candidate_qualified/production_qualified=false保持。不训练、不加轮数/权重/速度/seed，不自动运行下一长任务。
