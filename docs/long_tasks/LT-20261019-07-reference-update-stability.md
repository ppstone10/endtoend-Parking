# LT-20261019-07：训练域参考更新稳定性与实际可执行性验证

## 状态与目标

已运行完成（2026-10-19 19:31–19:35，Agent 代跑并已核验）：24/24 源完成，配对 19、缺失 5，覆盖条件满足，判定 `mechanism_audit_ready_no_training`；本页命令不再重复执行。不是训练，不生成回灌标签，不部署。固定原R0和原MPC，回答：网络更新是否把同状态旧安全窗口变危险；一次更新对实际后续到达/碰撞/余量有何影响。不得把单次延迟当作生产策略。

沿用项目已有快照、精确续执行和矩形连续扫掠，不引入新算法。唯一分叉变量是第一次重规划返回新参考，还是仍正常感知/推理消耗同次RNG后丢弃此次输出、返回已有旧参考对象（保持控制进度）；随后两分支均正常重规划。相同起点、MPC热启动/RNG/感知管道，使用原控制器，不使用此前未通过筛选的两个求解器候选。

## 固定来源与身份

根目录 `runs/diagnostics/reference-update-stability-20261019-v1/`，配置 `config.json` SHA256：

`3474f667b06253515926546f4071087cf754c434cb505a61f6221d6e3175c211`

冻结7输入、89执行代码文件、Python/NumPy/torch环境。先逐项验证旧LT04的85个代码摘要和5个输入未变，再建立新全集；不要改旧hash绕过检查。新增实验核心会使旧入口的代码全集检查拒绝重跑，这是保护机制。

| 输入 | SHA256 |
|---|---|
| 原train.npz（2400原任务） | `8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f` |
| 原manifest.json | `424fcb68cc8257707a2568c8204bf27f8db2c75c16ddca9a0752557da7ebd077` |
| val.npz（仅身份排除） | `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963` |
| selection.npz（仅身份排除） | `797e9e5d0d02576c45df65c74f142cc05e83770eb6bbfcc28be560dbc722d51d` |
| R0 dep-epoch0007.pt | `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8` |
| LT04 config.json | `f20a17905febf3db49922adc8960d165521e4240b21ce598263ca77d7122d37f` |
| LT04 results/report.json | `a8790b6d549d40517bfbc44b06fbf46e17891ae054ec3965ed3752a20633a3dc` |

固定24条唯一原训练任务，按机动/噪声轮询（S4/S8 seed20261019，S1/S2 seed20261020），不读验证结果挑训练来源，不涉及S9。实际task_id清单在配置中逐项冻结。

| 场景 | train索引，按执行顺序 |
|---|---|
| S4（12） | 1376,1298,1394,1455,1451,1675,1559,1711,1581,1334,1441,1680 |
| S8（6） | 2348,2330,2332,2364,2317,2356 |
| S1（3） | 338,301,345 |
| S2（3） | 703,731,780 |

前12个已知失败源保留用于机制复查；扩展来源预先固定，绝不因缺成功对照补挑。完整源结果决定success_control分类，不把成功与严格余量通过混为一谈，S1/S2不替代S4/S8同场景对照。

原E5实际感知、父停止阈值约0.26、车辆6×3m、控制seed0+train索引、max600、replan10、MPC dt0.1/horizon10全部保持。接触余量0，规划余量0.2m不降低；checkpoint目标通道元数据与实际pipeline的既有不一致不在本轮切换，每条初态BEV必须逐元素等于归档train输入。

## 观察、执行与判定

每次更新在同一状态、同一MPC控制时间网格比较“旧参考正常延续窗口”和“新参考初始化窗口”：xy RMS/max（m）、最短yaw差RMS/max（rad）、窗口内部/状态连接的接触及规划安全；初次没有旧参考明确排除。报告旧安全→新危险次数，不用任意RMS阈值宣称稳定。变化小不代表安全，旧新同时危险也不是稳定通过。

成功源选择接近中段的安全非零重规划点；失败源仅选择首次真实余量损失前最近安全非零重规划点，缺失保留。正常分支必须在1e-9内完整复现原后缀（含长度）才允许延迟分支；与LT04重叠的12个源还须复现旧完整源摘要。每分支预算600−上下文步数，记录到达、碰撞、真实余量损失、失败元数据与全程状态/命令。不可用短预测安全替代实际执行。

程序完成要求24个源账本齐全且适用的完整后缀/历史复现通过。机制覆盖要求配对≥8、S4首次危险前失败配对≥4且成功对照≥2、S1/S2成功对照各≥2。不足则`insufficient_coverage_no_training`，不能事后降低门槛。达到覆盖只返回`mechanism_audit_ready_no_training`，也不代表稳定、可执行或达到70%/10%。S8缺同场景成功对照必须单独声明，不跨场景替代。

回传后人工复核分场景更新风险、原后缀复现、严格余量恢复/原成功损失，再决定是否有证据支持新的训练方案。一次延迟只是离线因果探针，不能自动采用，也不是专家可执行标签；不自动训练、扩大、调参或部署。

## 已完成的有限验证

34项定向测试通过：观察非干预、RNG一致、时间窗口/环绕yaw、上下文选择、精确后缀、旧参考对象/进度、只延迟一次但仍推理、固定抽样、身份篡改/输出复用拒绝及覆盖判定。

S4 #1298/S8 #2330各源60步，从step10分别续20步，5.505秒完成；正常后缀状态/命令最大误差均0。源与分支均未到达、无接触；S8源已在step2首次损失余量，强制step10只用于入口预检，绝不作为正式首次危险前上下文。其短正常/延迟分支余量损失2/0步不能外推全程收益。10个更新对中未出现旧安全→新危险，仅是极小截断样本。`decision=bounded_preflight_only`，不重跑smoke。

## 环境、时间和资源

工作目录 `D:\program\endtoend Parking`，唯一Python `D:\conda\envs\endtoend-parking\python.exe`，3.12.13、NumPy2.5.2、torch2.13.0+cpu，CPU单线程，无新安装/GPU要求。不要同时编辑执行代码或训练。预计数分钟、建议预留5～15分钟；24源及最多48续执行回合长度不定，短预检不能线性保证耗时。建议4GB可用内存/1GB磁盘，非实测峰值。

## PowerShell运行（同一窗口，依次执行）

### 1. 只读身份预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$updatePython = 'D:\conda\envs\endtoend-parking\python.exe'
$updateRoot = 'D:\program\endtoend Parking\runs\diagnostics\reference-update-stability-20261019-v1'
$updateConfig = Join-Path $updateRoot 'config.json'
$updateHash = '3474f667b06253515926546f4071087cf754c434cb505a61f6221d6e3175c211'
if ((Get-FileHash -LiteralPath $updateConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $updateHash) { throw '配置摘要变化，停止反馈' }
& $updatePython -u -B scripts/validate_reference_updates.py --config $updateConfig --config-sha256 $updateHash --check-only
if ($LASTEXITCODE -ne 0) { throw '身份校验失败，不继续' }
```

预期 `INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`。不要重新prepare或smoke。

### 2. 正式24源及适用分叉

```powershell
if (Test-Path -LiteralPath (Join-Path $updateRoot 'results')) { throw '已有正式输出，禁止覆盖或重跑，先反馈' }
$updateLog = Join-Path $updateRoot 'console.log'
if (Test-Path -LiteralPath $updateLog) { throw '已有日志，禁止覆盖，先反馈' }
& $updatePython -u -B scripts/validate_reference_updates.py --config $updateConfig --config-sha256 $updateHash 2>&1 | Tee-Object -FilePath $updateLog
if ($LASTEXITCODE -ne 0) { throw '失败/中断，保留所有文件，不自动重试' }
```

进度：每任务`COLLECT BEGIN`；有合格上下文时`FRESH CONTINUATION`→`FRESH SUFFIX MATCHED; SINGLE DELAY BEGIN`；最后`COMPLETE role=...`。缺上下文的任务直接COMPLETE，仍算完整账本而非成功配对。整批完成以`results/report.json`的`status=completed`且24条tasks为准，不以最后一行控制台替代。

## 输出、停止和回传

根目录console.log；results/identity.json完整身份；results/task-NNNN/report.json源/分支结果、逐次更新变化与几何事件；trajectories.npz保存原始网络参考、原源与两分支状态/命令和上下文，report内含SHA256；results/report.json汇总覆盖及全部源。逐源落盘，不覆写旧目录。

配置/代码/输入漂移、初态BEV不符、旧源或正常后缀不匹配立即停止。异常写status.json并保留已完成源；Ctrl+C也保留，强杀可能没有status.json。没有自动resume/retry；不要删除results、改hash、换源或自行补跑，反馈后另定剩余任务计划。缺覆盖是有效诊断结果，不是运行故障。

完成后回复“训练域参考更新验证完成”；中断时附控制台末尾和status.json即可。Agent随后核验24源、轨迹摘要和机制证据，仍不自动启动训练。

## 运行结果（2026-10-19 19:31–19:35，Agent 代跑）

- 只读预检通过；正式运行 235.4 s，24/24 源均落盘 report 与 trajectories，`status=completed`，无 `status.json`。24/24 起始 BEV 与训练归档一致，19/19 配对源正常后缀复现（状态/命令误差 0.0），与 LT04 重叠的 12 源旧摘要全部匹配。
- 角色：`pre_loss_recovery` 12（S4 10、S8 2）、`success_control` 7（S4 2、S1 2、S2 3）、`missing_nonzero_pre_loss_context` 4（S8）、`failure_without_margin_loss` 1（S1 #345）。配对 19、缺失 5。
- 覆盖：配对≥8、S4 恢复 10≥4 且成功对照 2≥2、S1 2≥2、S2 3≥2 全部满足 → `mechanism_coverage_satisfied=true`、`decision=mechanism_audit_ready_no_training`；S8 同场景成功对照 0，`s8_success_control_available=false` 已单独声明。
- 更新变化（605 次更新对）：xy RMS 中位 0.082 m、p95 0.396 m；yaw RMS 中位 0.031 rad、p95 0.147 rad。**旧安全→新危险：规划 13 次、接触 2 次**（占 2.5%），规划退化 S8 9、S4 4，接触退化 S4 2（#1394、#1441）。
- 单次延迟（19 条配对）：成功 7/7、碰撞 12/12 与 fresh 完全一致；余量侵入步数延迟更好 5 条、更差 4 条、相同 10 条，`delay_margin_safe_wins=0`。单次延迟不改变任何任务分类，也没有系统方向。
- 结论：机制审计可用但不训练；主成绩 68.3%/25.7% 与正式 MPC/模型/部署未变。S8 仅 2 条配对且无同场景成功对照，结论受限；明细见 `runs/diagnostics/reference-update-stability-20261019-v1/results/summary.md`。
