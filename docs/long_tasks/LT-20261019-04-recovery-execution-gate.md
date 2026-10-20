# LT-20261019-04：训练域恢复标签配对执行验证

## 状态与目的

已运行完成（2026-10-19 15:46，Agent 代跑并已核验），脚本判定 `insufficient_or_execution_gate_failed_no_training`；本页命令不再重复执行。这是诊断，不是训练，也不生成可直接回灌的数据。只改变同一状态之后的参考来源（网络/重规划专家），固定原MPC与实际E5感知，判断专家恢复标签是否真正可执行。最终成功率≥70%、碰撞率≤10%仍须后续原300条完整验收，本实验不能代替。

前置证据：S4/S8高噪声训练任务1298/2330各从第10步保留MPC热启动和感知RNG，网络继续20步与原回合状态/命令误差均为0；专家均完成两次重规划。预检总计算4.66秒，20步截断的timeout/label_passed=false不是完整恢复失败。27项相关单测通过。

## 冻结输入与协议

工作目录 `D:\program\endtoend Parking`；Python固定 `D:\conda\envs\endtoend-parking\python.exe`，3.12.13、NumPy2.5.2、torch2.13.0+cpu；CPU单线程，不需GPU。

配置：`runs/diagnostics/recovery-execution-gate-20261019-v2/config.json`，SHA256：

`f20a17905febf3db49922adc8960d165521e4240b21ce598263ca77d7122d37f`

配置包含5个输入文件、85个执行代码文件及运行环境的摘要；运行前后自动逐项核验。不可修改源码后沿用本配置，也不可改配置摘要绕过失败。v1是旧预检，不能使用。

| 输入 | SHA256 |
|---|---|
| `data/task_dataset/tracked_pivot_v8_3000/train.npz` | `8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f` |
| 对应训练manifest（精确路径见config） | `424fcb68cc8257707a2568c8204bf27f8db2c75c16ddca9a0752557da7ebd077` |
| 原val（仅身份排除，不运行或查询专家） | `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963` |
| independent selection（仅身份排除） | `797e9e5d0d02576c45df65c74f142cc05e83770eb6bbfcc28be560dbc722d51d` |
| 原父模型 `runs/validation/arch-ab/dep-epoch0007.pt` | `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8` |

固定种子、车辆、到达容差和原MPC；max_steps=600、每10控制步重规划、接触余量0、规划余量0.2m、父停止阈值约0.26；不加运行时安全边界干预。恢复状态从首次实际规划余量损失之前最近的安全重规划发出点选取；无余量损失且成功的任务提供中段成功对照。状态0与非零访问状态须在审核时区分，状态0本身不证明获得新恢复分布。

选样seed=20261019，按训练任务的机动×噪声分层，预先固定，不根据结果换样：

- S4：1376、1298、1394、1455、1451、1675（前进/倒车各含clean/high/low）。
- S8：2348、2330、2332、2364、2317、2356（现有池均前进，clean/high/low各2）。

每源任务最多一对分支，12源不保证有12个合格状态；无安全状态、专家无解、碰撞或超时全部保留。网络分支先按1e-9容差复现完整原后缀（包括长度），失败立即停止，不把不可信比较计入专家结论。

输入契约警告：父模型元数据声明geometry目标通道，但现有E5实际构建camera目标通道。已核对实际输入与存档训练BEV逐元素相同，本实验冻结现有真实管道并逐任务校验起始BEV，不静默切换geometry；声明差异不是已证明的性能原因。

## 时间、资源与依赖

依赖当前冻结文件及已安装conda环境，无新增安装。不要同时训练或修改执行源码。建议预留4GB内存、1GB磁盘（资源建议，不是实测峰值）。

预计数分钟至十几分钟，困难专家搜索可能显著更久，不能用快速短预检推算全部耗时。最多约720次专家重规划，按每次8秒搜索预算约96分钟，另有原回合/网络与规划开销；这不是严格墙钟上限。因此按长任务由用户运行，Agent不自动启动。每次专家规划有BEGIN/END进度，单个BEGIN长时间未结束时先保留日志，不自动重试。

## 运行步骤（同一个PowerShell窗口）

### 1. 环境、摘要和只读预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$probePython = 'D:\conda\envs\endtoend-parking\python.exe'
$probeRoot = 'D:\program\endtoend Parking\runs\diagnostics\recovery-execution-gate-20261019-v2'
$probeConfig = Join-Path $probeRoot 'config.json'
$probeHash = 'f20a17905febf3db49922adc8960d165521e4240b21ce598263ca77d7122d37f'
if ((Get-FileHash -LiteralPath $probeConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $probeHash) { throw '配置摘要不匹配；停止并反馈' }
& $probePython -u -B scripts/probe_recovery_execution.py --config $probeConfig --config-sha256 $probeHash --check-only
if ($LASTEXITCODE -ne 0) { throw '输入/代码/环境预检失败；不要继续' }
```

预期 `INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`。不要重新运行`--smoke`：它的preflight目录已经存在，入口会拒绝覆盖。

### 2. 正式12源任务（只有这一步启动计算）

```powershell
if (Test-Path -LiteralPath (Join-Path $probeRoot 'results')) { throw '已有正式目录；禁止覆盖或重跑，请反馈现有结果' }
$probeLog = Join-Path $probeRoot 'console.log'
if (Test-Path -LiteralPath $probeLog) { throw '已有控制台日志；请先反馈，不自动覆盖' }
& $probePython -u -B scripts/probe_recovery_execution.py --config $probeConfig --config-sha256 $probeHash 2>&1 | Tee-Object -FilePath $probeLog
if ($LASTEXITCODE -ne 0) { throw '任务失败或中断；保留目录与日志并反馈，不重试' }
```

入口不带`--smoke`才是正式运行。输出依次出现COLLECT、NETWORK CONTINUATION、NETWORK SUFFIX MATCHED、EXPERT CONTINUATION、各次EXPERT REPLAN BEGIN/END及COMPLETE。无可选状态也须记账；不是跳过后补挑其他任务。脚本完成后检查 `results/report.json` 的status，不以PowerShell最后一行或单个COMPLETE认定全部完成。

## 输出与中断恢复

输出根 `runs/diagnostics/recovery-execution-gate-20261019-v2/`：

- `config.json`、`preflight/report.json`：已冻结配置与已有短预检。
- `console.log`：本次正式日志。
- `results/identity.json`：配置、输入、代码和环境身份。
- `results/task-NNNN/report.json`、`trajectories.npz`：逐源结果和原/网络/专家真实轨迹与控制序列。
- `results/report.json`：正式完成时的全量账本、覆盖与决策；异常时可能只有`results/status.json`和已完成任务。强制结束进程可能来不及写status。

支持Ctrl+C停止；**没有自动续跑参数**。已完成/部分目录不可删除、覆盖或重复运行。中断后反馈日志和已有目录，Agent核对后另开新目录、为剩余任务制订计划，不让重跑改变分母。冻结校验失败同样停止，不临时更新hash。

## 预声明决策门槛与反馈

先检查正式completed、12源账本、身份冻结、网络后缀复现、所有失败及缺失均保留，再讨论训练：

1. 至少8对完整验证；S4/S8各自均有首次余量损失前恢复状态和成功对照。
2. 每个已配对专家分支均须完整到达、无实际接触、无任何规划余量侵入；短截断不算通过。
3. 两场景各至少一例网络失败而专家上述条件全部通过，并保留成功对照安全性。

全部满足才输出`candidate_for_separate_data_training_plan`，仍须Agent审核后另交单变量数据训练计划；不自动训练。否则输出`insufficient_or_execution_gate_failed_no_training`，区分覆盖不足与专家执行失败，继续查参考/执行兼容性，不扩大挑选或盲目重训。这是小样本机制门槛，不是整体性能验收。

运行完回复“12任务配对验证完成”，提供上述目录即可；失败则直接反馈console.log、status和最后任务编号。后续由Agent复核原始轨迹、成功对照、余量时间线，再决定训练改法。

## 运行结果（2026-10-19 15:46，Agent 代跑）

- 只读预检通过（`INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`），正式运行 15:46:12–15:46:58，46.1 s；`results/report.json` 为 `status=completed`，12/12 任务均落盘 report 与 trajectories，无 `status.json`，未中断。
- 身份核验：12/12 `initial_bev_exactly_matches_training_archive=true`、网络后缀 12/12 复现（状态/命令最大误差 0.0，步数 5～197）、`for_training=false`、`smoke=false`。
- 覆盖：12 对、缺失 0；但两场景成功对照均为 0（12 个状态全是“首次余量损失前”恢复状态），S8 有 4 个 `context_step=0` 表示分叉点在任务初态（分叉状态本身仍有规划余量）。
- 结果：`network_successes=0`、`expert_successes=9`、`expert_execution_label_passed=6`。S4 六例专家全部带余量到达（31～79 步、零接触、零余量侵入），网络同状态 5～90 步即碰撞；S8 六例专家 0 通过（3 例到达但侵入余量 2～12 步，另 3 例在第 10/20 步重规划时因起始位姿与障碍物冲突提前结束，engine 记为 timeout）。原臂是**网络回合**（不是专家回合），其在 S8 的第 2～3 步即侵入 0.2 m 余量；后续首次侵入回放显示专家参考与当时的 MPC 窗口均安全，问题落在选中控制的第一步就离开余量集合（更正见 [S8 首次侵入分析](../../runs/diagnostics/s8-expert-first-loss-20261019-v1/analysis.md)）。
- 判定：`insufficient_or_execution_gate_failed_no_training`。S4 的标签可执行性门槛通过、S8 的专家执行门槛未通过、两场景都缺中段成功对照；不输出训练候选、不自动训练、不生成回灌数据。
- 明细见 `runs/diagnostics/recovery-execution-gate-20261019-v2/results/summary.md`；主成绩 68.3%/25.7% 与部署未变。
