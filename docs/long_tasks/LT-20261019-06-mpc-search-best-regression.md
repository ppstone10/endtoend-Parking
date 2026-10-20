# LT-20261019-06：当前搜索最佳样本69条双臂回归

## 状态、目的与边界

已运行完成（2026-10-19 17:24–17:26，Agent 代跑并已核验）：原臂逐条复现历史 36/27/6，试验臂 36/28/5（保留原成功 32/36，S4 丢 3、S6 丢 1），四项门槛全不满足，`decision=stop_no_expansion_or_deployment`；本页命令不再重复执行。不训练、不更新原MPC或部署。

上一候选版（原均值/热启动/零候选）正式69条联合失败，七条首分岔均为直接执行热启动。新假设是：只比较当前CEM搜索实际采样过的最佳序列和原最终精英均值，热启动仍作为原搜索初值，但不直接列为返回候选，零序列也不额外注入。原均值平局优先，采样/限幅/拟合/RNG消耗不变，额外评估最终均值1条序列。

唯一主要变量是相对于原MPC的返回解选择规则；不是对上一版中“热启动单独贡献”的完整因果消融。不改代价、权重、种群、迭代、模型、阈值、感知、参考、车辆或碰撞/到达标准，无安全盾/专家接管，不使用S9。

依据[MBRL-Lib原实现](https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/planning/trajectory_opt.py)中最佳样本/最终均值返回的区分及[iCEM原论文](https://proceedings.mlr.press/v155/pinneri21a.html)的采样记忆约束；未引入完整iCEM、有色噪声、梯度法、外部依赖或复制外部代码。

低代价不保证控制平滑、安全或任务到达；本版不保证静止参考零代价，也不抹去前轮静止求解误差证据。

## 短预检证据及风险

44项测试覆盖当前搜索各代种群逐元素一致、最终均值一致、RNG相同、代价不劣、候选来源/限幅/平局和异常、显式注入恢复、冻结篡改/覆盖拒绝及账本校验，并覆盖原MPC与前期诊断相关回归。

双臂分别运行S4 #164/S8 #292各60步，计算1.969秒。原臂timeout2，试验臂S4为oscillation、S8为timeout；均未到达，无接触，短预检门槛强制不通过。试验S4有58步原均值/2步搜索样本，S8为60步原均值。此结果只验证入口/账本，不证明全回合提升，不重跑preflight。

## 冻结输入与参数

根目录 `runs/validation/mpc-search-best-ab-20261019-v1/`，配置 `config.json` SHA256：

`2210270db4b1a24daa65f952dd5676b6d8dac65a2575614c69fd3fa81e83df5c`

冻结6输入、88执行代码文件及环境；原86文件逐项保持不变，新增两个独立文件，不修改上版子类/脚本或其身份文件。新增文件改变代码全集，旧入口可能拒绝原清单重跑，这是保护机制，不改旧hash绕过。

| 输入 | SHA256 |
|---|---|
| 既有69条protocol.json | `63c3566fa33367853938d26569ec94a4b2e1aa5ebdf64e3606e80c68c34b25d1` |
| 原val.npz | `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963` |
| 原manifest.json | `424fcb68cc8257707a2568c8204bf27f8db2c75c16ddca9a0752557da7ebd077` |
| R0 dep-epoch0007.pt | `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8` |
| 原完整E5基线报告 | `87ad68d81e3dfd97853df2aaf01da612f6f70fe8d523ec49949fcf2709723a9c` |
| 上版冻结config.json（仅来源身份） | `45246333b17039c3a289420721773c13629ec39ea8f17209dccc9bf721686ec8` |

精确绝对路径见新配置。固定已知69索引，不补选、不用其标签训练；它不是独立选型或完整300条验收。

参数沿用原E5：R0父阈值约0.26、原实际感知口径、seed0（每任务加索引）、max600/replan10，MPC dt0.1/horizon10及全部原代价/种群/迭代、车辆和到达容差；接触余量0、规划标准不降低。

## 环境与时间资源

工作目录 `D:\program\endtoend Parking`，Python `D:\conda\envs\endtoend-parking\python.exe`，3.12.13/NumPy2.5.2/torch2.13.0+cpu，CPU单线程，无GPU或新安装。不要同时训练或编辑执行源。

预计3～10分钟：上版完整双臂151.2秒，本版增加逐步JSON账本，仍受CPU负载与回合长度影响，不能用1.969秒短预检线性承诺。建议4GB可用内存、1GB磁盘（非实测峰值）。正式双臂是多轮批量评测，由用户运行，Agent不自动代跑。

## PowerShell运行（同一窗口，依次执行）

### 1. 只读预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$searchPython = 'D:\conda\envs\endtoend-parking\python.exe'
$searchRoot = 'D:\program\endtoend Parking\runs\validation\mpc-search-best-ab-20261019-v1'
$searchConfig = Join-Path $searchRoot 'config.json'
$searchHash = '2210270db4b1a24daa65f952dd5676b6d8dac65a2575614c69fd3fa81e83df5c'
if ((Get-FileHash -LiteralPath $searchConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $searchHash) { throw '配置摘要变化，停止并反馈' }
& $searchPython -u -B scripts/compare_search_best_mpc.py --config $searchConfig --config-sha256 $searchHash --check-only
if ($LASTEXITCODE -ne 0) { throw '输入/代码/环境校验失败，不继续' }
```

预期`INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`，不要再运行prepare或smoke。

### 2. 正式原臂69条→试验臂69条

```powershell
if (Test-Path -LiteralPath (Join-Path $searchRoot 'results')) { throw '已有正式目录，禁止覆盖/重跑，先反馈现有结果' }
$searchLog = Join-Path $searchRoot 'console.log'
if (Test-Path -LiteralPath $searchLog) { throw '已有日志，禁止覆盖，先反馈' }
& $searchPython -u -B scripts/compare_search_best_mpc.py --config $searchConfig --config-sha256 $searchHash 2>&1 | Tee-Object -FilePath $searchLog
if ($LASTEXITCODE -ne 0) { throw '失败或中断，保留所有产物并反馈，不自动重试' }
```

正常进度：`ARM original BEGIN`、`original [n/69]`、`ORIGINAL 69 HISTORICAL RESULTS MATCHED`、`ARM candidate BEGIN`、`candidate [n/69]`、`COMPLETE`。原臂任一历史结果不符即在试验臂之前停止。

## 输出、停止与恢复

根目录`console.log`记录控制台；新`results/identity.json`记录完整身份；`original.json`与`candidate.json`记录69条和显式solver_class/solver_variant；`report.json`为完成状态及门槛。试验类必须是SearchBestMPC，不能混报为上版CandidatePreservingMPC。

`results/audits/task-NNNN.json`逐控制步记录前态、实际窗口、首命令、原均值/最佳样本/选中代价与来源、最佳样本代次/索引；候选报告`solver_audit_files`保存各账本SHA256。每试验回合结束落一份账本，整臂结束才写该臂完整报告；账本是诊断产物，不是自动续跑检查点。

异常保留`results/status.json`、已完成臂及逐回合账本；硬终止可能无status。允许Ctrl+C，无自动续跑参数。不要删除/覆盖部分目录或重复启动；反馈日志及已有文件，Agent核验后另开新身份/目录安排需要重跑的整臂。不手改hash、阈值或分母，不自动启动后续训练/评测。

## 联合门槛和反馈

先核验冻结、各69条唯一且相同身份/容差/协议、无复原失败、原臂历史36成功/27碰撞/6振荡及逐条步数/终点误差复现、试验逐步账本完整/摘要一致。

试验臂同时要求：成功≥39，碰撞≤24，保留原36成功中的≥33，每场景原成功丢失≤1；与上版门槛一致，不因新方法放宽。额外人工检查振荡、方向切换、路径长度/耗时与S4/S6/S8退化，不能把少动/不动算到达。

任一失败停止本候选，不扩大300条/部署/为其补训。全部通过也只获得另拟独立300条选型计划的资格，不代表主目标成功≥70%且碰撞≤10%已达标，不自动替换原MPC。

完成后回复“当前搜索最佳样本双臂69条完成”，提供该根目录；异常直接反馈日志末尾及status。Agent再核验结果、候选选择及场景回归，不按单个汇总数字宣布提升。

## 运行结果（2026-10-19 17:24–17:26，Agent 代跑）

- 只读预检通过；正式运行 136.0 s（原臂约 68 s、试验臂约 68 s）。两臂各 69/69、`status=completed`；原臂在试验臂开始前逐条复现历史 36/27/6；试验臂 69 份逐步账本齐全。
- 求解器身份：原臂 `MPCController`（额外序列 0）、试验臂 `SearchBestMPC`（额外 1），均 `solver_variant=current_search_best`、`production_default_unchanged=true`。
- 试验臂 **36 成功 / 28 碰撞 / 5 振荡**，保留原成功 **32/36**，场景丢失 S4 3、S6 1。四项门槛：成功 ≥39 ✗、碰撞 ≤24 ✗、保留 ≥33 ✗、每场景丢失 ≤1 ✗，`gate_passed=false`。
- 转移 8 条：碰撞→成功 3（#121 S3、#163 S4、#223 S5），振荡→成功 1（#253 S6，600 步停滞变 424 步到达），成功→碰撞 4（#164/#168/#175 S4、#246 S6）。净效应为成功持平、碰撞 +1、振荡 −1。
- 分场景：S3 6→7、S5 6→7、S4 6→4（丢 3 换回 1）、S6 持平但丢 1 得 1，S1/S2/S8 不变。
- 开销：路径 10.688→10.232 m、停车时间 20.54→19.45 s、终点位置误差 1.784→1.677 m；额外每解 1 条均值序列，账本 69 份。
- 结论：停止该候选，不扩大、不部署、不补训；明细见 `runs/validation/mpc-search-best-ab-20261019-v1/results/summary.md`。下一步应查 S4 三条原成功（#164/#168/#175）在选到低代价样本后为何反而碰撞，而不是继续更换返回规则。
