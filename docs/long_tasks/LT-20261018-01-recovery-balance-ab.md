# LT-20261018-01：恢复数据场景覆盖双臂训练

状态：**两臂已运行完成（2026-10-18 22:37–22:50，Agent 代跑并已核验）**。R1/R2 各 4 轮、`status=completed`，产物齐全，未部署、未校准、未评测。本页命令已执行完毕，不再重复运行；是否进入 §9 的下一阶段由用户决定。

## 1. 本次验证什么

唯一主要变量是恢复样本的选择。现有全局机动裁剪会清空 S6/S8 的恢复样本、使 S4 仅剩倒车恢复。本次比较：

| 组别 | 数据 | 初始化 | 预算 |
|---|---|---|---|
| R1 对照 | 原2918条训练集 | 当前v40 epoch7校准权重 | 4轮 |
| R2 候选 | 同总量、保持场景覆盖的2918条 | 与R1完全相同 | 4轮 |

两组均保留原2400条专家和前进/倒车恢复各259条。R2优先保留全部碰撞回溯样本，只替换162条恢复来源状态；S4/S6/S8恢复数为89/50/21。与主val和独立选型集任务身份零重叠；原2400条输入及有效轨迹标签已核验一致。

这仍是待验证的改进假设，不能因配额合理就宣布碰撞率下降。

## 2. 环境与已准备输入

- 工作目录：`D:\program\endtoend Parking`。
- 唯一Python：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12。
- 本次任务目录：`D:\program\endtoend Parking\runs\training\recovery-balance-ab-20261018-v2`。
- 训练入口：`scripts/train_fixed_threshold.py`；不要换回会自动校准的 `scripts/train_model.py`。
- 父模型：`runs/validation/arch-ab/dep-epoch0007.pt`，SHA256=`d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`。
- R1：`data/task_dataset/v40_dagger_collision/train-balanced.npz`。
- R2：任务目录下 `candidate-train.npz`，SHA256=`a569ec7ea3e099c8a0f10b565ade29d88c9a9505a02bce808a191b01d45087b3`。
- val：原 `tracked_pivot_v8_3000/val.npz` 300条，只用于训练过程诊断，不据此loss挑部署权重。
- 身份清单：任务目录下 `manifest.json`，SHA256=`0da9473018ab22ce23e29086c5ef50be661dc5c6dac63f46d2cf2e0454eb700b`，含输入/配置/执行代码摘要、Python/NumPy/PyTorch版本、准备时Git HEAD及选样清单。Git HEAD仅作记录，实际执行代码由摘要冻结；已有未提交代码也纳入摘要。

v2已经准备好，不要重复prepare，也不要使用v1。v1是准备阶段的旧代码身份产物，没有正式训练，不删除、不混用。两臂都完成前不要修改相关代码、配置、输入或升级环境；身份不符会主动拒绝运行/恢复。仅阅读文档无影响。

## 3. 已冻结参数

| 参数 | 两臂共同值 |
|---|---|
| 模型 | net-v1，hidden128，max_horizon320，dt0.2，原BEV约定 |
| 训练 | CPU，PyTorch单线程，batch8，4轮，Adam重置，lr0.00005 |
| seed | 20260827 |
| teacher forcing | 0.44 → 0.36 → 0.28 → 0.20 |
| 停止阈值 | 固定父模型值0.26000000000000006（约0.26） |
| 原损失 | stop0.2，endpoint0.5/末8点，terminal1.0/yaw10，collision0，progress0 |
| 其他 | 原恢复batch平衡、shuffle、梯度裁剪1.0保持；最后一轮才启用监控且patience4，不会提前结束本次4轮 |

不引入近端新损失、数据增强、新架构或控制器改动；不自动校准、选型、部署或串行评测。`best.pt`只是基类附带的val-loss快照，不能当作部署选择；保留全部4轮快照供后续配对评测。

## 4. 预计耗时与资源

- 两组顺序运行合计暂估 **15～40分钟**，每组约7～20分钟。仅是工程估算，不是已跑完整训练的承诺。
- 依据：真实R2单批次前向/反向预检约0.16秒；每轮有400个训练batch、38个val batch和403个自由反馈评估batch，另有读盘、快照、日志和长轨迹开销。单批次不能准确外推整轮，首轮结束后以日志修正估算。
- CPU执行，PyTorch线程数1，不需要GPU；建议预留约4GB可用内存、1GB磁盘。这是运行余量建议，非已测峰值。
- 两组不要并行跑，不要为了加速临时改变线程数、设备或batch。保持电脑接电并避免休眠。

## 5. 你需要执行的命令

打开PowerShell，先定位项目：

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
```

### 第一步：运行 R1 对照

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -u -B scripts/train_fixed_threshold.py --config runs/training/recovery-balance-ab-20261018-v2/R1/config.json
if ($LASTEXITCODE -ne 0) { throw 'R1失败：停止，不要继续R2；保留日志并反馈。' }
```

等待命令结束，确认退出码0、终端出现 `TRAINING COMPLETED; WAIT FOR AGENT REVIEW`，并检查：

```powershell
Get-Content -LiteralPath 'runs/training/recovery-balance-ab-20261018-v2/R1/status.json' -Encoding UTF8
```

必须为 `status: completed`、`completed_epochs: 4`，才进入第二步。R1指标不好看也不手动换超参数；本次需要同预算对照。

### 第二步：运行 R2 候选

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -u -B scripts/train_fixed_threshold.py --config runs/training/recovery-balance-ab-20261018-v2/R2/config.json
if ($LASTEXITCODE -ne 0) { throw 'R2失败：停止，保留日志并反馈。' }
```

同样确认4轮完成：

```powershell
Get-Content -LiteralPath 'runs/training/recovery-balance-ab-20261018-v2/R2/status.json' -Encoding UTF8
```

**到这里停止。** 训练结束不会自行跑闭环；暂时不要调用其他评测、校准脚本，也不要覆盖旧deployment。

## 6. 正常进度与产物

正常控制台/`run.log`顺序包括：`START` → `epoch N/4 train BEGIN` → 每50个batch输出loss → val → 两次 `free rollout BEGIN/END` → `EPOCH N/4 COMPLETE`。自由反馈阶段会有一段时间没有batch输出；根据日志阶段和进程占用判断，不把无输出直接当卡死。

另开PowerShell窗口，可只读查看日志末尾（不启动重复训练）：

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
Get-Content -LiteralPath 'runs/training/recovery-balance-ab-20261018-v2/R1/run.log' -Encoding UTF8 -Tail 20
```

运行R2时将路径中的R1换为R2。每个组应产生：

```text
R1/ 或 R2/
  config.json                 已冻结配置
  preflight*.json             已有短预检证据
  run-identity.json           实验臂/配置/manifest身份
  run.log                     UTF8完整进度与异常
  status.json                 running / interrupted / failed / completed
  history.json                已完成轮的指标
  report.json                 仅完整4轮后写入
  checkpoints/
    epoch0001.pt … epoch0004.pt
    last.pt                   可恢复训练
    best.pt                   辅助val-loss快照，不能当部署选择
```

正常结束后 `run.lock` 不存在；不会产生 `deployment.pt`。

## 7. 中断、异常与恢复

- 看到Traceback、非有限loss、身份变化、磁盘错误或非0退出码：停止，不启动另一组；保留日志，反馈给Agent。
- 正常Ctrl+C会记录interrupted并释放锁。若已有 `checkpoints/last.pt`，代码/配置/输入/环境没改变，且确认旧进程已退出，可显式从上一完整轮恢复：

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -u -B scripts/train_fixed_threshold.py --config runs/training/recovery-balance-ab-20261018-v2/R1/config.json --resume
```

R2中断时只把路径换成R2，绝不能把另一组last复制过来。恢复会加载模型、优化器及history；当前未完成轮会从上一轮重新计算，不声称逐batch续跑。

- 首轮完成前中断、没有last：不要强行resume或删除目录，先反馈Agent，必要时另建任务目录重新准备。
- 强制结束进程/断电可能残留 `run.lock`，其中保存PID。不要直接删锁重开；先反馈锁内容与日志，由Agent核实该PID对应进程已经退出后再给出处理步骤。
- 已有report的完成任务拒绝重复运行。需要新实验时另开目录；禁止覆盖父模型、旧训练数据、原验证/选型数据及旧报告。

## 8. 跑完后交回什么

本地同一项目中，你直接回复：

> LT-20261018-01 两组都跑完了，结果在 runs/training/recovery-balance-ab-20261018-v2，请检查。

不需要粘贴长日志，不要移动或删除整个目录。Agent将读取：

1. manifest、R1/R2的config、run-identity、status、report和history；
2. 两组run.log末尾及异常信息；
3. 两组各4份epoch快照及其摘要、last；
4. 代码/环境/输入身份与两组共同配置是否一致。

若失败，告诉我失败组别、终端最后20行及目录位置；不要用“可能跑完了”替代状态核验。

## 9. Agent后续把控标准

先核验产物，再决定是否准备下一份69条回归评测的人工长任务计划；本页不授权自动启动下一阶段。

- 两组必须各4轮、同初始化、固定阈值和预算；没有自动deployment或混入其他变量。
- 对照R0原模型、R1同轮和R2同轮的闭环，而非只比较训练loss或ADE。
- 预先声明的扩大筛选：R2成功数比R0/同轮R1较好者至少+3，碰撞数比两者较低值至少-3；36条原成功至少保留33条，各场景原成功不得少超过1条。该筛选只是工程出口，不是统计显著性或正式达标证明。
- 合格候选后续才进入独立300条选型及原完整300条主验收；主标准仍为成功≥210、碰撞≤30，并做多控制seed复核。
- 4轮均无可信收益则停止这项对照，不自动追加epoch、调loss或改变停止阈值。

## 10. 准备证据与复现边界

新增测试覆盖配额保护、来源去重、配置/代码/输入变化拒绝、锁、玩具模型中断恢复及缺失同轮快照恢复、冒烟不调用fit不更新权重。R1真实check-only、R2真实单batch前反向均已通过；没有跑正式模型的完整epoch或正式恢复长跑。

现成v2准备目录若不存在，本页不再是“直接可运行”的状态，请告知Agent。可重建的准备入口是 `scripts/prepare_recovery_balance_ab.py --output <全新目录>`，它只准备数据/配置，不训练；不要盲目把此占位参数复制执行。不同目录或代码身份需更新本计划及预检证据。

对应规范：[RECOVERY-AB](../../spec/recovery_balance_training.md)；准备任务：[T-221508](../traces/2026-10-18.md#t-221508--l3--人工双臂训练交付准备)。

## 11. 运行结果（2026-10-18 22:37–22:50，Agent 代跑）

- R1：22:37:37–22:43:59，381.7s，4/4 轮，`status=completed`，末轮 val_loss 1.530351。
- R2：22:44:08–22:50:18，370.2s，4/4 轮，`status=completed`，末轮 val_loss 1.672187。
- 两臂均写出 epoch0001–0004、`last.pt`、`best.pt`、`history.json`、`report.json`；`run.lock` 已释放，无 `deployment.pt`，未触发中途恢复。
- 身份核验：两臂 `config.json` 只在 `arm`/`output`/`train`/`trainer.checkpoint_dir` 上不同，manifest 摘要与运行期身份一致，父模型与阈值未变。
- 实测耗时约 6.4 分钟/臂，短于 §4 的 7～20 分钟/臂估算；两臂合计 12.7 分钟。
- 执行环境偏差：本会话沙箱以低完整性令牌运行，对既有工作区目录写入被拒。首次 R1 启动在创建 `run.lock` 时以 `PermissionError` 退出，未产生任何正式产物；经用户授权改为完全权限执行后重跑，命令、配置与身份与本页 §5 一致。用户自行在普通 PowerShell 执行时的行为不变。
- 逐轮开环指标与快照摘要见任务目录 `summary.md`（`runs/training/recovery-balance-ab-20261018-v2/summary.md`）。训练侧差异不是闭环收益证据，仍需按 §9 做 69 条已知回归与完整 300 条无干预 E5 对照。
- 下一步：由 Agent 核验后再交一份 69 条回归闭环人工长任务计划；本页不授权自动启动评测、校准或部署。

## 12. 独立产物复核（2026-10-19）

已核对输入/代码/配置/环境摘要、两组完成状态与日志，实际加载8份epoch快照及2份last；摘要、每轮400次优化器更新、固定阈值及history一致，未见ERROR/Traceback，无残留锁和deployment。报告耗时约R1 6.36分钟、R2 6.17分钟。机器可读审计为 `runs/training/recovery-balance-ab-20261018-v2/agent-training-audit-20261019.json`。

验证ADE随轮次交叉，不能由loss或末轮挑模型；[LT-20261019-01闭环配对评测计划](LT-20261019-01-recovery-balance-regression.md) 已准备，尚未启动，仍由用户执行。
