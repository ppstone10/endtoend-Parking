# LT-20261019-02：航向周期损失双臂验证

状态：**两臂已运行完成（2026-10-19 10:50–11:03，Agent 代跑并已核验）**。W0/W1 各 4 轮 `completed`，产物齐全，未部署、未校准、未评测；本页命令执行完毕，是否进入 69 条闭环对照由用户决定。

## 1. 目的与边界

验证已确认的 yaw 周期监督缺陷是否能改善闭环，不继续重复“只换恢复数据”的旧实验。原模型在候选训练集自由预测时，124/2918 条、4347 个有效点存在 raw 航向差绝对值 >π。这里确认的是损失错误，不是它已解释全部 S4 碰撞。

| 臂 | 唯一主要变量 | 数据/初始化/预算 |
|---|---|---|
| W0 | yaw_error_mode=raw，旧损失 | 同一 R2 候选2918条、原 v40 父权重、4轮 |
| W1 | yaw_error_mode=wrapped，最短角差 | 与 W0 相同 |

逐点、末8点对齐、终点航向三处统一使用该模式。其余全部固定：net-v1 hidden128、horizon320、dt0.2、原BEV约定；CPU单线程、batch8、Adam新建、seed20260827、lr5e-5、clip1；teacher forcing=0.44/0.36/0.28/0.20；stop0.2、endpoint0.5、terminal1/yaw10、collision0、progress0、原恢复batch平衡。停止阈值固定0.26000000000000006；不做新数据、近端损失、架构或控制器改动，不自动校准/部署/闭环评测。S9不动。

## 2. 冻结身份与短预检

- 工作目录：`D:\program\endtoend Parking`；Python仅用 `D:\conda\envs\endtoend-parking\python.exe`（3.12.13，NumPy2.5.2，PyTorch2.13.0+cpu）。
- 已准备根目录：`runs/training/yaw-loss-ab-20261019-v2`。**不要重新prepare，不用v1。** v1为准备阶段旧代码身份，没有正式训练，保留不混用。
- manifest SHA256：`0a29bf3172e311b1a48017923cb18ccaa560ecd8c314d04318f2ab781af343be`。含所有输入、两份配置、执行代码摘要及依赖环境；工作树含未提交修改，以实际代码摘要而非仅Git HEAD锁定执行版本。两臂完成并审核前不得编辑相关代码/配置/输入或升级环境。
- W0 config SHA256：`b81cf19f8ae56f1c1e667699288481862746678d14818df3da15882f138a975b`。
- W1 config SHA256：`854d8bd874be1d1ad8dab012c1b5639effc008dcc981d74140ca6d93778fe230`。
- 父模型：`runs/validation/arch-ab/dep-epoch0007.pt`；SHA256=`d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`。**不是 R1/R2 末轮或最好轮，优化器不沿用。**
- 两臂数据均为 `runs/training/recovery-balance-ab-20261018-v2/candidate-train.npz`；SHA256=`a569ec7ea3e099c8a0f10b565ade29d88c9a9505a02bce808a191b01d45087b3`。原2400条+恢复518条；复用冻结身份与零重叠审计。
- val为 `data/task_dataset/tracked_pivot_v8_3000/val.npz`，SHA256=`ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`。val-loss只诊断，不据此部署。
- 两臂 `preflight-smoke.json` 已通过：2918/300条，400训练batch/轮，课程与阈值相同，前向/反向有限，**没有 optimizer.step、fit 或保存正式checkpoint，权重逐项未改变**。第一批恰未触发角度分支，因此两臂smoke_loss相同是正常的；不能拿它作效果证据。

模型/训练代码已变化，旧恢复数据实验的冻结代码校验将拒绝再次运行，这是预期保护；没有改写任何旧manifest、模型或结果。不要绕过检查复用旧计划。

## 3. 耗时与资源

两臂顺序合计估计 **12～25分钟**（每臂6～12分钟）；不确定性来自后台负载与 wrapped 三角计算。依据为上轮同数据量/模型/四轮实际每臂6.2～6.4分钟，以及本轮真实单batch前反向约0.28秒（不含数据加载）。不是实测完整新训练时长。

CPU运行、PyTorch线程1，不需要GPU；建议4GB可用内存、1GB剩余磁盘（非已测峰值）。不要并行两臂，不改batch/线程/设备；电脑接电，避免休眠。

## 4. 可复制 PowerShell 命令

先设置环境并运行W0；不要使用会自动校准/部署的默认训练入口：

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'D:\conda\envs\endtoend-parking\python.exe' -u -B scripts/train_fixed_threshold.py --config runs/training/yaw-loss-ab-20261019-v2/W0/config.json
if ($LASTEXITCODE -ne 0) { throw 'W0失败：停止，不运行W1，保留日志并反馈。' }
Get-Content -LiteralPath 'runs/training/yaw-loss-ab-20261019-v2/W0/status.json' -Encoding UTF8
```

确认W0 `status=completed`、`completed_epochs=4` 和 `TRAINING COMPLETED; WAIT FOR AGENT REVIEW`，再单独执行W1：

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -u -B scripts/train_fixed_threshold.py --config runs/training/yaw-loss-ab-20261019-v2/W1/config.json
if ($LASTEXITCODE -ne 0) { throw 'W1失败：停止，保留日志并反馈。' }
Get-Content -LiteralPath 'runs/training/yaw-loss-ab-20261019-v2/W1/status.json' -Encoding UTF8
```

**到这里停止，告诉Agent“W0/W1训练完成”。不要自行追加epoch、校准、运行八组评测或替换部署。**

## 5. 进度、日志与产物

每臂为 START → train每50batch日志 → val → 两次free rollout → EPOCH N/4 COMPLETE。自由滚动阶段输出较少不等于卡死。可在另一个PowerShell只读查看：

```powershell
Get-Content -LiteralPath 'D:\program\endtoend Parking\runs\training\yaw-loss-ab-20261019-v2\W0\run.log' -Encoding UTF8 -Tail 20
```

W1运行时换W1路径。每臂应有 `run-identity.json`、UTF8 `run.log`、`status.json`、`history.json`、`report.json`、`checkpoints/epoch0001.pt` 至 `epoch0004.pt`、`last.pt`。`best.pt`是辅助val-loss快照，不作为部署选择。正常结束无 `run.lock`，没有新deployment。

## 6. 中断与恢复

- Traceback、非有限数、身份错误、磁盘问题、非0退出码：立即停在该臂，保留所有产物反馈，不自动重试。
- 正常Ctrl+C会释放锁。只有旧进程已退出、存在该臂 `checkpoints/last.pt`、代码/环境/输入/配置未变时，显式从上一完整轮恢复；当前未完成轮会重算，不是逐batch恢复：

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' -u -B scripts/train_fixed_threshold.py --config runs/training/yaw-loss-ab-20261019-v2/W0/config.json --resume
```

W1中断只换W1配置。不可将 raw 的last用于wrapped，也不可复制其他臂的last。没有last、断电残留锁或目录半成品时先反馈，不删锁/删目录；由Agent检查PID和身份后决定。重跑必须新任务目录与新准备清单，不能覆盖v1/v2、旧R1/R2、原父权重或任何原评测。

## 7. 回传与预声明验收

保留整个v2根目录；至少反馈 manifest、两臂 config、preflight-smoke、run-identity、run.log、status、history、report、8份epoch快照及2份last。有异常同时提供最后20行日志与锁内容。

Agent先核验全部身份、四轮完整、权重有限、模式正确、阈值未变、未部署。raw与wrapped的loss数值不可直接当性能对比（目标函数已变）。审核后另给同一69条回归的人工评测计划，不在本计划里自动串跑。

后续八组回归预声明：比较同epoch的W1与W0；成功数至少 `max(36, W0成功)+3`；碰撞数至多 `min(27, W0碰撞)-3`；保留至少33/36条原成功；每场景原成功丢失≤1。任何候选必须同时满足，不挑不同轮拼指标。四轮全失败则停止此单变量，不加轮/调阈值凑结果。通过后还要独立选型300条及原主协议验证；**最终仍要求同一候选成功率≥70%、碰撞率≤10%，69条回归不是完整集成绩。**

## 8. 运行结果（2026-10-19 10:50–11:03，Agent 代跑）

- 两臂各 4 轮 `completed`：W0 377.0 s、W1 373.0 s（合计 12.7 分钟，落在 12～25 分钟估算内）；无 `run.lock`、无 `deployment.pt`、未校准未评测。
- 身份核验：两臂 config 只在 `arm`/`output`/`trainer.checkpoint_dir`/`trainer.yaw_error_mode` 上不同，manifest 摘要、父模型、数据、阈值与课程一致。
- 关键隔离证据：W0 四轮权重与上一轮 R2 同轮权重逐参数完全相同，raw 严格复现旧结果；W0/W1 同轮权重互异，8 份快照有限且模式记录正确（raw/wrapped）。
- 指标：W1 val_loss 1.587169/1.506387/1.422938/1.397083，目标函数已变不能与 W0 比较；自由滚动 val ADE 的 W1−W0 为 +0.00670/+0.00167/+0.00008/+0.00249 m（四轮均略高，据 §9 更正），说明该损失修正对开环几乎无影响，判定必须依赖闭环。
- 逐轮指标、快照摘要与结论边界见 `runs/training/yaw-loss-ab-20261019-v2/summary.md`。
- 下一步：另交同一 69 条回归的人工评测计划（比较同 epoch W1 vs W0，门槛见上节），由用户决定是否执行；本页不授权自动串跑评测、校准或部署。

## 9. 独立复核与更正（2026-10-19）

重新加载8份epoch与2份last、核对冻结输入/代码/环境/配置、历史和日志，审核通过。W0每轮严格复现旧R2权重；W0/W1模式分别raw/wrapped，优化器步数400/800/1200/1600，阈值约0.26未变，无自动部署或评测。

更正§8的ADE差值说明：W1−W0逐轮为 +0.00669754/+0.00167325/+0.00008267/+0.00248624m；第一轮超过0.003m，因此不能表述全部差≤0.003m。四轮W1的ADE均未改善；不以开环ADE或跨目标函数loss推断闭环收益。

独立审核文件：`runs/training/yaw-loss-ab-20261019-v2/agent-training-audit-20261019.json`。下一阶段为 [LT-20261019-03人工闭环配对](LT-20261019-03-yaw-loss-regression.md)，103个冻结文件、8组×69条，尚未启动。
