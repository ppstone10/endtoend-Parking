# LT-20261020-12：冻结轨迹生成的停止头双臂训练

## 状态与问题

**已运行完成（2026-10-20 13:14–13:15，Agent 代跑并已核验）：训练与诊断完成，父基线 1144/920/336，guided 774/1290/336、free 184/1880/336，两臂均未超过父基线，free 明显劣于 guided；停止扩大。** 独立代码、30项相关测试和8样本×每臂1 epoch预检完成；原模型/训练器/数据/控制器源码未改变。本页只运行训练与缓存上的原训练域诊断，不串行启动闭环评测、选型或部署。

LT11原train有920条停点错位、336条整段没有联合达标点。这里检验：在轨迹生成完全不变时，停止头改用自由滚动特征训练，是否优于专家点引导特征训练？不预设其能改善碰撞。336条轨迹问题是本试验不能修好的边界。

## 对照、固定参数与研究边界

父模型R0另作未训练基线，两条训练臂都从同一个父停止头开始：

| 项目 | guided臂 | free臂 |
|---|---|---|
| 缓存解码输入 | 完整专家点引导，tf=1 | 不输入专家点，tf=0 |
| 更新参数 | 只更新decoder.stop_head的weight/bias | 相同 |
| 优化 | Adam lr2e-4、梯度裁剪1、20 epoch、batch8、seed20261020 | 相同 |
| 目标 | 原专家有效长度的累计停止BCE，系数0.2、不做类别平衡加权 | 相同 |
| 阈值/控制 | 继承父阈值约0.26、原MPC | 相同 |

固定BEV编码、条件层、GRU和point_head，停止头共129参数。两臂同一每epoch随机排列、同样20轮，不选择最好epoch，固定比较最后一轮；每轮checkpoint均保存。guided是全引导机制对照，**不是重现父模型历史的混合teacher forcing日程**。两臂都均匀打乱旧2918池，不使用父训练器的恢复三池均衡采样；只有guided与free之间的差用于隔离引导策略，任一臂与未训练R0之差还包含头再训练等变化。原训练器会给引导分支提供补齐至320的目标，真值长度之后专家点为补齐零值，此处沿用该语义，不偷偷另定尾部保持策略。

不改轨迹点MSE/航向权重/阈值，不采新专家标签，不把模型的oracle首联合点变成监督。完全冻结特征且tf0/1无抽样，故各臂可缓存一次隐藏状态，再仅训练线性停止头；此等价性有直接logit核验，不能将缓存方法用于继续训练GRU。

[Scheduled Sampling原文](https://arxiv.org/abs/1506.03099)支持调查引导与自由生成的分布差别，[PyTorch冻结输出层示例](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)支持冻结主干仅更新头的实施方式；不证明本任务有效。无新依赖、无外部实现拷贝，使用现有torch2.13.0+cpu；官方网页2.14不等于更换本机版本。

## 输入与冻结身份

目录：`runs/training/frozen-stop-head-ab-20261020-v1/`，配置SHA256：

`3dd49985a2d81ac95fe2b88c522db03122f1d01a0f0a2d70186a49f7db252002`

冻结11输入/101执行源与环境，旧99源逐项未改；输入包括LT11既有5输入，以及既有2918训练池、val/selection身份隔离文件、LT11 config/report/原始预测NPZ。具体路径和SHA以config为准，不手改或重新prepare。

- 父模型SHA：`d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`。
- 训练池：`data/task_dataset/v40_dagger_collision/train-balanced.npz`，SHA `b9531bbfc3577b9a8e7db3e622cb2578fc0d3cb926fc792dfa0eadb5cf2cb8cd`，2918=原任务2400+既有恢复518；原任务为前2400，原始BEV/goal/state/有效轨迹逐元素匹配LT11原train，不新增或重标注任何恢复样本。
- 原train SHA：`8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f`。
- LT11 report SHA：`c6a99f711603dabf691ff3155adea59ee85797ea8f6ec996c0a4ea22c23cf09b`，预测NPZ SHA：`9f7f46f7acef0be595ce2cbb548200967e1a38e943d9a485ccfef7f34082110b`。

训练来源与val/selection任务ID及恢复source_task_id交集均为0，无S9；val/selection仅用于准备时身份隔离，不参与优化、校准、模型选择或本次指标推理。既有恢复标签没有因本实验获得新的控制可执行认证；不绕过#1298/#1455的兼容性门槛。

归档BEV与原局部坐标转换不变，不按模型geometry元数据重新生成输入。H320/dt0.2、CPU单线程固定。正式free缓存前2400的完整points/logits/target/mask/goal必须逐元素复现LT11，父基线统计也须一致，否则立即停止。

## 已验证与未验证

30项相关测试最终通过：新增9项覆盖缓存/直接logit、引导/自由特征差异、hook异常清理、原累计BCE等价、训练确定性与源参数不变、只改变头且完整点不变、导出重载、来源索引、身份漂移及旧输出拒绝；另覆盖终点审计/环绕航向/模型变体/停止校准回归。

真实预检4.826秒：固定索引1298/1455/1559/1441/1334/1680/2400/2917，共8条，包含6原任务与2既有恢复；两臂各1 epoch。缓存[8,320,128]及摘要、导出checkpoint摘要/冻结主干、原来源索引独立复核通过。两臂只改变weight/bias，重载后全部非head张量相等、8输入的完整320点torch.equal，129参数确认。三类统计父/两臂都是1/4/3，**没有提升证据，不比较两个不同输入分布上的训练loss判胜负**。见[预检复核](../../runs/training/frozen-stop-head-ab-20261020-v1/preflight/agent-audit.md)。

正式2918缓存/20 epoch、全原2400基线复现及训练后统计尚未执行。8条预检不能替代这些验收；不再跑smoke。

## 环境、资源与进度

唯一Python：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12.13/NumPy2.5.2/torch2.13.0+cpu；无需GPU或安装依赖。运行期间不编辑代码/数据、不并行训练或更换环境。

预计2～15分钟，基于LT11 2400自由推理16秒、本轮2918双特征缓存及20轮头训练/约1GB缓存写盘，具体IO和CPU速度不确定；**只是估算**。建议6GB可用内存、3GB磁盘，非实测峰值。

进度依次为CACHE guided、CACHE free（每25 batch打印一次），然后TRAIN guided epoch=1/20…20/20、TRAIN free同样20轮，最终COMPLETE training_complete_requires_independent_review。guided训练特征loss与free训练特征loss不可直接当统一评测指标比较。

## PowerShell：同一窗口依次执行

### 1. 只读身份预检

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$stopHeadPython = 'D:\conda\envs\endtoend-parking\python.exe'
$stopHeadRoot = 'D:\program\endtoend Parking\runs\training\frozen-stop-head-ab-20261020-v1'
$stopHeadConfig = Join-Path $stopHeadRoot 'config.json'
$stopHeadHash = '3dd49985a2d81ac95fe2b88c522db03122f1d01a0f0a2d70186a49f7db252002'
if ((Get-FileHash -LiteralPath $stopHeadConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne $stopHeadHash) { throw '配置漂移，停止反馈' }
& $stopHeadPython -u -B scripts/train_frozen_stop_head.py --config $stopHeadConfig --config-sha256 $stopHeadHash --check-only
if ($LASTEXITCODE -ne 0) { throw '身份预检失败，不继续' }
```

应显示INPUT/CODE/RUNTIME VERIFIED; NO TRAINING STARTED。

### 2. 正式训练与原训练域诊断

```powershell
if (Test-Path -LiteralPath (Join-Path $stopHeadRoot 'results')) { throw '已有正式输出，禁止覆盖或重跑，先反馈' }
$stopHeadLog = Join-Path $stopHeadRoot 'console.log'
if (Test-Path -LiteralPath $stopHeadLog) { throw '已有日志，禁止覆盖，先反馈' }
& $stopHeadPython -u -B scripts/train_frozen_stop_head.py --config $stopHeadConfig --config-sha256 $stopHeadHash 2>&1 | Tee-Object -FilePath $stopHeadLog
if ($LASTEXITCODE -ne 0) { throw '失败/中断，保留所有产物，不自动重试' }
```

不要运行旧审计/评测命令替代本入口。新增执行源使旧代码全集身份校验拒绝重跑属保护，不改旧hash绕过。

## 产物与完成标志

`results/identity.json`、`cache-guided.pt`、`cache-free.pt`、`parent/samples.json`与stop-logits.npz；guided/free各自epoch-0001.pt…epoch-0020.pt、samples.json、stop-logits.npz；根report.json含所有摘要/训练history/冻结审计及三方统一自由滚动评估。

缓存是中间特征，不是新增可训练标签或可恢复Trainer断点；checkpoint为模型加载兼容的独立实验产物，resumable=false、optimizer_state=null，记录父源epoch6与head适配epoch、未重校准的继承阈值。最后一轮固定为两臂epoch-0020.pt，不据训练指标挑某一轮部署。

成功完成应为status=completed、smoke=false、training_samples=2918、metric_samples=2400，两臂history各20项、checkpoint指向epoch-0020.pt；integrity全部非head相等、probe_points_bitwise_equal=true、head_parameters=129，统一自由预测的no_joint_point仍336。训练完成不等于性能门槛通过。

异常保存status.json、已完成epoch及可能的.partial缓存，不删目录/日志/模型，不自动重试或resume、不换样本/阈值/预算。把console末尾和status反馈；Agent检查后另定新目录/计划，不在旧目录拼接不同身份。

## 回传后验收

先核对输入/代码/样本/缓存/所有checkpoint与冻结主干，确认原2400轨迹和父基线复现，再比较**同一free特征**上的停止/联合到达指标、原开放停点合格保留与分场景回归。必要时只在已有数组上分析头的变化，不把原训练域拟合当泛化或安全。

若free未优于父基线/同预算guided，或有明显原成功退化，则停止扩大、分析原因，不扫epoch/阈值。即使开环改善，也只允许另行准备配对闭环长任务，不自动运行；必须验证原成功保留、碰撞不增后再谈≥70%/≤10%主验收。无联合点336条需另做轨迹/终端联合对齐，不能被本实验洗成通过。

本任务已完成并停止扩大，本页命令不再执行、不重新prepare或挑中间轮。历史回传口令为“停止头双臂训练完成”。

## 独立复核与停止判断

见[agent-audit.md](../../runs/training/frozen-stop-head-ab-20261020-v1/results/agent-audit.md)：11输入/101源/环境、两缓存5836来源记录、40个checkpoint、父2400轨迹/标签对LT11逐元素复现及三方指标重算通过。仅停止头变化不是口头保证，两最终模型重载首8完整轨迹及logits也逐元素相符。

候选不通过：原1144条开环合格，guided保留534、free保留63。free虽使同free特征平均BCE0.124400→0.070340，却让末点前继续标签BCE0.050981→0.104846、单点越阈率4.15%→9.77%，更多早截断；不能仅归因于阈值、据loss下降宣布成功或用收车描述未运行的闭环。下一阶段先研究停止目标/聚合与首次事件的对齐，尚未实施新loss，不扫描阈值/epoch或启动新训练。

## 运行结果（2026-10-20 13:14–13:15，Agent 代跑）

- 只读预检通过；正式运行 67.1 s，`status=completed`、`smoke=false`、`training_samples=2918`、`metric_samples=2400`；两臂各 20 epoch、检查点固定 `epoch-0020.pt`，`all_non_head_tensors_equal=true`、`probe_points_bitwise_equal=true`、`head_parameters=129`，`fixed_threshold=0.26000000000000006` 未重校准。
- 父基线复现 LT11：1144/920/336、early 692/late 228、`stop_not_found=0`。
- guided：**774** 达标 / 1290 错位（early 420、late 870）/ 336 无点，`stop_not_found=24`。free：**184** 达标 / 1880 错位（early 1813、late 67）/ 336 无点，`stop_not_found=0`。
- 分场景达标（父/guided/free）：S1 278/124/31、S2 266/113/24、S3 332/176/43、S4 36/42/10、S5 196/236/53、S6 30/28/23、S8 6/55/0。guided 在 S4/S5/S8 有改善，但 S1/S2/S3 大幅退化。
- 结论：free 特征明显劣于 guided（184 vs 774），两臂都低于未训练父基线（1144）——头重训后 logit 相对继承阈值漂移，按预声明不重扫阈值。训练 loss 两臂不可比（guided 0.00946→0.00780、free 0.02081→0.01637）。停止扩大：不扫 epoch/阈值、不挑中间轮、不进闭环评测或部署；336 条无联合点仍未被修好。明细见 `runs/training/frozen-stop-head-ab-20261020-v1/results/summary.md`。
