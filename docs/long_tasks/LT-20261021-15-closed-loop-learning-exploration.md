# LT-20261021-15 · 连续闭环学习探索

状态：**已运行完成（2026-10-21 10:26–10:51，Agent 代跑并已核验）**。16 单元、28 检查点全部完成、无分支失败，活动耗时 1481.9 s；裁决 `matrix_completed_no_qualified_direction_review_contrasts`——没有任何方向达标，两个诊断方向在确认集全部退化。代码/配置已准备，40项定向测试通过；最终版本完整smoke为109.939秒。它是有总预算的多对照诊断探索，不是已验证有效的正式训练配方，也不是部署。

补充独立复核：[结果、数据有效性缺口与下一方向](../../runs/diagnostics/learning-exploration-20261021-v1/independent-review.md)。两个最终方向全部未晋升；聚合每seed虽追加27帧，与原库比较的有效新增输入/标签均0，不能据此否定真正状态聚合。保留上方实际运行记录与以下原计划供追溯，无须再次启动本任务。

## 1. 目的与出口

不再“试一个方向，无收益就停”。一次运行完成五个问题的配对实验，再统一复核：更新量是否不足、执行帧是否缺覆盖、stop监督是否干扰轨迹、成功保持是否压制恢复、学习器状态标签是否比等量继续训练有效。

保留全部预声明分支：碰撞/无收益是实验结果，不中止下一臂；身份漂移、预算耗尽、全局监督不可用或运行故障才中止。无法找到方向也是合法结果，但必须给出对照证据与未识别项，不把loss下降包装成收益。

主目标仍为纯网络E5成功率≥70%、碰撞率≤10%，无安全盾/在线教师。此次小规模阶段筛选不替代主300或独立选型300，也不能证明架构有解/无解。

## 2. 唯一主要变量与实验链

唯一研究对象为“可执行监督如何转成纯网络闭环收益”；每个邻接对照只改一项，固定网络架构、原MPC、感知、lr1e-5、停止阈值、角色损失权重和任务。

| 对照 | 左臂→右臂 | 仅改变什么 | 判读重点 |
|---|---|---|---|
| 更新量 | sparse_joint同一路径12→60→240步 | 优化步数 | 同一缓存误差与拟合任务是否真改善，而非轮换批次loss |
| 时间覆盖 | sparse_joint240→dense_joint240 | 每源均匀4帧→全部执行帧 | 源权重不变，是否补到关键访问状态 |
| stop目标 | dense_joint→dense_parent_stop | 恢复监督stop目标改为R0软概率 | 拟合/发现闭环与原生首停变化，未声称冻结stop输出 |
| 成功保持 | dense_joint→dense_recovery_only | 移除R0保持锚点 | 恢复收益和旧成功损失的交换 |
| 学习器状态 | continued_dense→aggregated_dense | 加入一次新访问状态的严格教师标签 | 同权重/优化器起点，各额外240步，排除多训带来的假收益 |

四主臂各两训练顺序seed20261020/20261021，检查点12/60/240；两个续训臂各两seed至总480。每角色每步2帧，同角色按源轮换，每源4次后切换，避免dense把长轨迹偷偷加权。第一seed的稀疏12步保持旧采样语义，精确等价已用单测核对。所有主臂从同R0开始，续训两臂从各seed相同dense_joint240模型和优化器状态开始。

## 3. 固定数据、模型和代码

工作目录：`D:\program\endtoend Parking`。

Python：`D:\conda\envs\endtoend-parking\python.exe`，禁止C:\Python314；当前Python3.12.13、torch2.13.0+cpu、numpy2.5.2。PyTorch CPU单线程、无GPU要求。

配置：[config.json](../../runs/diagnostics/learning-exploration-20261021-v1/config.json)。SHA256：

`a41fc716c469e64098ea16ea36bb4e709145397b95e2ebdb0e4f0172d465bfe8`

`identity.json`固定此配置；配置记录109份执行代码SHA、6输入、10份旧R0状态/命令档案SHA。可人工提交现有改动，但运行期间不要编辑Python执行源、输入模型/数据，不能重新prepare原目录规避漂移检查。

| 输入 | 路径 | SHA256 |
|---|---|---|
| R0 | runs/validation/arch-ab/dep-epoch0007.pt | d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8 |
| train2400 | data/task_dataset/tracked_pivot_v8_3000/train.npz | 8313e72bca5539f2ecade47ecf159181c5da27db716ea2fbd075408599a6ba1f |
| manifest | data/task_dataset/tracked_pivot_v8_3000/manifest.json | 424fcb68cc8257707a2568c8204bf27f8db2c75c16ddca9a0752557da7ebd077 |
| R2（仅身份冻结，不参与本次拟合） | runs/training/recovery-balance-ab-20261018-v2/R2/checkpoints/epoch0002.pt | fe258ad0186a0ce40717befbc96452bd2f959ea2e97293b323aaa2a33f26b620 |

val与selection仅核验身份/来源不重叠，不用于此次梯度更新、选臂或确认评测；其SHA和完整路径见配置。

固定参数：R0 net-v1/H320/dt0.2/原stop阈值0.26000000000000006；E5模拟感知，保留原实际target输入，不切换BEV通道；原MPC dt0.1/H10/replan10/max600，接触余量0、规划余量0.2；完整回合不缩短。确认控制seed为数据索引+0/10000，配对原臂与候选。

## 4. 三组任务，严格分工

- 拟合来源固定 `[860,1975,1334,1559,2330]`：三条R0成功保持；S4 #1559持续教师、S8 #2330固定教师及最多两次既有有界补偿。完整实际到达、零接触、零原规划余量侵入才加监督；失败不得替换来源或降低标准。
- 发现探针固定 `[861,1976,1680,1441,2348]`：只跑纯网络、用于方向排序，不拟合/查询教师。原R0为2成功/2碰撞/1振荡。
- 确认60：配置预先按task_id固定哈希排序，S1/S2/S3/S5/S6各6，S4/S8各15，task/source_task与前两组隔离。原train没有S7，明确不覆盖S7；不用S9。它是本次适配/选臂隔离的训练域复核，R0可能在原始训练时见过，不能称模型未见泛化测试。

## 5. 可复制运行命令（依次执行）

正式目录已prepare，无须再创建。先只读预检：

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/explore_closed_loop_learning.py precheck --output 'runs/diagnostics/learning-exploration-20261021-v1'
if ($LASTEXITCODE -ne 0) { throw '身份预检失败，请停止并反馈，不要启动探索' }
```

预检显示`IDENTITY AND COMPLETED UNITS VERIFIED`后，启动整条探索：

```powershell
& 'D:\conda\envs\endtoend-parking\python.exe' scripts/explore_closed_loop_learning.py run --output 'runs/diagnostics/learning-exploration-20261021-v1' 2>&1 | Tee-Object -FilePath 'runs/diagnostics/learning-exploration-20261021-v1/console.log' -Append
if ($LASTEXITCODE -ne 0) { throw '探索未完整结束，请反馈report.json和日志尾部' }
```

只需运行这一条run，不要逐臂手动挑选、边看结果边改学习率/迭代数，不要同时启动第二份进程。

## 6. 时间、资源和正常进度

估计25–60分钟，当前CPU上不保证；总活动墙钟预算90分钟，含恢复前已记录时间。估算依据是完整smoke约百秒、正式2880次更新、28个检查点的280次拟合/发现回合、最多600次确认回合及采集分支；短拟合时长不能精确线性外推，CPU负载/失败规划会影响耗时。不是8组完整epoch训练。

估计内存2–6GB、磁盘预留2GB；未实测峰值，无GPU占用承诺。检查点保留模型和Adam状态，不只保存最优点。阶段日志为`BEGIN`→`CHECKPOINT`→`END`；模型优化期间可能几十秒没有新stdout，`progress.json`约每30秒更新活动时间。

完整16个单元：collect1、主臂8、聚合标签2、续训4、confirmation1。每个检查点保存5拟合+5发现的实际状态/命令。一次学习器状态聚合仅用拟合5源，严格教师不通过则记拒绝；无新增标签如实标记，仍跑等量对照，不伪称完成分布修复。

主臂ValueError类局部失败记账后继续其他臂；错误退出、身份漂移或超时不是普通无收益，不盲目继续。正常结束打印`FINISHED ...`，报告`completed`或`completed_with_branch_failures`。后者须复核缺失臂，不能当全量成功。

## 7. 输出、停止与恢复

主目录：`runs/diagnostics/learning-exploration-20261021-v1/`。

- `config.json`、`identity.json`：冻结身份；`report.json`：总状态、配对对照、原因线索、确认结果。
- `progress.json`：累计活动时间；`console.log`：人工run命令日志。
- `units/<单元>/completed.json`：全部已完成文件SHA清单；`attempt-*/payload.pt`：缓存或单元结果。
- 主/续训`step*/state.pt`含模型/优化器，`step*/report.json`含完整回合结果，NPZ含真实状态/控制。
- 失败attempt保留`error.json`，不覆盖；实验checkpoint只供诊断，不是部署格式。

需要停止时按Ctrl+C，等待进程结束；无自动续跑。再次运行时，先执行同一precheck，再执行同一run。已完成单元核验SHA后跳过，未完成单元另建attempt从该单元起点重算，**不从中途优化步恢复**，避免优化器/RNG语义错配。旧失败attempt保留供审查。

若强制关闭窗口/机器崩溃留下`run.lock`，先查看其PID并确认该探索进程确已退出。不能判断时反馈，不删锁。确认后只删除这个锁文件，再precheck/run：

```powershell
Get-Content -LiteralPath 'D:\program\endtoend Parking\runs\diagnostics\learning-exploration-20261021-v1\run.lock'
# 根据上行PID在任务管理器核对进程确已结束，确认后才执行下一行。
Remove-Item -LiteralPath 'D:\program\endtoend Parking\runs\diagnostics\learning-exploration-20261021-v1\run.lock'
```

活动预算每30秒记账，强制崩溃可能丢失最后约30秒计时。90分钟预算耗尽时不清progress绕过，应反馈已完成结果，由Agent决定是否需要新的预算/任务。输入/代码漂移、缺文件或磁盘不足同样先反馈，不能重写identity追认。

禁止覆盖原R0/R2、原train/val/selection、LT14、前次行为预检、两个smoke目录及本任务已完成产物；不删除attempt以凑出好结果，不执行git commit/push。

## 8. 统一裁决，不以单臂结束

四主臂所有预声明检查点完成后，两续训对照也完整完成。仅最终240/480点发现集排序，12/60只作剂量诊断；按两seed均通过发现门槛优先，其次净成功−净碰撞、旧成功丢失、碰撞数、方向名固定排序，取最多两个方向。

即使没有发现过线方向，也复核排序前两个，但标明“仅诊断”，不会把确认结果反过来追认发现过线或继续训练。每方向两训练seed×两控制seed，均完整60条配对。

探索阶段稳健收益要求四组合均：成功净增≥3、碰撞净减≥3、原成功丢失≤2、任一场景成功净减≤1，且此前发现门槛通过。报告始终`main_protocol_passed=false`，即使该阶段通过也须Agent复核后另定正式主验收计划，不自动部署/串行长训。

## 9. 完成后反馈与Agent责任

反馈“LT15连续探索完成”，并给出主目录；至少保留/提供：

1. `report.json`、`config.json`、`identity.json`、`progress.json`。
2. `console.log`末60行及退出是否正常。
3. 全部`units/`完成清单与checkpoint/NPZ；不用手动拷贝所有大文件，原目录保留即可。

Agent后续必须先核验身份、16单元状态、28检查点/分母/实际轨迹/训练与控制seed、候选加载复现和确认配对，再给“最有希望方向+支持它的对照+反例/适用范围+下一步”。需要区分：未学会原任务、只记住原任务未迁移、stop干扰、保持/恢复权衡、聚合无新增或新增后是否有收益。证据不足时明确未识别，不把所有失败归因容量。

不是每看到一个负结果就另起一轮询问；本计划内全链已经预声明，可一次run完成。若所有方向失败，下一步应基于整组证据重新设计，不自动无限扩展参数搜索。

## 运行结果（2026-10-21 10:26–10:51，Agent 代跑）

- precheck 通过（`IDENTITY AND COMPLETED UNITS VERIFIED`），run 一次完成：1481.9 s（约 24.7 分钟，90 分钟预算内）；`status=completed`、`smoke=false`、`branch_failures=[]`、`all_predeclared_branches_completed=true`；16 单元（collect 1 + 主臂 8 + 聚合 2 + 续训 4 + confirmation 1）、28 份 `state.pt` 齐全。
- 教师覆盖通过：3 个保持锚点、S4/S8 各有 1 条严格改善来源（拟合 5 源、发现 5 源固定不变）。
- 五对照均无正收益：覆盖 sparse→dense 两 seed 均 0/0；stop 目标 −1/+1 与 0/0；成功保持两 seed 均 **−1/−1**（去锚点丢成功）；学习器状态聚合 0/0 与 0/0（probe −1/+1）。剂量 12→60→240 无改善：240 步 fit 仍 3 成功/2 碰撞、probe 1 成功/3 碰撞，seed 20261021 的 probe 期间还丢 #861。
- 发现排序仅有 continued_dense / aggregated_dense，均 `discovery_eligible=false`；按规则仍作诊断确认（60 条 × 两训练 seed × 两控制 seed）：四组合全部退化——continued_dense Δ成功 −7/−9/−8/−5、Δ碰撞 +6/+9/+5/+2，aggregated_dense −7/−9/−9/−5、+6/+9/+8/+5，丢失原成功 6～12 条，`robust_stage_benefit=false`。
- 结论：`matrix_completed_no_qualified_direction_review_contrasts`；`main_protocol_passed=false`、`formal_training_recipe_validated=false`、无自动部署。拟合集能记住监督但发现/确认集退化，指向“记住训练状态未迁移”而非容量不足；不自动扩大参数搜索。明细见 `runs/diagnostics/learning-exploration-20261021-v1/summary.md`。
