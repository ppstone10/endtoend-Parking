# LT-20261020-14：固定旧候选的完整闭环收益裁决

状态：**已运行完成（2026-10-20 21:22–21:34，Agent 代跑；22:20起独立复核通过）**。两臂各300完成，联合收益+13成功/−6碰撞成立，但未达预声明阶段线与选型门槛，S4/S5退化。最终决定：[不晋升、不扩大主300、不继续同配方追加训练或参数搜索](../../runs/validation/closed-loop-benefit-20261020-v1/results/independent-review.md)。本任务结束，不重跑；以下命令仅作已完成任务记录。

## 1. 目的与本次路线调整

只回答一个问题：旧R2第2轮相对原R0的局部净收益，能否在完整独立选型300条中成立？唯一变量为checkpoint；不加新损失、不调停止阈值、不替换控制器、不使用安全过滤/专家回退。

旧69条回归：R0为36成功/27碰撞/6振荡，R2e2为42/24/3，净+6成功/−3碰撞，但损失5条S4旧成功，**旧筛选失败仍成立**。本次是在用户要求“以闭环收益为出口”后，明确重新声明的一次完整独立测量，不把旧失败追认通过，不改旧计划历史，也不把69条的偏失败样本比例冒充全量表现。

停止继续停止头数学支线；不同时重训或测试W1/其他epoch，不边跑边挑参数。只跑本候选一次。结果只可能为：选型集联合过线、阶段联合收益但未过线、没有足够联合收益；后者结束该候选，不再追加局部优化。任何场景退化须单列审查，不自动晋升。

## 2. 冻结输入与协议

工作目录 `D:\program\endtoend Parking`；Python固定 `D:\conda\envs\endtoend-parking\python.exe`。

任务目录：`runs/validation/closed-loop-benefit-20261020-v1/`。

配置：`config.json`，SHA256 **`2119619d45d7ff2b974b1187c106e83ed15ddbad4d3054e14af48932adbe555f`**。

| 输入 | 路径 | SHA256 |
|---|---|---|
| R0父模型 | runs/validation/arch-ab/dep-epoch0007.pt | d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8 |
| 固定R2e2 | runs/training/recovery-balance-ab-20261018-v2/R2/checkpoints/epoch0002.pt | fe258ad0186a0ce40717befbc96452bd2f959ea2e97293b323aaa2a33f26b620 |
| 独立选型300 | data/task_dataset/independent_selection_v1/selection.npz | 797e9e5d0d02576c45df65c74f142cc05e83770eb6bbfcc28be560dbc722d51d |
| 选型manifest | 同数据目录manifest.json | a4c804ec327220d3698ade8ec891c43295a408a51d9afa470470f640ab0a1a5b |

其余来源训练报告、训练NPZ、旧失败摘要及104份执行源/环境身份见配置，共7项输入。两个模型配置完全一致，net-v1/hidden128/H320/dt0.2，停止阈值固定`0.26000000000000006`。数据/恢复来源身份交集为0；同场景大类任务，只用于选型，不是未触碰的最终测试，不使用S9。

每臂索引完整0～299，E5模拟传感器→网络→原MPC→车辆；控制seed=0（runner实际MPC seed=0+任务索引），replan_every=10，max_steps=600，MPC dt0.1/horizon10。保留原任务容差/车体外廓/真实接触定义与E5感知构建方式，不根据checkpoint的geometry声明切换BEV通道。不允许中途修改代码、模型、数据、阈值、控制器、步数或任务集合。

原独立选型基线是200成功/85碰撞/15振荡，不是原主300的205/77/18。本次重新跑两臂以确保同一代码/环境；若R0与存档不符，反馈完整报告，Agent先检查可比性，不能直接把差异算候选收益。

## 3. 预检证据

- `tests.test_closed_loop_benefit`与`tests.test_independent_selection`共28项通过。
- `--check-only`通过，冻结104源/7输入；正式results目录不存在。
- 每臂固定S3/S4/S8来源110/162/289、每条最多20步的真实smoke完成，评测内计时1.453秒。两臂均0成功，R0一次碰撞而候选超时；明确`bounded_smoke_not_formal_evidence`，不算联合收益。
- 预检报告 `preflight/report.json` SHA256 `35a2daf8637fe899c21e447d3c70a4e91196be78c7c979b992341b7086aea216`。不得重跑/覆盖preflight。

## 4. 预计耗时与资源

只跑2×300=600回合，无训练。按旧69条每组约70～87秒及原完整300约868秒估算，CPU单线程约**10～35分钟**；失败回合可能达到600步，实际耗时不确定，不能由20步冒烟外推全程。内存预计1～4GB，产物预留500MB；均为估算。关闭其他训练/评测，不并行开重复作业。

## 5. 按顺序运行

不要重新prepare。先保存其他文件改动，运行期间不要改执行源。

### 5.1 只读检查

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$taskPython = 'D:\conda\envs\endtoend-parking\python.exe'
$taskConfig = 'D:\program\endtoend Parking\runs\validation\closed-loop-benefit-20261020-v1\config.json'
$taskSha = '2119619d45d7ff2b974b1187c106e83ed15ddbad4d3054e14af48932adbe555f'
& $taskPython -B scripts/validate_closed_loop_benefit.py --config $taskConfig --config-sha256 $taskSha --check-only
if ($LASTEXITCODE -ne 0) { throw '身份预检失败，停止并反馈，不修改SHA强行运行。' }
```

应看到`INPUT/CODE/RUNTIME VERIFIED; NO ROLLOUT STARTED`。

### 5.2 正式一次性对照

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $taskPython -B scripts/validate_closed_loop_benefit.py --config $taskConfig --config-sha256 $taskSha 2>&1 | Tee-Object -FilePath 'D:\program\endtoend Parking\runs\validation\closed-loop-benefit-20261020-v1\console.log' -Append
if ($LASTEXITCODE -ne 0) { throw '正式评测异常，停止并反馈status.json与日志尾部。' }
```

正常进度为`ARM baseline BEGIN samples=300 smoke=False`、逐任务`baseline [n/300]`，随后candidate同样300条，最后`COMPLETE`和结论JSON。600步失败会慢些，不据短期无输出启动重复作业。

### 5.3 完成检查

```powershell
Get-Content -LiteralPath 'D:\program\endtoend Parking\runs\validation\closed-loop-benefit-20261020-v1\results\report.json' -Raw
```

须`status=completed`、`smoke=false`、两臂`samples=300`、config SHA正确。`selection_target_met`只表示选型集门槛，不等于主协议已达标。

## 6. 异常、恢复与禁止覆盖

- 输入/执行源/环境SHA不符：立即停，反馈，不自行改配置或摘要。
- Python报错/中断：停止，不自动重试。反馈`results/status.json`与日志尾部。完整臂已保存，未完成臂没有逐回合恢复点。
- **只有确认原进程已退出且没有run.lock时**，可在Agent核对后显式恢复；它只复用SHA与冻结协议核验通过的完整臂，未完成臂从该臂开头重跑，不重复完整臂：

```powershell
& $taskPython -B scripts/validate_closed_loop_benefit.py --config $taskConfig --config-sha256 $taskSha --resume 2>&1 | Tee-Object -FilePath 'D:\program\endtoend Parking\runs\validation\closed-loop-benefit-20261020-v1\console.log' -Append
if ($LASTEXITCODE -ne 0) { throw '恢复失败，停止并反馈，不继续重试。' }
```

- 强制杀进程遗留run.lock：不要直接删锁/清理results，先反馈确认无活动进程。报告未登记/缺失/被改写时拒绝恢复，不自动采用孤立报告。
- 已completed拒绝再次运行。禁止覆盖旧训练、69条回归、主300、独立选型数据、当前config/preflight或已有正式报告；不调整参数后用同一目录重跑。

## 7. Agent验收与明确出口

先核验config/代码/输入身份、两臂报告SHA、完成状态、完整300分母、零复原失败、逐任务身份/容差及相同控制协议；重算成功/碰撞/全部失败构成、新增与丢失成功、配对转移及七场景结果，并核查R0对存档的一致性。单seed不宣称统计显著或训练稳定性。

1. **选型联合过线：** 候选成功≥210、碰撞≤30；只报选型过线，且报告是否真有相对R0收益及场景退化。另拟原主300与后续多控制seed计划，经人工审查后才能继续，不自动部署。
2. **阶段收益、未达最终标准：** 相对同批R0成功增加且碰撞减少；预声明有意义线为至少+9成功、−9碰撞（各3个百分点），不是下调70%/10%标准。小于该线标记小收益不足。任何S3/S4/S8或其他场景净退化必须单列，不能靠总体掩盖，不自动晋升。
3. **没有联合收益：** 成功未增加，或碰撞未减少，判该候选结束；碰撞换超时不算成功改进。停止该旧恢复续训候选，不再添加loss/梯度/步长分析给它续命。失败只否定该候选及现有证据，不宣称整个net-v1架构不可能达标。

最终仍只有原主300同一候选成功≥70%、碰撞≤10%同时满足，才能报本阶段达标。当前主成绩保持68.3%/25.7%，本次准备不产生新成绩。

## 8. 完成后反馈

回复“LT14闭环收益对照完成”，保留并提供此目录：

- `config.json`、`console.log`；
- `results/identity.json`、`status.json`、`baseline.json`、`candidate.json`、`report.json`；
- 若异常，反馈退出信息和日志最后约50行，不启动下一阶段。

## 运行结果（2026-10-20 21:22–21:34，Agent 代跑）

- 5.1 检查通过；5.2 一次性对照完成（约 11 分钟，基线 337.8 s + 候选 326.1 s）；两臂各 300/300、复原失败 0、`status=completed`、`smoke=false`；config/模型/数据/协议身份与计划一致。
- 基线 **200 成功 / 85 碰撞 / 15 振荡**，与独立选型存档一致，可比性成立；候选 **213 / 79 / 8**。
- 差值：成功 **+13（+4.33 pp）**、碰撞 **−6（−2.0 pp）**、振荡 −7；保留 180 条原成功、丢失 20、新增 33。`joint_gain=true`、`meaningful_joint_gain=false`、`selection_target_met=false`，裁决 `small_joint_gain_not_sufficient`。
- 门槛：选型需 成功≥210 且 碰撞≤30（213/79，碰撞差49条）；意义线需 ≥+9成功且 ≤−9碰撞（+13/−6，碰撞改善少3条）。71.0%/26.3%仅为独立选型结果；本候选未运行原主300，不能称为主协议成绩。
- 分场景：S3 +12/−12、S6 +8/−1、S1 +3/−3、S8 +1/−1、S2 持平（换 1 条）；**S4 −9/+9（丢 11 条原成功、只换回 2 条）、S5 −2/+2（原 0 碰撞变 2）**，`scene_regressions=["S4_dump_area","S5_crusher"]`、`requires_scene_review=true`。
- 结论：按第 7 节第 2 类出口——有阶段联合收益、未达最终标准，不晋升、不部署、不调阈值、不自动开训。S4/S5 退化必须单列；单 seed 不宣称统计显著。明细见 `runs/validation/closed-loop-benefit-20261020-v1/results/summary.md`。
