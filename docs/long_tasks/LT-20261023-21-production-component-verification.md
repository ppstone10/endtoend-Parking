# LT21：原生E5同300任务的生产分层与耦合验证

状态：**已执行完成**（用户要求运行完整300条；1405.3秒、退出码0、inspect通过）。300条全覆盖，205成功/77碰撞/18振荡，正式资格`false`——这是模型/契约的负结果，不是脚本异常。本计划不训练、不搜索、不修改生产输入/模型/MPC、不部署。对应 `T-170000`、`VERIFY-E5-001..005`。执行结果与Agent定向诊断见文末。

## 目的与唯一主要变量

将四案例冒烟扩大到原同300完整覆盖，含205个成功对照和95个已知失败（77碰撞、18振荡），逐个定位参考、实际执行和边界问题。唯一改变是验证覆盖范围，任何生产参数均不改变。全量同一次原生运行生成L1/L2/L3和L1+L2、L2+L3，勿逐层重复仿真。已知回归集不能用于宣称新的泛化收益。

短预检的四个起点（0、163、247、292）精确复现原结果，覆盖1,288控制步、130次预测。感知/网络/MPC独立重放一致；发现130次输入的checkpoint target声明geometry而生产image，但初始BEV与val保存值全部相同，因此声明冲突尚不能证明训练部署分布错配。S4#163参考窗口危险、S8#292安全接触窗口下实际撞车、S6#247运行600步振荡，已进入分层报告。此轮不是验证一个修复候选，已知R0总体不达标仍应如实失败。

## 冻结输入和固定参数

- 配置：`runs/verification/component-v1-formal-20261023-v2/config.json`，已由prepare生成；包含基线、data、checkpoint、manifest、相关代码逐文件SHA及Python/NumPy/Torch身份。代码变动后precheck必须拒绝，不能静默重冻。
- 基线：`runs/validation/live-20261018-170806/full/E5/report.json`，原生E5、sensor/network/mpc_vehicle/safety_mode=none。
- 数据：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，SHA256 `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`；索引顺序0..299，manifest原文件只读。
- 模型：`runs/validation/arch-ab/dep-epoch0007.pt`，SHA256 `d7379852c3f07c3a2a665eda580677951eefca801190437303150a93e802a7a8`；net-v1、hidden128、max_horizon320、dt=.2、stop_threshold=.26000000000000006，声明target=geometry、height=semantic。不校准、不替换target，记录实际生产行为。
- 原完整6×3m车辆，max_v=2、max_omega=1，物理碰撞margin=0，规划余量=.2；MPC dt=.1、horizon10、每10控制步重规划，max_steps600，seed=index+0。车位终止容差由同任务复原，位置和环绕航向必须同时严格小于容差。
- 代码版本以配置的`code`为准，工作树未提交；不以当前git HEAD代替实际SHA。原LT18/19/20冻结源和产物均保持。

## 环境与可复制PowerShell命令

严格按顺序执行，每步失败即停。所有Python均为指定conda环境；不用本机Python。第一段只预检，第二段才启动长任务。

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$parkingPython = 'D:\conda\envs\endtoend-parking\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

& $parkingPython -X utf8 scripts/verify_closed_loop_components.py --phase precheck --config runs/verification/component-v1-formal-20261023-v2/config.json
if ($LASTEXITCODE -ne 0) { throw 'LT21 precheck失败，停止；不要启动正式任务或修改冻结配置' }
if (Test-Path -LiteralPath 'runs/verification/component-v1-formal-20261023-v2/result') { throw 'result已存在，禁止覆盖/自动续跑' }
```

```powershell
Start-Transcript -LiteralPath 'D:\program\endtoend Parking\runs\verification\component-v1-formal-20261023-v2\console.log' -NoClobber
& $parkingPython -X utf8 scripts/verify_closed_loop_components.py --phase run --config runs/verification/component-v1-formal-20261023-v2/config.json --output runs/verification/component-v1-formal-20261023-v2/result
$parkingRunExit = $LASTEXITCODE
Stop-Transcript
if ($parkingRunExit -ne 0) { throw 'LT21未完整完成；请反馈console.log和result/report.json，勿自行重试' }
```

运行返回0只代表采集/分析完整，模型指标失败仍会返回0，不能据退出码宣布达标。最后只读核验：

```powershell
& $parkingPython -X utf8 scripts/verify_closed_loop_components.py --phase inspect --output runs/verification/component-v1-formal-20261023-v2/result
if ($LASTEXITCODE -ne 0) { throw '产物/身份/分母核验失败，请反馈异常' }
```

三个旧验证入口默认已统一，也可将最后一条的脚本名换为`evaluate_bev_fidelity.py`、`evaluate_visited_state_trajectory.py`或`evaluate_single_cycle_tracking.py`查看对应层。它们的inspect只读同一次运行，勿分别再run。旧参数仅用于显式`--legacy-diagnostic`诊断。

## 耗时与资源

四案例1,288控制步及130预测的最终短预检约23秒，含原生完整运行、几何检查、全程独立感知/网络/MPC重放和产物写入。按同300步数线性外推仅能估量级，复杂场景射线/扫掠与磁盘有差异，保守预计 **10–35分钟**，慢机器可能更长。单进程，Torch固定CPU线程1，不依赖GPU；内存预留2–4GB，磁盘预留2GB。配置8小时是异常硬上限而非目标耗时；不能由Agent自动加预算/重启。不要并行运行另一套验证/训练以免争用资源或改变代码。

## 输出、日志与正常标志

- 控制台先`capture [n/300] #index matched`，然后`analyze [n/300] #index ...`。短暂无新行时检查进程CPU和新文件，勿重复启动。
- `result/config.json`：冻结输入/代码/参数副本。
- `result/capture/episode-*.json/npz`：300套原任务起点、每次BEV/局部预测/世界参考、所有控制窗口/进度/选中序列/rollout/真实状态及分余量事件；`replayed-report.json`：原生E5报告。
- `result/layers/{L1,L2,L3,L1_L2,L2_L3}.json`：模块结构重放、质量观测及对应原生任务结果。
- `result/report.json`：统一总报告，`summary.md`：逐任务定位，`artifacts.json`：所有交付文件SHA；外部`console.log`：PowerShell日志。
- 正常完成必须status=completed、完整300唯一索引、无复原失败、全部基线逐回合一致、capture/analyze各300。R0的production_qualified预期仍false，这是模型/契约的负结果，不是脚本异常。

## 异常、停止、恢复与保护

- traceback、基线不一致、代码/输入漂移、缺文件或预算耗尽时停止；报告为incomplete或保留running则一律无验收资格。Ctrl+C可中止，不删部分产物。
- **不支持自动恢复/续跑/覆盖**。先反馈异常，由Agent诊断。若需要重新长跑，另建新配置路径与新输出目录，并由Agent更新计划；不能手改SHA继续。不要删除旧result/console.log制造“首次运行”。
- 禁止覆盖所有`runs/validation/live-*`基线、模型/val/manifest、LT18–20/旧runs诊断和本轮smoke-v1..v6/冻结配置。prepare不启动正式回合；precheck不改配置。

## 完成后反馈与Agent验收

告知“LT21执行完毕”或异常，并保留/反馈以上固定目录，至少`config.json`、`console.log`、`result/report.json`、`result/summary.md`、`result/artifacts.json`、`result/capture/replayed-report.json`和五层JSON。逐回合JSON/NPZ全部保留用于独立复核。

Agent先核对身份、配置、日志结尾/退出状态、300分母、文件SHA和原生结果，再汇总95失败的参考/桥接/执行/非到达及分场景分布。接口声明冲突与实际轨迹危险并行保留，不能以其中一个覆盖另一个。只有具体边界差异才认定已测接口故障；全部已查局部正常仍未到达保留unresolved，不能自动归罪链接。

正式模型联合门槛仍为成功≥210/300、物理碰撞≤30/300，并要求完整身份/复现/结构与接口证据。本计划不放宽门槛，不将旧工具完成、局部低RMS、低ADE或回归子集结果追认为闭环通过。后续生产修复或新模型A/B需要单独决定，本计划不串行启动。

## 执行结果（初次分析，分类和L1归因已由末尾独立复核更正）

- 执行：`status=completed`、`mode=formal`、`elapsed_seconds=1405.3`（23.4分钟）、退出码0、`--phase inspect`通过。捕获300/300、分析300/300、300个唯一索引，逐回合`baseline_match`与`initial_dataset_bev_equal`均为true，即没有任何新的成功或失败。
- 身份：冻结配置SHA256 `a11105d267b81bc556d0b6e7f0a341efcce3448726a21178224688b043116b75` 与`result/config.json`逐字节相同；报告`identity.config_sha256`一致；137代码/4输入（基线报告、`val.npz`、`ep_dep-epoch0007.pt`、manifest）SHA在本机重算全部一致；609个登记产物SHA经inspect全量校验通过；产物220.2 MiB、`capture`下300个episode JSON+300个NPZ、`console.log`34行、五层JSON齐全。
- 任务结果：205成功（68.33%）、77碰撞（25.67%）、18振荡未到达；`task_metrics_pass=false`、`checks_ok=false`、`production_qualified=false`。与既有主成绩68.3%/25.7%一致。
- 分层契约与质量：L2（网络参考）300条契约通过，250条消费窗口无接触风险、50条观测到参考风险；L3（MPC执行）300条契约通过，223条无观测接触、77条接触安全失败；L2_L3耦合300条通过（205到达、95失败）。L1感知296条通过，4条`production_sensor_replay`失败（#78/#80/#82/#246，plan 0）。L1_L2耦合300条全部`failed_input_contract`：checkpoint声明`target_channel=geometry`而生产为`image`（7758条计划级不一致），因此逐回合`observed_signature`一律`boundary_mismatch`——这是签名覆盖，不等于失败归因。
- 95个失败的观测分层（去掉签名覆盖后）：41条`unsafe_consumed_reference`（参考本身已危险）、36条`execution_left_safe_reference`（参考安全而执行偏离）、18条`unresolved_nonarrival`（无接触振荡未到达）；0条"仅桥接危险"、0条无法解释的碰撞。分场景：S4_dump_area 34（32参考危险+2执行偏离）、S6_loading_face 23（18未到达+5参考危险）、S3_maintenance 23（20执行偏离+3参考危险）、S8_weigh_station 11（全部执行偏离）、S1_parking_lot 3（执行偏离）、S5_crusher 1（参考危险）、S2_diagonal_lot 0。S8为11/11全失败，S2为0/55全成功。
- 成功任务中的观测风险：205个成功里有9条带参考危险（S4×4、S5×2、S6×3，其中2条同时有桥接危险），成功不等于参考安全。
- 4条L1失败的定向诊断（Agent，单回合重跑+同姿态探针，均写TEMPORARY目录，不动正式产物）：四条均可稳定复现`production_sensor_replay@plan0`。其中#78/#80/#82用同一条新建管线在同姿态连续采集，结果彼此不同；改变`np.random`全局种子可使该BEV变化140–620个栅格单元。直接来源：`sensor2bev/lidar_bev.py`的`_downsample`在ROI点数超过`max_points`时调用未播种的全局`np.random.choice`，叠加传感器自身带噪逐次采集流。#246可稳定复现但与全局种子无关（3个栅格差），根因未定。结论：这4条是"同位姿逐位重放"契约对随机感知过度严格所致；4条任务仍成功，不能当作生产感知故障，也不能用`boundary_mismatch`解释任何失败；#246需单独诊断。**本条归因已被下文「撤回『全局随机降采样导致4条L1差异』」更正**：360束远低于`max_points=10000`，从未触发降采样，实为分析器以NumPy float64重放Python float位姿的类型不保真，四条（含#246）均已定位。
- 判读与后续：模型联合门槛未通过（68.3%<70%、25.7%>10%），与主成绩一致；分层证据把失败集中到"参考已危险"（41条，S4为主）、"参考安全但执行偏离"（36条，S3/S8为主）和"无接触未到达"（18条，S6）三类边界。本计划不修复、不训练、不晋升、不串行启动后续；生产修复或新模型A/B需另行决定。

## 独立复核更正（2026-10-23）

本节取代上述初次分析的事件分类和4条L1差异归因，保留原正式产物，不追改报告或冻结代码。

- 本轮重新inspect通过，609产物/137代码/4输入身份有效；300个原生回合逐字段精确复现、76,282控制步和7,758预测完整。最终205成功/77碰撞/18振荡不变。本次未新增完整回合或训练。transcript正常结束但没有记录原生stdout/历史退出值，完成证据以完整结果及本次inspect为准。
- 77次碰撞从归档NPZ独立复原任务并重算真实矩形扫掠，全部一致。**碰撞当步**为35条危险参考、42条参考及桥接接触安全但实际执行碰撞；另18条S6无接触未到达。35条来自S4×30、S6×4、S5×1；42条来自S3×23、S8×11、S4×4、S1×3、S6×1。
- 失败中41条“曾有参考危险”是另一种全程观测，其中6条最终在安全接触参考窗口下撞车；按全程历史风险优先级得到41/36，不能将其冒认为碰撞当步分类或唯一因果。总报告observed_signature的boundary_mismatch也不作为排他根因。
- **撤回“全局随机降采样导致4条L1差异”的解释。**四任务生产初态的Python float重放均逐位等于归档；将相同数值换成NumPy float64才出现6/12/6/3个张量元素差异。两种类型各自对全局seed0/1都不变，16次探针numpy.random.choice调用为0（360束远低于max_points10000）。四条均已定位为分析器初始位姿标量类型不保真，含#246，不是生产感知故障；后续不能简单全程强转float或放宽容差。
- L2/L3独立重放及已查坐标、dt、参考窗口/进度/状态连续接口一致；L2参考安全、L3执行安全和S6策略收敛已分别出现负结果，不能解释成所有局部通过后的单一链接问题。target声明冲突仍需核训练实际语义，不能盲目改生产输入。

详细证据及处理顺序见[独立复核结论](../../runs/verification/lt21-independent-review-20261023/review.md)、[结构化重算](../../runs/verification/lt21-independent-review-20261023/report.json)和[L1标量探针](../../runs/verification/lt21-independent-review-20261023/l1-scalar-probe.json)。此次只完成诊断及文档更正，没有修复或晋升停车策略；无需重跑LT21。
