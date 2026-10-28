# LT23：终端朝向接管的同300验证

状态：**已执行完成**（用户要求运行正式同300；485.4秒、退出码0、inspect返回`completed_and_verified`）。255成功/0碰撞，双基线零丢失、#168恢复、+12新成功，`candidate_qualified=true`、`production_qualified=false`；明细见文末「执行结果」。148个执行源、20个输入冻结；固定六例短检、独立证书审计、正式配置precheck此前已通过。任务T-233422，契约TERM-CAPTURE-001..005。

## 目的和唯一主要变量

相对LT22仅增加显式终端朝向模式：每次仍消费原约束MPC求解；当同一引擎的原位置误差严格小于该车位原`tol_pos`、航向尚未小于原`tol_yaw`，且从当前位置到目标航向的完整最短转向角程车体扫掠无物理接触，才以零线速度和原限幅角速度转向。否则使用原约束解。每步重新验收位置与航向，不锁存位置放宽，不改网络/阈值/轨迹过滤，不加平移、专家或目标权重。0.3m/10°只适用于其中212条，不能作为全300的统一阈值。

范围仍是**仿真完整几何辅助的条件实验**，不是BEV占用约束接入。只适用于当前允许原地转向且无加速度约束的差分驱动仿真。production_qualified固定false；不部署、不训练、不产出教师银行，不从已知300回归集宣称泛化。

## 短检证据

- 固定索引0/123/166/168/247/292，12.135秒。LT22的3成功/0碰撞→5成功/0碰撞，新增#166/#168且原生/LT22成功无丢失。余量侵入1191→375步。
- #166在206步到达、#168在183步到达，两例实际零碰撞/零0.2m余量侵入；#0/#123/#292保持原结果。#247仍600步振荡，未宣称修复远端S6。
- 独立审计重建1920个实际动作以及5个终端接管步的限幅序列/完整转向证书，精确一致；非终端CEM未再次独立复算。78项聚合测试通过。
- 短检：`runs/verification/terminal-capture-pilot-20261023-v2/smoke/`；审计：`runs/verification/terminal-capture-audit-20261023/report.json`。首个不带v2的pilot只有旧版prepare配置，没有运行，保留但不作为证据。

## 固定输入、代码和参数

- 配置：`runs/verification/terminal-capture-formal-20261023/config.json`，实际执行148源/20输入逐文件SHA，含原模型/训练谱系/数据、LT21原生证据、LT22配置/报告/604产物清单及新短检/审计；以配置SHA登记为准，当前工作树未提交。
- 模型：`runs/verification/input-contract-corrected-20261023/deployment-image-contract.pt`，SHA `e917fed2ee965de2b497f744a8cd0f398bf85787c87176ab90f12aafc0973009`。仅target声明校正的旧权重，net-v1/hidden128/max_horizon320、NN dt=.2、stop_threshold=.26000000000000006，仍实际image/semantic输入。
- val：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，SHA `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`，manifest不变；固定索引0..299。
- 6×3m原车体、max_v2/max_omega1、控制dt=.1/H10、原population80/iterations5、候选最多16个原样本、seed=index、replan_every10、max_steps600。原位置/环绕航向双阈值严格小于，同时成立才成功。
- 实体接触仍为原零余量矩形扫掠；规划余量0.2m单列。新模式完整转向证书使用同一实体检查器，不能把零接触当作0.2m余量通过。
- 缓存原生205/77/18与LT22的243/0/57两套同300逐回合基线，不重跑两套基线。原生保留是205个成功；还要求保持LT22的243个成功，不能以新增成功抵消回退。

## 可复制PowerShell命令

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$parkingPython = 'D:\conda\envs\endtoend-parking\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

& $parkingPython -X utf8 scripts/compare_terminal_capture.py --phase precheck --config runs/verification/terminal-capture-formal-20261023/config.json
if ($LASTEXITCODE -ne 0) { throw 'LT23预检失败，停止；不得重新冻结SHA绕过' }
if (Test-Path -LiteralPath 'runs/verification/terminal-capture-formal-20261023/results') { throw '正式结果目录已存在，禁止覆盖或自动续跑' }
if (Test-Path -LiteralPath 'runs/verification/terminal-capture-formal-20261023/console.log') { throw 'console.log已存在，请反馈异常，禁止覆盖' }
if (Test-Path -LiteralPath 'runs/verification/terminal-capture-formal-20261023/python.log') { throw 'python.log已存在，请反馈异常，禁止覆盖' }
```

```powershell
Start-Transcript -LiteralPath 'D:\program\endtoend Parking\runs\verification\terminal-capture-formal-20261023\console.log' -NoClobber -ErrorAction Stop
& $parkingPython -X utf8 scripts/compare_terminal_capture.py --phase run --config runs/verification/terminal-capture-formal-20261023/config.json 2>&1 | Tee-Object -FilePath 'runs/verification/terminal-capture-formal-20261023/python.log'
$parkingRunExit = $LASTEXITCODE
Write-Host "LT23 RUN EXIT $parkingRunExit"
Stop-Transcript
if ($parkingRunExit -ne 0) { throw 'LT23执行不完整，请反馈日志与部分产物，不要自动重试' }
```

最后只读核验：

```powershell
& $parkingPython -X utf8 scripts/compare_terminal_capture.py --phase inspect --source runs/verification/terminal-capture-formal-20261023/results
if ($LASTEXITCODE -ne 0) { throw 'LT23产物或联合裁决未通过完整性核验，请反馈异常' }
```

退出0表示执行完整，不代表质量通过；正式不调用只接受固定短检的`--phase audit`，Agent后续另按预算核验实际动作。禁止在用户运行期间改代码或并行启动训练/重复评测。

## 耗时、资源与正常进度

单进程CPU，Torch线程1，无需GPU，预留2–4GB内存、2GB磁盘。LT22同300耗时9.5分钟；新固定六例1920步约12.1秒，终端会提前结束部分案例但全量收益未知，估计**5–20分钟**，磁盘/CPU差异可能更慢；硬预算8小时为异常上限，不是预期耗时。

正常输出`[terminal n/300] #index success/oscillation/...`，每回合落JSON/NPZ，最终生成completed汇总和产物清单。较长时间不输出可查看进程与文件增长，勿启动第二个作业。

## 输出、完成标志及验收

输出目录`runs/verification/terminal-capture-formal-20261023/results/`：config.json、report.json、summary.md、artifacts.json、control/raw-report.json及300套episode-XXXX.json/npz；同级console.log/python.log。逐帧保存选中标签、原/新控制成本、terminal_certificate、真实动作/状态、参考、时域/实际接触与余量声明。

完整条件：completed/formal、300唯一索引、无场景复原失败、300套账本、全部SHA不变、inspect退出0、退出码0。Agent先核验任务身份/日志/完成状态/分母/文件，再分析指标。

条件实验候选资格：成功≥210、碰撞≤30；**原205成功与LT22的243成功均不得丢失、#168必须恢复、相对LT22有新增成功、实际碰撞为0、余量侵入合计≤17,337步**。现有pilot_eligible要求零碰撞比基础门槛更严，不事后放宽。分别报告新成功、丢失、S4/S6/S8、航向/位置、终端接管/零动作和按步数归一化余量。新增成功不能掩盖任何基线回退，安全停车不算到达。

即便candidate_qualified=true，仍production_qualified=false。BEV来源约束、原18条S6未到达、剩余危险参考、独立泛化和实车动力学均未被此次候选完整解决。

## 停止、恢复、重跑与保护产物

预检失败、traceback、预算耗尽、状态/物理证书异常或Ctrl+C均停止，保留incomplete和全部部分文件。当前不支持自动恢复/续跑；反馈后由Agent诊断，需要长重跑时另建新目录/配置与计划，仍由用户执行。不得删除旧目录或手改配置、步数、容差、模型/代码SHA来继续。

保护LT21/LT22、旧checkpoint/原数据/manifest、声明校正副本、终端分诊、短检v2/独立审计和本次正式冻结config，不覆盖任何已有产物。

完成后回复“LT23执行完毕”并保留固定目录；反馈console.log、python.log和全部results，包括300套JSON/NPZ。Agent随后核完整性、实际动作/终端证书、双基线回退与场景失败，不自动启动下一阶段或训练。

## 执行结果

- 执行：`status=completed`、`mode=formal`、`elapsed_seconds=485.39`（8.1分钟）、退出码0，`inspect`返回`completed_and_verified`；300唯一索引、300套JSON/NPZ、604产物SHA、148代码与20输入SHA全部重算一致，`results/config.json`与冻结配置逐字节相同，产物84.8 MiB。
- 结果：**255成功（85.0%）/0碰撞**；对LT21原生（205）零丢失、+50；对LT22（243）零丢失、+12（S4新增8、S6新增3、S5新增1）；`#168`恢复（183步，末位置0.2636m/末航向0.1084rad，余量侵入0）。失败45条=43振荡+2超时，全部在S4（23振荡+2超时）、S6（19振荡）、S8（1振荡）。
- 分场景（原生/LT22/LT23）：S1 52/55/55、S2 55/55/55、S3 29/52/52、S4 16/17/**25**、S6 21/22/**25**、S5 32/32/**33**、S8 0/10/10。
- 终端接管：43个回合用到`terminal_rotation`，合计109步（占91,103控制步的0.12%）；109帧全部带完整转向证书，且`full_rotation_contact_safe=true`、位置误差严格小于该车位原`tol_pos`、航向误差大于等于原`tol_yaw`、最短转向角非零；43个用到的回合全部成功，其余212个成功回合未用终端接管。终端帧中仅3步同时有0.2m规划余量侵入（#200），物理接触仍安全。
- 干预与余量：控制步91,103，干预12,617步（13.8%；移位热启动11,180、原采样安全序列1,043、零动作285、终端转向109）；`unsafe_selected_horizons=0`。余量侵入14,944步≤LT22的17,337，按步归一0.164<0.182（原生为0.0415/步，仍高约4倍）；参考接触不安全窗口11,034步，网络危险参考仍未修。
- 联合条件逐项：成功255≥210 ✅、碰撞0≤30 ✅、原205零丢失 ✅、LT22的243零丢失 ✅、#168恢复 ✅、相对LT22新增12 ✅、实际碰撞0 ✅、余量14,944≤17,337 ✅ → `candidate_qualified=true`、`pilot_eligible=true`；`production_qualified=false`不变。
- 判读：终端朝向接管只增加109个转向步，就把S4的振荡/超时近端失败转为8个成功、S6加3、S5加1，同时不丢任何既有成功、不新增碰撞、不增加余量总量，说明"位置已进入容差、航向未对齐"是一个可独立修复的缺口。但它不解决远端起点的振荡（仍有45条失败，S4 25/S6 19）、不改变网络危险参考（11,034步），也不代表BEV占用接入或实车可用；属仿真完整几何辅助的条件实验，主联合68.3%/25.7%与所有生产准入保持false，不替换默认控制器、不部署、不训练。

## 2026-10-26完成反馈独立复核

- 只读inspect重新通过，148源/20输入/604登记产物、完整300分母与双基线联合裁决一致。另用原车辆动力学与原矩形扫掠检查器，独立推进91,103个归档实际动作，状态逐位精确、实际接触0、余量14,944步/94条；109个终端步的完整转向证书与10步限幅零平移序列均独立重建一致，约5.37秒，未重跑正式评测、网络或CEM。
- 容差口径更正：原始车位标准实际为33条0.2m/5°、11条0.2m/10°、212条0.3m/10°、44条0.5m/15°。执行代码始终读取同一引擎/车位原容差；109个终端帧中53帧不符合文档此前误写的统一0.3m/10°条件，但全部符合各自原车位标准。不是运行时放宽验收，也不需要重跑LT23；历史TRACE及旧只覆盖六例的审计保持冻结，以本勘误和按车位重算证据为准。
- 按原生碰撞当步分组：原42条安全参考下执行碰撞恢复36、剩6；原35条消费危险参考恢复11、剩24；原18条无接触未到达恢复3、剩15；原205成功全部保持。剩余45条末态全部未进入各自原位置容差，其中33条最后一条已裁剪参考无联合合格点、12条有联合点仍未到达。继续扩大原地朝向接管范围不能替代位置接近、参考可执行性与重规划进度诊断。
- [独立报告与边界](../../runs/verification/lt23-independent-review-20261026/review.md)：没有独立重算全部非终端CEM排名/候选时域，不把已裁剪最后参考等同未裁剪网络全部输出，不作BEV接入、泛化或生产准入证明。
