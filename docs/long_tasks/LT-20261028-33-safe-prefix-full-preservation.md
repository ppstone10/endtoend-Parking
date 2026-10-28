# LT33：安全前缀完整闭环与全量旧能力保持

状态：**执行完毕，独立复核完成；联合候选验收未通过**。原冻结配置与执行命令保留，不重新运行。

## 目的与唯一主要变量

在LT32相同300个任务、LT31 control权重、原始停止阈值和严格安全控制器下，只把参考准入策略由strict_footprint改为safe_prefix。新策略保留原样安全前缀，并在距目标不超过3倍原位置容差时要求原位置/航向联合目标支撑；两项构成一个策略变量，本次不能分别归因。控制器、终端接管算法、0.2m余量、原车位容差和600步上限不变。

本次只运行候选300回合，与不可变LT32缓存的strict/native各300回合逐例配对，不重复跑基线。成绩属于静态仿真GT几何/专家辅助的混合系统，不是纯网络改善证明，不训练、不部署、不回灌评测源。

## 已完成预检

- 三条同模型回退80/81/196全部恢复，分别523/528/268步，10.675秒。
- 预先固定LT32全部13失败及各场景/机动首个成功共24例：11→19成功，8例恢复（22/80/81/83/85/96/187/196）；全部7条历史回退恢复，抽查11条LT32成功全保持。剩余183/186/207/292/296失败。
- 最终同冻结代码短检62.228秒，8241实际控制、833发出参考、全部选中H10和26终端证书独立审计：接触/余量侵入/危险参考/危险时域均0。269次安全前缀、109次严格回退、455次原样通过；原始与发出参考逐值/顺序/dtype/dt及前缀边界核验通过。
- 当前矩阵下旧strict_footprint的80号回合与LT32结果、步数及末态误差完全一致；这只证明该回合兼容，不替代全量。
- 最终短检：`runs/verification/safe-prefix-expanded-final-20261028`；[复核报告](../../runs/verification/safe-prefix-review-20261028/review.md)。24例不能外推300结果。

## 冻结输入与参数

- 环境：`D:\conda\envs\endtoend-parking\python.exe`，Python3.12，CPU，Torch/OMP/MKL线程1；工作目录`D:\program\endtoend Parking`。
- 模型：`runs/training/lt31-terminal-hold-20261027/control/last.pt`，epoch3、image/semantic/per_view_v2、horizon320、dt0.2、原生stop0.5，不替换权重。
- 数据：`runs/verification/surround-target-rebuild-lineage-20261027/val/val.npz`及同目录manifest，数据SHA `b8250edeb24f966374cb1900a101c6239723888183a37c62046214d14885cc38`。
- 基线：`runs/verification/strict-footprint-formal-20261028/results`；历史保持：`runs/validation/live-20261018-170806/full/E5/report.json`。LT32旧代码141项独立归档`runs/prepared/safe-prefix-20261028/source-before`，原1204产物SHA均已核验，不用修改旧身份绕过检查。
- 正式配置：`runs/prepared/safe-prefix-20261028/formal-config.json`，SHA **`4e99aa67166e685596881875b1725508f119818177820d25fec1d66abd612e3f`**，冻结143代码/10输入及环境身份。数据/manifest/模型/历史、基线与短检各文件SHA逐项在配置内。
- 0..299完整顺序，control_seed0、MPC seed=dataset_index；max_steps600，replan_every10，MPC dt0.1/H10，网络dt0.2；VehicleConfig原参数及各车位原tol_pos/tol_yaw严格小于判据。参考/选中H10/实际转移使用0.2m规划余量，实体接触另独立统计。

## 依序执行的PowerShell命令

在同一个PowerShell窗口依次执行三段。不要运行prepare或重新冻结，不改代码、数据、参数或样本分母。

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$prefixPython = 'D:\conda\envs\endtoend-parking\python.exe'
$prefixRoot = 'D:\program\endtoend Parking\runs\verification\safe-prefix-formal-20261028'
$prefixConfig = 'runs/prepared/safe-prefix-20261028/formal-config.json'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
if ((Get-FileHash -LiteralPath $prefixConfig -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4e99aa67166e685596881875b1725508f119818177820d25fec1d66abd612e3f') { throw 'LT33配置身份不符，停止' }
& $prefixPython -X utf8 scripts/validate_safe_prefix.py --phase precheck --config $prefixConfig
if ($LASTEXITCODE -ne 0) { throw 'LT33预检失败；反馈，勿重冻SHA' }
if (Test-Path -LiteralPath $prefixRoot) { throw 'LT33目录已存在，保留现场并反馈，不覆盖或续跑' }
New-Item -ItemType Directory -Path $prefixRoot -ErrorAction Stop | Out-Null
```

```powershell
Start-Transcript -LiteralPath (Join-Path $prefixRoot 'console.log') -NoClobber -ErrorAction Stop
& $prefixPython -X utf8 scripts/validate_safe_prefix.py --phase run --config $prefixConfig --output (Join-Path $prefixRoot 'results') 2>&1 | Tee-Object -FilePath (Join-Path $prefixRoot 'python.log')
$prefixRunExit = $LASTEXITCODE
[ordered]@{task='LT-20261028-33';exit_code=$prefixRunExit} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $prefixRoot 'exit.json') -Encoding UTF8
Write-Host "LT33 RUN EXIT $prefixRunExit"
Stop-Transcript
if ($prefixRunExit -ne 0) { throw 'LT33不完整；反馈日志与results/report.json，勿自动重试' }
```

```powershell
& $prefixPython -X utf8 scripts/validate_safe_prefix.py --phase inspect --output (Join-Path $prefixRoot 'results')
if ($LASTEXITCODE -ne 0) { throw 'LT33产物/分母/身份/裁决核验失败，保留结果并反馈' }
```

## 耗时、资源与进度

24回合62.228秒，线性300约13分钟；困难场景、专家重规划单次8秒上限和全扫掠审计会增加波动。预计**10–35分钟**，CPU/磁盘不同可能更久，8小时是异常硬预算而非耗时承诺。CPU单进程、无需GPU，建议2–4GB内存及2GB可用磁盘。

正常逐例打印进度，每回合落盘一套JSON/NPZ；无输出时看日志和文件增长，不另起重复作业。运行期间不要改冻结文件或继续启动后续阶段。

## 输出与完整完成标志

外层`runs/verification/safe-prefix-formal-20261028/`包含console.log、python.log、exit.json。results内为formal-config.json、identity.json、raw-report.json、report.json、artifacts.json，以及300套episode-XXXX.json/npz；604个文件登记SHA，artifacts.json本身不自登记。

NPZ记录原始网络参考、发出参考、查询状态、dt、原车位目标/容差、实际前后状态/动作、完整选中H10和终端时刻；逐例JSON记录结果、准入决策与独立审计。没有训练检查点，输入模型原样保留。

完整运行要求：run退出0、status=completed、0..299唯一完整、无复原失败、冻结代码/环境/10输入一致、604产物SHA一致、inspect通过。completed/exit0不等于candidate_qualified；负结果也是有效完成。

## Agent联合验收标准

先核配置/任务身份、日志尾部、exit、分母、模型/短检/基线SHA及文件完整性，再看汇总和逐例回归。

候选必须同时满足：完整300、至少210成功、**LT32 strict的287个成功全部保持、当前native的93个成功全部保持、历史205个成功全部保持**、相对LT32至少1个新增成功，以及碰撞/实际规划余量侵入/危险选中H10/危险发出参考全部0。不能用新增抵消丢失；三个保持集合并集要求实际上至少294成功，不预报295或把24例改善外推全量。

分列S4/S6/S8、场景/机动/噪声、8条恢复、剩余5失败；对参考原样/截断/回退、控制安全改选和终端接管分别解释。检查安全前缀逐值及桥接、首危险点排除、完整时长与近端联合支撑，不把几何安全等同可达性。独立动作/参考/H10审计不等于重算全部CEM种群排名。

即使candidate_qualified=true，也仅是当前同分母GT辅助候选资格；production_qualified与all_five_modules_qualified保持false。纯网络仅31%的基线没有因参考策略自动改善；未知障碍、感知绝对质量、真实制动、其他seed/分布泛化需单独验证。

## 异常、恢复、保护与反馈

身份漂移、非有限、预算耗尽、扫掠/动作/终端证书不一致、Ctrl+C或非零退出时保留现场，不自动恢复、重试、重复单例或覆写目录。先反馈错误及日志尾部；Agent诊断后另立版本配置/新目录交由用户重跑。安全停止是失败类别，不能算到达或裁剪分母。

禁止覆盖LT21–LT32结果、旧/新短检、source-before/source-preflight-v1、模型/val/manifest、冻结配置及正式残留。停用safe_prefix可回退原入口，无数据迁移；默认none不变，不commit/push。

完成后回复“LT33执行完毕”或异常，并保留外层三文件与results全部文件。Agent复核后再决定后续，不自动启动另一长任务。

## 正式结果与独立复核

- 身份/完整性：配置SHA与交付一致，143代码/10输入、604产物SHA、0..299唯一完整、任务/容差身份与LT32逐例一致、日志300/300、exit0、inspect均通过；正式728.961秒（12.15分钟）。原24例结果/步数/误差/审计/准入决定与短检完全一致。
- 结果：273/300到达（91%）、0碰撞/0余量/0危险H10/0危险参考；23振荡、2安全停止、2超时。相对LT32新增8、丢22，净少14；LT32成功保持265/287，当前native成功保持93/93，历史成功保持194/205，联合候选失败。不以达到主线70%替代全部保持门槛。
- 独立落盘复算300场景：42.521秒，89187实际步/9031完整参考/全部H10/629终端证书及前缀准入逐例一致；未重跑网络或CEM排名。准入原样4673、前缀3073、回退1285。
- 定位：全部22新回退的首次参考变化均发生在相同状态、远离近端门槛时选择safe_prefix；S3丢11、S4丢5、S5丢5、S1丢1。172号590步零动作，而S4四个回退仅少量安全改选仍振荡，证明几何/时长不足以保证推进与恢复。127/148近端无联合支撑而安全停止；不能只归因安全控制过严。
- 决定：safe_prefix不晋升默认或混合候选；生产/五模块资格仍false。后续针对参考候选的实际控制推进、跨周期连续性与恢复条件做同状态诊断，固定全部22回退+8恢复+5残余，不只挑成功例；本轮未改运行代码或启动下一长任务。
- 证据：[LT33独立复核](../../runs/verification/lt33-independent-review-20261028/review.md)，含report、逐例失败与独立审计。LT32/短检/冻结配置/正式原产物均未改。
