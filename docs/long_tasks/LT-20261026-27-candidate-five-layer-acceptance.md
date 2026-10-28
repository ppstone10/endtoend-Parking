# LT27：当前闭环候选的五单元验收补证

状态：**已执行完成**（用户要求执行；校准→inspect→prepare→precheck→capture→analyze→inspect 全阶段退出0）。证据链完整、`observer_exact_to_LT26=true`，但五层质量未全部通过（L2/L1_L2全部failed、L1 1 failed+4 blocked、L3全passed、L2_L3随任务成败），`all_five_modules_qualified=false`；明细见文末「执行结果」。代码、44项定向聚合测试、单完整回合采集/重放、两任务校准预检此前已完成。任务T-190818，契约LAYER-ACCEPT-001..006。

## 目的与唯一主要变量

相对LT26控制策略不变，唯一主要变量为验收观察和判定：保存实际BEV/元数据、局部预测、模型输入状态/目标/标量类型/语义，独立重放五层；完整参考安全、近终端联合支撑、实际H10/动作安全与原任务到达分开判定，缺证据blocked、明确失败failed。

先用训练源构建感知**经验参考包络**及独立留出检查，再对原LT26固定14进行完整采集和五层分析。这不是新控制/训练策略收益验证，也不要求已知失败变成功；正常负结果应保留，不能为追求通过更改门槛。本轮不进入全300。

感知包络只检验相对固定参考源的一致性，不能建立绝对检测安全线；absolute_quality_status仍为not_established，L1/L1_L2不会因此被宣布绝对质量合格。该未闭合项必须保留，不能伪装“五模块全通过”。

## 固定输入、版本与参数

- 原LT26结果：`runs/verification/feasible-reference-pilot-20261026/smoke/`，报告SHA `0b8921c9e8fe19065176dec5e7d72b4cc863e8dfbfbc9c5cd23f200f50bee4c0`。源14回合5成功、零碰撞/余量，当前候选效果未晋升。
- 已冻结初始配置：`runs/verification/candidate-layers-regression-20261026/calibration-config.json`。**181执行源、41输入**及运行时身份；工作树未提交，SHA为实际版本。初始配置SHA：`7717286f7e796d44457f8d7f3734d9fb8e10e22d0e1deecf6ad2d6e20404bda1`。后续配置不得重冻SHA绕过漂移。
- 训练源：`data/task_dataset/v40_dagger_collision/train-balanced.npz`，来自`runs/training/v40-lineage/net-v1/report.json`，实际SHA在冻结配置inputs内。只读meta对应原任务起点，去除恢复任务/重复seed/与val任务ID或seed交集，不读取val来拟合阈值。
- 静态选中1780个唯一训练任务，最多每scene/noise层96个，再按目标是否在BEV内分组。按seed排序序号mod3=0为留出，其余拟合；至少20拟合、10留出，界为各指标拟合最小值，留出越界任务比例≤0.05。此为明确经验协议，不是统计置信或总体安全保证。
- 已知覆盖缺口：S8 clean仅29、high仅26个原任务，在进一步分组前已不足最低样本数，结果预期blocked；其他组也可能留出失败或不足。不得补val、改样本量/容差/界、反复拟合或自动造数据。
- 验证源：`data/task_dataset/tracked_pivot_v8_3000/val.npz`，SHA `ed0dcb12832354dd0155c8d2daa7c2faef03c21882b2287597393a5e5c029963`；模型：`runs/verification/input-contract-corrected-20261023/deployment-image-contract.pt`，SHA `e917fed2ee965de2b497f744a8cd0f398bf85787c87176ab90f12aafc0973009`。
- 模型权重/实际输入/停止阈值不变：原net-v1 hidden128/max_horizon320、NN dt=.2、stop .26000000000000006。车辆6×3m、max_v2/max_omega1；MPC dt=.1/H10、80样本×5轮、seed=index、每10步重规划/600步上限、原逐车位位置和航向容差。
- 保持LT26 FeasibleReferenceCandidateMPC及证书保持；正常感知/推理继续，原始/使用参考分别归档。完整参考、实际动作/H10都按实体0和规划0.2m矩形扫掠，近终端沿用3×原tol_pos。
- 固定14：0/131/132/139/161/167/168/172/183/267/268/279/292/296。采集必须与LT26的结果/动作/状态/所有原始和使用参考/证书逐值一致，才可分析。静态GT几何、确定性差分模型且可立即停车，production_qualified=false。

## 已完成短预检

- 单回合#0：188步，19次真实预测，全输入采集6.053秒、五层独立重放6.887秒；与LT26逐值一致，五层接口均通过。L1缺正式profile为blocked；L2因1条完整参考实体危险、6条规划余量危险为failed；L3/L2_L3仅本回合覆盖内passed，不是全域通过。两次近终端均有联合目标支撑。
- 两训练任务校准0.316秒，preflight=true、ready_groups=0、绝对质量未建立；该产物不能用于正式通过。
- 44项相关聚合unittest通过；删除实际BEV键的临时副本在网络/MPC重放前被拒绝；文件篡改、包络逻辑重算、缺组/少样本/留出失败/空真值/异常恢复有定向验证。新8个Python文件编译通过。依赖profile的prepare已使用真实预检profile跑通。

## PowerShell命令（严格依次执行）

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$parkingPython = 'D:\conda\envs\endtoend-parking\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$parkingBase = 'runs/verification/candidate-layers-regression-20261026'
$parkingCal = 'scripts/calibrate_perception_acceptance.py'
$parkingLayers = 'scripts/verify_candidate_layers.py'

& $parkingPython -X utf8 $parkingCal --phase precheck --config "$parkingBase/calibration-config.json"
if ($LASTEXITCODE -ne 0) { throw 'LT27身份预检失败，禁止重冻SHA绕过' }
foreach ($parkingTarget in @("$parkingBase/perception-reference", "$parkingBase/config.json", "$parkingBase/capture", "$parkingBase/analysis", "$parkingBase/console.log", "$parkingBase/python.log")) {
    if (Test-Path -LiteralPath $parkingTarget) { throw "已有产物 $parkingTarget；禁止覆盖/自动续跑，请反馈" }
}

function Invoke-LT27Python {
    param([string[]]$ParkingArgs)
    & $parkingPython -X utf8 @ParkingArgs 2>&1 | Tee-Object -FilePath "$parkingBase/python.log" -Append
    $parkingExit = $LASTEXITCODE
    Write-Host "LT27 PHASE EXIT $parkingExit"
    if ($parkingExit -ne 0) { throw 'LT27阶段异常，保留全部产物并反馈；不自动重试' }
}
```

```powershell
Start-Transcript -LiteralPath 'D:\program\endtoend Parking\runs\verification\candidate-layers-regression-20261026\console.log' -NoClobber -ErrorAction Stop
try {
    Invoke-LT27Python -ParkingArgs @($parkingCal, '--phase', 'run', '--config', "$parkingBase/calibration-config.json", '--output', "$parkingBase/perception-reference")
    Invoke-LT27Python -ParkingArgs @($parkingCal, '--phase', 'inspect', '--config', "$parkingBase/calibration-config.json", '--output', "$parkingBase/perception-reference")
    $parkingProfile = Get-Content -LiteralPath "$parkingBase/perception-reference/profile.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($parkingProfile.preflight -ne $false -or $parkingProfile.status -ne 'completed') {
        throw '必须是完整校准产物；不能以预检代替'
    }
    Invoke-LT27Python -ParkingArgs @($parkingLayers, '--phase', 'prepare', '--purpose', 'regression', '--profile', "$parkingBase/perception-reference", '--config', "$parkingBase/config.json")
    Invoke-LT27Python -ParkingArgs @($parkingLayers, '--phase', 'precheck', '--config', "$parkingBase/config.json")
    Invoke-LT27Python -ParkingArgs @($parkingLayers, '--phase', 'capture', '--config', "$parkingBase/config.json")
    Invoke-LT27Python -ParkingArgs @($parkingLayers, '--phase', 'analyze', '--config', "$parkingBase/config.json")
    Invoke-LT27Python -ParkingArgs @($parkingLayers, '--phase', 'inspect', '--config', "$parkingBase/config.json")
    Write-Host 'LT27补证阶段完成；等待Agent核验质量失败及blocked原因，不自动全300或训练'
}
finally {
    Stop-Transcript
}
```

初始校准配置已存在；实际采集config.json要等完整profile生成后才prepare，固定其4个产物SHA。该依赖准备阶段是计划中的真实步骤，不是占位配置。校准组不足或留出失败可完整产出blocked/failed并继续补齐五层证据，不能当程序异常，也不能因此重选数据。

## 耗时、资源与正常进度

预计**15–45分钟**，单进程CPU，无需GPU，预留2–6GB内存、3GB磁盘。估算依据：两训练任务0.316秒，1780个任务线性约4.7分钟，但复杂场景3600束真值/磁盘/分组开销不同；LT26固定14推理+控制107.69秒，本轮多输入归档及独立重放可能各2–8分钟。该估计不保证，按长任务交用户执行。完整校准硬预算2小时、采集15分钟、分析30分钟；超时反馈，不擅自放宽预算。

校准每25任务输出`[perception-reference n/1780]`，采集每回合输出`[layer-capture n/14]`，分析每回合输出`[layer-analysis n/14]`。持久化report.json先running、正常completed、异常incomplete。回合内可能较久无文本，尤其安全样本检查与独立重放，不重复启动或并发改冻结源码/配置/模型。

## 输出及正常完成标志

在`runs/verification/candidate-layers-regression-20261026/`：

- calibration-config.json（既有）及依赖生成的config.json；console.log/python.log（PowerShell 5的Tee日志可能UTF-16，保留BOM）。
- perception-reference/：config.json/report.json/profile.json/artifacts.json，3个登记产物；profile包含训练任务、分组、拟合/留出seed、指标界、失败率及绝对质量未建立状态。
- capture/：config.json/report.json/artifacts.json/control/raw-report.json；14套episode/retention/inputs的JSON+NPZ，通常87个登记产物。inputs存实际BEV、局部输出、模型状态与原始参考；JSON存位姿类型、模型局部目标、空间通道及实际语义。
- analysis/：config.json/report.json/artifacts.json及14套episode-XXXX.json，通常16个登记产物。逐例包含五层证据/接口/质量状态、参考风险/近终端支撑/真实H10/任务指标和感知包络对应。

完成标志：每阶段退出0、profile非preflight、capture/analyze均completed、14唯一任务、observer_exact_to_LT26=true、inspect退出0。all_five_modules_qualified=false或某层failed/blocked属于预期可报告质量结果，不能因退出0宣布模块全部通过。

## Agent后续验收

先核任务身份/运行时/日志尾和各阶段退出、181代码/41输入及profile4文件SHA、配置、校准训练源排除val/恢复/重复seed、分组/拟合/留出/界和失败率、14分母及必要产物。再独立核对采集没有改变LT26动作/状态/参考/证书，实际BEV/局部预测重放及输入语义、原车位终端/完整车体安全、五层规则可从事实复算。

重点报告：哪些失败由完整参考安全暴露、哪些近终端缺支撑、哪些L1组因不足/留出失败/适用域不符阻断；不得将经验参考相似度当绝对感知质量、不把危险远端参考直接计成实际碰撞、不把有限窗口未含终点一律判错、不将全部失败推断为接口错误。14任务结果应仍与LT26相同，验收补齐不等于策略改善。

## 异常停止、恢复与保护

身份漂移、traceback、预算耗尽、输入/分母缺失、观察改变原结果或动作/参考、Ctrl+C均停止，保留running/incomplete和部分文件。反馈后诊断，必要时新配置/目录重新运行指定阶段；不自动删除、续跑、重复作业或串行下一阶段。禁止修改阈值/样本数/原tol_pos/tol_yaw/.2/600步/模型/种子或SHA绕过问题。

保护旧174源及LT21–LT26全部配置/结果/模型/数据，新181源、短预检证据和初始calibration-config。停用新CLI即可回退，无数据迁移、训练或默认入口替换。完成后反馈“LT27执行完毕”或异常阶段及产物路径；全部文件保留，Agent核验后再决定后续内容。

## 执行结果

- 阶段：precheck `passed` → 校准run **94.27秒**（1780任务/24组/16 ready）→ 校准inspect → prepare（purpose=regression）→ precheck → capture **111.60秒** → analyze **132.95秒** → inspect；全部退出0，profile `preflight=false`、`status=completed`。
- 身份与产物：`calibration-config` SHA `7717286f…4bda1`与计划一致；两个配置均181代码/41输入零漂移；perception-reference 3个、capture 87个、analysis 16个登记产物SHA全部一致；capture 14套episode/retention/inputs齐全、analysis 14套；`observer_exact_to_LT26=true`、14唯一索引。
- 感知经验参考（训练源）：24组中**16 ready、3 failed、5 blocked**。failed=S1|high(target=1) 留出失败率0.125、S3|low 0.062、S8|low 0.100（门槛≤0.05）；blocked=S2|{clean,high,low}|target=0（拟合2–5个）与S8|{clean,high}（拟合19/17<20）。组界为拟合最小值，`absolute_quality_status=not_established`，与计划预期一致。
- 五层结果（14例）：
  - **L1**：9 passed、1 failed（#131 `outside_perception_reference`，其组ready但57/60次查询越界）、4 blocked（#139 组failed；#183 有4次查询落在**profile中不存在的**`S4|clean|target=0`组；#292/#296 的S8|high组blocked）。
  - **L2**：14/14 failed——每例完整原始参考含1–60条实体或余量危险（#131/#132/#161/#167为60条），部分另有`terminal_support_not_observed`。
  - **L3**：14/14 passed——所有实际转移与完整H10均无实体/余量危险（`unsafe_actual_steps=0`、`unsafe_horizons=0`）。
  - **L1_L2**：14/14 failed（继承L2的完整参考危险，L1的越界/缺参考并列）。
  - **L2_L3**：5 passed（=本子集成功的#0/#168/#172/#267/#292）、9 failed，原因均为`original_task_not_reached`。
- 近终端支撑：未到达的S4/S6案例普遍缺联合支撑——#167 47/47、#268 28/28、#183 13/17、#279 5/7 近终端查询无原位置+航向联合达标点；成功的5例中#168有2次、#292有1次未支撑。
- 判读：本轮把"证据缺口"与"质量失败"彻底分开——采集与重放**全部精确**（14例与LT26逐值一致），实际执行层(L3)在本次覆盖内**全部安全**，但完整参考(L2)在**每一例**都已危险，其中若干近终端无联合支撑；感知层还有3组留出失败与5组样本不足导致1 failed+4 blocked。因此`all_five_modules_qualified=false`，不能宣称"五模块全通过"，L1/L1_L2的绝对感知质量仍未建立。14任务结果与LT26完全相同（5成功），验收补齐不等于策略改善；不进入全300、不训练、不替换默认入口。


## 完成反馈后的独立复核与新增几何诊断

重新inspect通过，181代码/41输入、profile4文件及3/87/16产物身份一致；7个阶段EXIT 0与正常日志结束标记确认。1780任务选择/val交集排除/去重/拟合留出可复算，6个训练测量独立重算一致。694预测/6919控制，五层接口全passed。更正上方统计：L1应为9包络内+1越界+4受阻；131有57查询越界；168有2次近端无支撑。

新增复核：16个ready组target IoU/hit_rate下界全部为0；694查询中351次几何目标在BEV内但零命中，39次目标通道空、312次非空但与真值无交集。绝对质量未建立，经验通过不能排除目标错位。

已用无噪声独立几何对照确认：SimulatedCamera把四个视角叠到同一image，Camera2BEV仅按前视IPM解释。左右各8m目标产生逐元素完全相同的相机图像及BEV，均落在车前8m，而真实左右区域不相交；保留单视角方向后旋转的离线对照可分别还原左右（IoU 1）。既有训练NPZ第2条clean任务目标在车后6.42m，存储target却在车前5.93m，与真值无交集。因此L1内部存在具体几何缺陷，不能仅将剩余问题归因网络或控制。未修改冻结代码、数据、模型，未证明修复投影即可解决全部闭环失败。

后续优先保留相机视角和外参，逐视角投影到统一车体系再融合，增加左右后向回归并审计受影响训练输入；然后再验证网络和闭环收益，不直接换元数据或增加采样。证据：runs/verification/lt27-independent-review-20261026/{report.json,review.md,surround-projection-probe.json,surround-projection-probe.npz,stored-training-target-probe.json,stored-training-target-probe.npz,surround-direction-loss.png}。
