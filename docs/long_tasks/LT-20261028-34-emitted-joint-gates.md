# LT34 · 原生发出轨迹联合目标：教师与小集合两个关口

状态：**关口1已完成并独立复核，严格6/8，未通过；暂停关口2拟合。** #860/#2315分别发生6/13步规划余量侵入；未启动训练、关口3/4。不得执行下方第3段条件拟合命令，不换样本或放宽标准。证据见[独立复核](../../runs/verification/lt34-independent-review-20261028/review.md)。

规范：[EMIT-JOINT-001–005](../../spec/emitted_output_joint_training.md)。实现任务T-161628。

## 目的与唯一主要变量

检验只增加自由解码首停分布加权目标位姿损失，能否让网络实际发出的轨迹合格。control保留LT31 control损失和teacher forcing0.2；joint仅增加`0.1*L_emit`，使用额外自由解码分支、温度0.25，坐标头及停止头同时反传。无停止剩余概率归最后一步。原生验收仍为原sigmoid>=0.5首次交叉，无交叉取H320；不使用教师长度截断、停止偏置或专家替换。

关口1：固定标签能否由原MPC按原终点/完整车体安全标准执行。失败则先处理标签或控制契约，不训练、不换样本掩盖失败。

关口2：同权重、同样本序列、同更新量的两臂两seed，小集合能否自主发出合格轨迹。失败即停止该具体配方，不追加轮数、数据、权重扫描或重挑seed。静态通过仍不代表原生闭环增量。

## 固定身份、参数与产物保护

- 工作目录：`D:\program\endtoend Parking`。仅使用`D:\conda\envs\endtoend-parking\python.exe`，Python3.12.13/torch2.13.0+cpu/numpy2.5.2，CPU单线程。
- 冻结：[config.json](../../runs/prepared/emitted-joint-20261028-v1/config.json)，SHA256 `86cbf4c55664de4cd51a9c4098d4edf2b7ec26a7302a269c654244917d3cad6f`。144个执行代码文件、7项输入和缓存/manifest均登记SHA；工作树未提交，以文件SHA为版本。不改冻结hash绕过漂移。
- 训练源：LT29修复后的`runs/verification/surround-target-rebuild-lineage-20261027/train/train-balanced.npz`，2918条，SHA `137b13a63f90188a227f9b8e6b64a36b604c37a71a08775db331c5ec00fb15e2`。
- 共同权重：`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA `359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c`。只加载模型、每臂重置Adam，不继承原优化器或epoch。net-v1/hidden128/image/per_view_v2/semantic/dt0.2/H320/stop0.5。
- 固定八源索引按顺序为`0,433,860,1290,1720,1975,2315,2`，覆盖训练池全部7个场景及正倒车。task_id/场景/机动登记于config。缓存`cohort.npz` SHA `c0ac1b6e1c8420627895547352ae9221ecc69f071fe0536251d40f7ea70ce28a`。不是按新结果挑选的容易样本，失败不得替换。
- 关口1使用归档完整全局教师；MPC原dt0.1/H10、seed=原数据索引（control_seed0）、每10步重规划、最多600步、原车位tol_pos/tol_yaw与0.2m余量不变。感知起点与归档BEV精确匹配。每回合记录全部实际沿程输入，从中等间隔取最多4帧（含原起点）用于拟合；剩余标签按真实消费偏移对齐，不重新求解专家。沿程可执行证据来自同一连续回合，不能宣称验证了每个沿程输入独立重置MPC后的可执行性。
- 关口2预固定seed20261028/20261029；每臂200步，共800次新Adam更新，lr1e-5、clip1、batch8（每步每任务一帧）、TF0.2；原累计BCE0.2/不平衡关闭、endpoint0.5/tail8、terminal1/yaw10/raw，collision0。两臂同seed复用同一预生成样本索引与TF随机序列。
- 保存0/50/100/200检查点及全H320输出、stop logits、专家索引指标、原生首停/完整参考安全。只按最终200步裁决，不挑中间最好。每个检查点使用B1自由解码，并与public predict逐点匹配。
- 生产默认、已有LT29/LT31/LT32/LT33、原模型/数据/配置、prepared与短检产物均禁止覆盖。正式输出为新的`runs/training/lt34-emitted-joint-20261028`；脚本拒绝已存在阶段目录，无resume/retry选项。
- 未来条件回归22回退/8恢复/5残余/93原生成功已在config按task_id并集冻结为**125个唯一任务**。val仅评估，不参与训练；本计划不运行该回归，不称独立未见测试。

## 已完成短检

- 指定conda环境60项聚合测试通过：`python -m unittest tests.test_emitted_output_joint tests.test_emitted_output_study tests.test_emitted_terminal tests.test_terminal_hold tests.test_angular_training_loss tests.test_trainer tests.test_training_config tests.test_training_runner -q`。Spec检查、所有新增Python的AST解析、计划三段PowerShell语法和冻结身份precheck均通过。
- [真实短检](../../runs/verification/emitted-joint-smoke-20261028-v1/report.json)4.33秒：教师#0原MPC80步到达、零碰撞/零余量侵入；四沿程帧标签末点及完整参考安全；只在这个已验证起点上每臂一次Adam更新。原损失分别0.042514290660619736/0.04251429066061973，新项对真实坐标头/停止头梯度范数51.0469/6.5101，参数均有限且两头更新。该短检不授权正式拟合，其他七个教师尚未核验。
- [八输入静态预检](../../runs/verification/emitted-joint-static-precheck-20261028-v1/native-0000.json)0.73秒：自由输出与public predict逐点/首停长度一致，父模型联合末点及完整参考安全3/8。只读基线，无训练收益或滚动结论。

## 耗时与资源

关口1估计1–5分钟，关口2估计5–20分钟。估算依据为本机单例完整教师+两次真实更新短检4.33秒、八输入静态检查0.73秒；其余任务可能执行600步，沿程数量/场景安全检查和负载带来不确定性，未实测正式800更新。脚本硬限关口1为1800秒、关口2为3600秒；耗尽即incomplete，不自动扩大。正式任务按项目规则由用户执行，即使实际提前完成也不自动串行。

CPU，无GPU要求；缓存拟合预计内存1–3GB，建议可用4GB以上；磁盘预留1GB。两臂、两seed顺序运行，不并行或后台启动。

正常进度：每个教师`BEGIN`及`passed=... steps=...`；拟合每臂打印step0/50/100/200和起点/全部沿程合格分母。任何阶段`status=completed`只说明运行完整，科学门槛看`gate.passed`。gate=false也是有效负结果，不能重试直到通过。

## 1. 身份和环境检查

在同一PowerShell窗口执行。原命令成功后才执行后续行；任一异常保留现场并反馈。

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt34Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt34Prepared = 'runs/prepared/emitted-joint-20261028-v1'
$lt34Root = 'runs/training/lt34-emitted-joint-20261028'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt34Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '86cbf4c55664de4cd51a9c4098d4edf2b7ec26a7302a269c654244917d3cad6f') { throw 'LT34 config身份改变' }
& $lt34Python -X utf8 scripts/train_emitted_output_joint.py --phase precheck --prepared $lt34Prepared
if ($LASTEXITCODE -ne 0) { throw 'LT34身份/环境预检失败；不修改hash绕过' }
if (Test-Path -LiteralPath $lt34Root) { throw '正式输出根目录已存在；不要覆盖或自行续跑' }
New-Item -ItemType Directory -Path $lt34Root | Out-Null
Copy-Item -LiteralPath "$lt34Prepared/config.json" -Destination "$lt34Root/config.json"
```

## 2. 先只运行完整教师关口，然后反馈

```powershell
& $lt34Python -u -X utf8 scripts/train_emitted_output_joint.py --phase teacher --prepared $lt34Prepared --output "$lt34Root/teacher" 2>&1 | Tee-Object -FilePath "$lt34Root/teacher.console.log"
$lt34TeacherExit = $LASTEXITCODE
[ordered]@{stage='teacher';exit_code=$lt34TeacherExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt34Root/teacher-exit.json" -Encoding UTF8
if ($lt34TeacherExit -ne 0) { throw '教师阶段不完整；停止并反馈日志与产物' }
& $lt34Python -X utf8 scripts/train_emitted_output_joint.py --phase inspect --prepared $lt34Prepared --output "$lt34Root/teacher"
if ($LASTEXITCODE -ne 0) { throw '教师阶段完整性核验失败；停止' }
```

**到此停止。** 反馈根目录位置及`teacher-exit.json`、`teacher.console.log`、`teacher/report.json`、`teacher/artifacts.json`。保留全部teacher-*.npz和samples.pt供Agent检查，不必上传大文件，本地路径即可。

Agent复核：身份/环境/代码SHA、退出及completed状态、恰好原8源、归档标签终点与完整矩形扫掠、实际状态/动作与原运动学、所有原终点容差、碰撞/0.2m余量、沿程输入及偏移/有效长度完整性。任意教师不合格：暂停训练，转监督或控制契约；不换样本、不放宽标准、不运行第3段。

## 3. 当前禁止运行：原条件拟合命令仅保留追溯

本次教师门禁已失败，以下命令不可作为当前下一步；入口也会在创建拟合输出或优化器前拒绝现有产物。保留原条件命令供审查，不授权删除失败样本、修改hash或更换目录绕过门禁。

```powershell
& $lt34Python -u -X utf8 scripts/train_emitted_output_joint.py --phase fit --prepared $lt34Prepared --teacher "$lt34Root/teacher" --output "$lt34Root/fit" 2>&1 | Tee-Object -FilePath "$lt34Root/fit.console.log"
$lt34FitExit = $LASTEXITCODE
[ordered]@{stage='fit';exit_code=$lt34FitExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt34Root/fit-exit.json" -Encoding UTF8
if ($lt34FitExit -ne 0) { throw '拟合阶段不完整；停止并反馈，不自动续跑' }
& $lt34Python -X utf8 scripts/train_emitted_output_joint.py --phase inspect --prepared $lt34Prepared --output "$lt34Root/fit"
if ($LASTEXITCODE -ne 0) { throw '配对拟合完整性核验失败；停止' }
```

完成后反馈`fit-exit.json`、`fit.console.log`、`fit/report.json`、`fit/artifacts.json`，保留两个plan-<seed>.json、四臂history.json、16个checkpoint和16组native-*.json/npz。

Agent复核：四臂共同初始化、Adam重置、各200更新、实际消费及TF序列配对、0/50/100/200全留证，分母不得丢源。两seed的joint最终都须起点8/8及全部登记沿程输入合格，native stop0.5自由输出末点满足原容差，完整原样参考含状态连接接触/余量安全。专家长度末点、软损失下降、一个seed成功或中间checkpoint成功都不替代门槛。

## 异常、恢复与后续决定

- Ctrl+C/异常/超时：停止当前进程，保留incomplete和console/exit记录，反馈Agent。没有自动恢复、断点续训或从后继阶段绕过失败的命令；输入/代码漂移也直接停止。确需技术性重跑时，先诊断并另冻新身份与新目录，科学预算/样本不因故扩张。
- 完整关口2不合格：该实现/权重/200步预算未建立局部收益，停止该具体配方，核梯度、优化冲突或表示；若control过而joint不过则拒绝干预。不能声称架构永远学不会，也不能增加终点损失继续试。
- 完整关口2通过：只准备独立关口3计划，在同8任务先分开固定首次网络预测与原生重新预测滚动执行。固定参考不通过转动态可执行性；固定通过而滚动失败转访问状态覆盖/重规划连续性。均不授权增加此损失。
- 关口3通过后才准备125任务的条件回归；两seed分别对同剂量control至少新增1成功、逐例保持所有control成功，碰撞与余量侵入不增，新增成功须零风险。有旧成功回退就拒绝，新增不能抵消丢失；还须保持冻结93成功。不自动启动完整300或生产部署，原联合主线标准继续有效。

本计划的成功是形成关口1/2的明确决定，不预告闭环有效。生产/五模块资格始终false，直到后续独立验收实际完成。
