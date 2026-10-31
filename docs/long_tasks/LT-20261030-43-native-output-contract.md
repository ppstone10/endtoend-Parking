# LT-20261030-43 · 自由输出停止与整段几何契约分解

**状态：已执行并完成独立复核，完整负结果。AB双seed起点均由旧5/8退至1/8，拒绝候选并停止本次配方；不续训、不覆盖既有产物。下列命令保留作历史执行记录。**

## 正式结果与退出决定

fit/inspect均完成、退出0，fit384.1101秒。171源、82登记文件、24检查点、1200新更新及全部768保存输入评估核验通过；独立重跑最终6模型的192次全头forward和192次公开predict，数组一致。任务分母为8训练任务/32相关输入，没有闭环执行或验证集回灌。

| 配方 | 双seed起点合格/8 | 双seed全部输入合格/32 |
|---|---|---|
| LT42 joint旧基线 | 5 / 5 | 16 / 16 |
| A | 1 / 1 | 12 / 11 |
| B | 5 / 5 | 18 / 17 |
| AB唯一候选 | 1 / 1 | 16 / 15 |

AB两个seed均丢#433/#1720/#1975/#2四个旧合格起点，完整头仍有安全目标点却提前停止；A同向退化。B减少参考几何风险，但无新增起点成功、各丢1份旧合格沿程输入，不能替代AB或据此称任务收益。困难起点#860/#1290/#2315仍未解决。**未通过预定8/8及32/32门槛，不进入固定/滚动或125保持阶段，不增加剂量。** A包含多个同步改变，后续先分解首次触发约束与条件监督覆盖，尚不能唯一归因某个子项。详见[独立复核](../../runs/verification/lt43-independent-review-20261030/review.md)。

后续[有界A机制诊断](../../runs/verification/lt43-a-component-diagnosis-20261030/review.md)已确认：最早窗口目标仍惩罚全部16/15份已合格AB输出，贡献72.31%/68.69%的加权窗口损失，并在冻结头梯度中推动四个回退起点提前触发。仅门控合格输入或mean换max均不足；固定坐标理想选停上界仍6/8起点。未实现新目标或启动新训练，先修订实际输出验收一致的契约测试；不能唯一归因A全部退化。

## 目的与唯一主要变量

LT42的联合项有沿程静态学习效果，但双seed均仅5/8起点、16/32全部输入，不能继续旧配方。本轮唯一主要变量为**训练目标与解码监督契约**，内部预声明两个因素A/B用于归因；数据、父权重、样本顺序、更新剂量、原生推理和控制/验收标准保持。

| 条件 | A：自由坐标及原生停止监督 | B：实际发出前缀几何 | 角色 |
|---|---|---|---|
| LT42 joint | 无 | 无 | 只读复用2旧臂，不重训 |
| A | 有 | 无 | 机制诊断 |
| B | 无，原LT42目标保持 | 有 | 机制诊断 |
| AB | 有 | 有 | **唯一预声明候选** |

A/B不能在AB失败后取代它救分，不能按任务或seed拼接成绩。目标是消除已知停止错位与早段危险参考，特别看#860/#1290/#2315三个起点；沿程数量不等于新增独立任务。规范：[NATIVE-FIT-001～005](../../spec/native_output_contract_training.md)。

## 固定输入、源码与配方

- 工作目录`D:\program\endtoend Parking`；conda Python `D:\conda\envs\endtoend-parking\python.exe`，CPU单线程，无外部依赖。
- prepared：`runs/prepared/native-output-contract-20261030-v1/config.json`，SHA256 **028996fbea9adf19b3fc8be80b0fd01d2ec608c37c3ffdc8c6e6ba47bc3879dc**。冻结原167源及新4脚本，共171源；原源码/默认Trainer/模型/MPC/predict不改。
- 前检目录`runs/verification/native-contract-20261030/preflight-v1`，11登记文件，`artifacts.json` SHA **95e0069a46d2d3d6ac897933aff3bf683e72874f79423656dedfd655f52437cb**；不保存可复用的短检权重。
- 教师授权`runs/prepared/emitted-recert-teachers-20261030-v1`，沿用原`artifacts.json` SHA **07cc44d6cb6bb4589df0a700e9be64426e9f18124aed80fda8e6e800350a8963**。原8训练索引`[0,433,860,1290,1720,1975,2315,2]`，每任务4个原LT39连续教师控制前态，共32输入。原实际BEV/局部目标/运动、原连续剩余标签及mask完全保留，无新增数据或标签生成。
- 原训练源2918行`runs/verification/surround-target-rebuild-lineage-20261027/train/train-balanced.npz` SHA **137b13a63f90188a227f9b8e6b64a36b604c37a71a08775db331c5ec00fb15e2**；原8源缓存`runs/prepared/emitted-joint-20261028-v1/cohort.npz` SHA **c0ac1b6e1c8420627895547352ae9221ecc69f071fe0536251d40f7ea70ce28a**。验证21/125/300不回灌训练。
- 每臂从原父`runs/training/lt31-terminal-hold-20261027/control/last.pt`初始化，SHA **359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c**；Adam重置，不从LT42终态或预检恢复。
- 复用LT42完整`runs/training/lt42-recertified-emitted-joint-20261030/fit`，56文件清单SHA **096e4d9a77e1dadfe55e02b2c84afdb561bda9ce919b3dbccddfb19ef87019c6**。绑定原配置SHA bfd133d6c993c88975f0e8cc7fdcadbb94f136f29b74c879ef441cc5c62cec32、两份退出0、inspect完整16检查点/512输入审计；不复制或改写旧基线，不将其计为新训练。
- seeds固定20261028/20261029，每seed新A/B/AB各200更新，共**6新臂/1200新更新**，另2旧joint臂复用；batch8（每任务一帧）、lr1e-5、clip1，预生成计划和原LT42相同。B保留teacher forcing0.2及原seed+step随机协议；A/AB纯自由解码，不注入教师或填充点。
- 只存0/50/100/200，24新检查点；最终只判200。原生B1 stop0.5/H320/dt0.2、原位姿严格双容差、整段接触/0.2m扫掠及桥接保持。专家长度只用于模仿/专家末端诊断，不裁剪推理。
- A保留自由轨迹模仿+0.5原专家末端对齐+原terminal_pose（yaw_weight10），替换为`0.2*窗口hinge + 0.1*真实原生末点误差`。窗口来自原严格双容差及完整硬安全前缀，取最早可终止点；此前负logit、该点正logit，固定间隔0.5 **logit**，不是修改0.5概率阈值。正/负前缀分别归一化。无窗口暂缓该样本停止项，仍参与坐标/末端/几何学习，每步重算，不伪造安全正例或删除输入。
- B使用冻结GT forbidden几何、6×3m车体加原0.2m膨胀、原0.1m分辨率逐段扫掠；计算完整原生前缀含桥接的最坏穿透量（米），固定系数1。支持本固定池矩形、圆、凸多边形，未知/非凸拒绝。长度、段采样数和窗口离散标记脱离梯度；坐标保留梯度。单段>10000子步或总采样>200000直接异常，不裁剪蒙混安全。
- 使用GT几何训练是显式实验边界，不更换感知BEV输入；原标量硬几何仍负责验收，惩罚不是连续时间解析安全证书。原MPC6/8失败、LT34旧fit阻断和LT40拒绝不变，无生产资格。
- fit/inspect各7200秒硬预算，无resume/自动重试。

## 已完成短检

[前检记录](../../runs/verification/native-contract-20261030/review.md)：43项定向测试2.392秒通过、新4脚本/1测试py_compile通过。覆盖0.49反例、首次越阈/空窗口、实际末点坐标梯度、近端/桥接/尾段、矩形/圆/凸多边形/旋转/地图边界、未知/非凸与采样上限拒绝、固定分母/AB裁决、授权前不建优化器、防覆盖/漂移/incomplete。

完整前检24.298秒：32安全教师零惩罚，旧64参考的全部22份余量风险与硬判据一致、有有限非零坐标梯度。A/B/AB各独立1更新通过，B去掉几何项后的首批损失与原LT42逐值一致。三臂更新后均起点3/8、全部5/32，**无收益结论**；正式权重仍从父模型开始。

交付检查：Spec校验、下列3段PowerShell语法解析、git diff --check及最终171源身份precheck通过；正式输出目录不存在。

## 预计耗时与资源

fit估计6–20分钟，inspect另1–5分钟。依据本机首批单步A/B/AB分别0.267/0.402/0.273秒，外推两seed各200约377秒，另24次原生评估、文件与旧证据哈希。几何扫掠数随预测坐标变化，首批不能保证后续成本，机器负载/文件压缩也有不确定性；每阶段2小时到限停止。

CPU单线程，无GPU；建议4GB可用内存、2GB额外磁盘。24检查点、原生输出和history预计数百MB以内，仍需保留旧数GB证据；不删除旧产物腾空间。未知几何/过大扫掠的拒绝不能改成放宽安全或静默封顶。

## PowerShell：在同一窗口顺序执行

### 1. 环境、冻结身份与目录保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt43Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt43Prepared = 'runs/prepared/native-output-contract-20261030-v1'
$lt43Root = 'runs/training/lt43-native-output-contract-20261030'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt43Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '028996fbea9adf19b3fc8be80b0fd01d2ec608c37c3ffdc8c6e6ba47bc3879dc') { throw 'LT43配置漂移，停止并反馈' }
& $lt43Python -X utf8 scripts/train_native_output_contract.py --phase precheck --prepared $lt43Prepared
if ($LASTEXITCODE -ne 0) { throw '源码/权重/监督/前检/旧基线身份漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt43Root) { throw '已有LT43目录，保留现场，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt43Root | Out-Null
```

须status=passed、tasks=8、samples=32、new_arms=6、reused_arms=2、updates=1200、source_files=171。fit自身仍在输出/优化器创建前复核授权；不得手改SHA或重建清单绕过。

### 2. 正式六新臂分解实验

```powershell
& $lt43Python -u -X utf8 scripts/train_native_output_contract.py --phase fit --prepared $lt43Prepared --output "$lt43Root/fit" 2>&1 | Tee-Object -FilePath "$lt43Root/console.log"
$lt43FitExit = $LASTEXITCODE
[ordered]@{task='LT43';stage='fit';exit_code=$lt43FitExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt43Root/exit.json" -Encoding UTF8
if ($lt43FitExit -ne 0) { throw 'LT43拟合未完整完成，保留现场反馈，不重试' }
```

顺序seed20261028的A/B/AB，再20261029的A/B/AB；每臂在0/50/100/200输出`FIT seed arm step=... initial=.../8 all=.../32`，共24行。每步损失、窗口/deferred或几何长度、样本及梯度记录在history。最终status=completed，gate只由两AB最终决定。**门槛失败是有效负结果，仍需第3段审计，不重训。**

### 3. 完整24检查点审计

```powershell
& $lt43Python -u -X utf8 scripts/train_native_output_contract.py --phase inspect --prepared $lt43Prepared --output "$lt43Root/fit" 2>&1 | Tee-Object -FilePath "$lt43Root/inspect.log"
$lt43InspectExit = $LASTEXITCODE
[ordered]@{task='LT43';stage='inspect';exit_code=$lt43InspectExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt43Root/inspect-exit.json" -Encoding UTF8
if ($lt43InspectExit -ne 0) { throw 'LT43完整审计失败，保留现场反馈，不重试' }
```

每检查点输出`AUDIT seed arm step=... inputs=32`。末尾须status=verified、tasks=8、samples=32、new_arms=6、reused_arms=2、updates=1200、checkpoints=24、native_evaluations=768、artifacts=82。每输入有全H320前向和公开predict一致性，即1536次模型前向；原始点/logits/目标/mask逐值核对。核全部origin与旧基线输出相同、父权重/空Adam、非零检查点Adam步数、1200记录/样本顺序、最终AB及逐任务比较一致；不修改正式结果。

## 产物、异常、恢复与禁止覆盖

反馈完整`runs/training/lt43-native-output-contract-20261030`：console.log、exit.json、inspect.log、inspect-exit.json及fit全部文件。fit含config/report、2计划、6 history、24 checkpoint、48 native JSON/NPZ，共**82登记文件**，另artifacts.json不自登记；旧基线仍原路径只读。

- 超时、非有限、梯度/几何错误、未知形状、采样超限、文件缺失/漂移、Ctrl+C或任意非零退出：incomplete，保留整个现场，先反馈诊断，不删除目录、补写报告、换seed或自动续跑。
- 几何超限可能是模型输出异常，不得扩大上限或裁剪轨迹来救分。无窗口deferred是保留的训练状态，不是成功或样本豁免；最终仍须全部32输入通过。
- 修复代码后必须重新绑定新版本前检/prepared与结果目录，Agent先短检再另交命令；现有前检和prepared不可覆盖，失败候选不能增加轮数/权重。
- 禁止覆盖LT31、LT34～42及本轮任何已存在权重/数据/授权/prepared/前检/结果目录。停用新入口即可回退，无默认训练/推理工厂或数据迁移变更。

## 后续验收与退出

1. Agent先核配置171源/输入/原56基线/前检身份、日志与两份退出0、82文件、24检查点、1200新更新及768输入完整审计；完整性成立才分析质量。
2. 只按两AB最终200裁决：起点8/8、全部32/32、旧合格全保留且零风险。特别核#860/#1290/#2315起点；不以沿程净增代替任务推进，不拼接A/B或两个seed。
3. AB失败立即停止此新目标/实现/固定预算；A/B只用于分流停止对齐、路径几何或二者耦合的问题。不得以某个较早检查点或某个单因素通过挽救AB。
4. 静态通过只允许另准备相同实验控制契约、无专家替换的固定/滚动验证。固定失败转参考—控制可跟踪性，固定过而滚动失败转访问状态覆盖和重规划连续性，不继续增加终点损失。
5. 闭环通过后仍须125去重任务的新增成功、旧成功全保留、零风险联合保持；任何回退拒绝，不用新增抵消丢失。本轮没有车辆实际到达/碰撞新结果，不授予部署、原MPC或生产资格。
