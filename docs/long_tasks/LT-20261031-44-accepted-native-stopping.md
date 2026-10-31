# LT-20261031-44 · 实际验收一致的停止监督分解

**状态：已执行并完成独立复核，完整负结果。accept双seed最终仅2/8、1/8起点及16/32、14/32全部输入，候选拒绝，停止该具体修正/固定预算。下列命令保留为历史记录，不重跑或覆盖。**

## 正式结果与退出决定

fit/inspect均完成、退出0，fit227.596秒；175源/56产物/16检查点/800更新与全部512保存输入核验通过。独立重新执行最终4模型128次全H320前向及128次公开predict，数组一致。

| 配方 | 双seed起点合格/8 | 双seed全部输入合格/32 |
|---|---|---|
| LT42 joint旧基线 | 5 / 5 | 16 / 16 |
| LT43 AB旧基线 | 1 / 1 | 16 / 15 |
| gate | 2 / 1 | 15 / 15 |
| accept唯一候选 | **2 / 1** | **16 / 14** |

合格输出零停止压力的契约在正式最终模型中生效，但未带来稳定任务收益：#1975只在一个seed恢复，且原LT42已成功；相对旧joint仍丢3/4个旧起点。accept相对gate只增1/减1份沿程输入，不能称稳定改善。早停、#1290晚停及#860/#2315前段不安全仍并存，固定坐标理想选停上界也仅6/8起点。**不进入固定/滚动/125保持，不增加轮数、权重或seed，不挑中间检查点。** 详见[独立复核](../../runs/verification/lt44-independent-review-20261031/review.md)。

## 目的与唯一主要变量

LT43的A/AB双seed起点由旧5/8退至1/8。后续诊断发现最早窗口目标仍惩罚全部16/15份合格AB输出，贡献72.31%/68.69%窗口损失；仅门控或mean换max仍不足。本轮唯一主要变量为**停止监督契约**，分两级比较以保留归因：

| 条件 | 实际已合格零停止压力 | 未合格输入正例与前缀 | 角色 |
|---|---|---|---|
| LT43 AB | 无 | 最早合格点、此前所有点为负 | 只读旧基线，不重训 |
| gate | 有 | 与旧AB完全相同 | 诊断移除合格输出压力 |
| accept | 有 | 集合内最高logit点，之前仅不合格点为负 | **唯一预声明候选** |

gate不能替代失败accept。原自由坐标模仿、专家末端、真实原生末点和完整发出GT几何损失逐分量保持；负前缀仍mean、0.2停止系数及±0.5 logit间隔保持，未同时修改解码反传、网络结构、损失权重、控制器或坐标监督。集合目标只是一项待证修正，短检不证明它能解决共享头冲突或空集合。

规范：[STOP-ACCEPT-001～005](../../spec/accepted_native_stopping.md)；[短检证据](../../runs/verification/accepted-native-20261031/review.md)。

## 固定身份、数据与剂量

- 工作目录`D:\program\endtoend Parking`，仅`D:\conda\envs\endtoend-parking\python.exe`，CPU单线程，无新依赖。
- prepared：`runs/prepared/accepted-native-stopping-20261031-v1/config.json`，SHA256 **d6fe79c52e0caac83f7555d5fc8ac99ff9e2ce941f8646497ca57fde1290ff06**。冻结旧171源及新增4脚本，共175源；新目标、拟合、前检和CLI均在冻结内。
- 前检：`runs/verification/accepted-native-20261031/preflight-v2`，9登记文件，`artifacts.json` SHA **946d1a918d2f053b1dab09bad41cf42d25917445e819b8a3bb8106ef9bc4b8ba**。v1为已保留的incomplete浮点合成断言现场，未执行更新；不得用v1放行。v2不保存可续训权重。
- 原父`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA **359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c**。不从LT43失败终态或短检恢复。
- 原LT39实验教师授权`runs/prepared/emitted-recert-teachers-20261030-v1`，清单SHA **07cc44d6cb6bb4589df0a700e9be64426e9f18124aed80fda8e6e800350a8963**。原训练索引`[0,433,860,1290,1720,1975,2315,2]`，每任务4个固定沿程输入，共32相关输入；原感知BEV/局部目标/运动/标签/mask保持，验证21/125/300不回灌。
- 旧LT43完整82产物/两份退出0/24检查点768输入inspect及旧LT42完整56产物链只读绑定。配对报告复用2个LT43 AB和2个LT42 joint，共4旧臂，不计新更新。原MPC教师6/8失败、LT34阻断、LT40拒绝及无生产资格不改。
- seeds20261028/20261029，每seed新gate/accept各200更新，共**4新臂/800新更新**；每臂同父、重置Adam，batch8一帧/任务、lr1e-5、clip1，计划与LT43/LT42逐值相同。自由解码，无教师或填充点混入。
- 检查点仅0/50/100/200，共16；最终只判200。原H320/dt0.2/首次sigmoid>=0.5及无触发全长回退保持，不用教师长度裁剪。硬验收仍原严格双位姿阈值及完整桥接/车体扫掠接触和0.2m余量。
- gate/accept均无合格集合时显式no_window/deferred，坐标/末端/几何继续训练，不删样本。每步保存分流、选点、原生长度及逐任务曝光覆盖；空集合不算停止学习成功。
- 每正式阶段7200秒硬预算，无resume、无自动重试或下一阶段。

## 已完成短检与未证明内容

21项定向测试通过：原生已合格零损失/零停止梯度、早停/晚停、集合正例选择、无触发回退、随机有集合零损失蕴含合格、严格位姿边界/危险桥接、空集合保留坐标、授权/漂移/防覆盖及incomplete保留。微型合成张量拟合/序列化/完整16检查点审计往返通过，并拒绝更新SHA后的检查点身份篡改；它不是停车模型训练。

v2真实短检21.426秒：旧AB64输入×2目标共128次核对，其中31份原合格输入两目标均零停止压力（62次检查）；其余gate项与旧AB一致。原父首批非停止各分量逐值相同。gate/accept各1更新，两头梯度范数分别[97.248,0.813]/[97.411,0.803]，有限非零；更新后仍起点3/8、全部5/32，**无学习收益结论**。原生predict与完整前向/硬几何仍原入口。

## 预计耗时与资源

fit估计4–15分钟，inspect另1–5分钟。首批gate/accept单步0.306/0.205秒，双seed各200外推约204秒，另16次原生评估、I/O及旧完整证据哈希；LT43真实1200更新384秒也仅作量级参考。几何采样量随预测变化、后期状态和机器负载会影响耗时，故不能保证上界；各阶段2小时到限停止。

CPU单线程、不用GPU，建议至少4GB可用内存及2GB新增磁盘空间，保留全部旧产物。超限几何可能是预测异常，不得扩大上限或裁剪输出救分。

## PowerShell：同一窗口按顺序运行

### 1．环境与冻结预检

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt44Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt44Prepared = 'runs/prepared/accepted-native-stopping-20261031-v1'
$lt44Root = 'runs/training/lt44-accepted-native-stopping-20261031'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt44Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'd6fe79c52e0caac83f7555d5fc8ac99ff9e2ce941f8646497ca57fde1290ff06') { throw 'LT44配置漂移，停止并反馈' }
& $lt44Python -X utf8 scripts/train_accepted_native_stopping.py --phase precheck --prepared $lt44Prepared
if ($LASTEXITCODE -ne 0) { throw 'LT44源码/监督/前检/旧完整证据身份漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt44Root) { throw '已有LT44目录，保留现场，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt44Root | Out-Null
```

须status=passed、tasks=8、samples=32、new_arms=4、reused_arms=4、updates=800、source_files=175。fit自身仍在输出/优化器前检查授权；不得改SHA或重建清单绕过。

### 2．正式四新臂

```powershell
& $lt44Python -u -X utf8 scripts/train_accepted_native_stopping.py --phase fit --prepared $lt44Prepared --output "$lt44Root/fit" 2>&1 | Tee-Object -FilePath "$lt44Root/console.log"
$lt44FitExit = $LASTEXITCODE
[ordered]@{task='LT44';stage='fit';exit_code=$lt44FitExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt44Root/exit.json" -Encoding UTF8
if ($lt44FitExit -ne 0) { throw 'LT44拟合未完整完成，保留现场反馈，不重试' }
```

顺序20261028 gate/accept，再20261029 gate/accept。每臂0/50/100/200输出`FIT seed arm step=... initial=.../8 all=.../32`，共16行；每步history记录样本、原生长度、accepted/early/late_or_gap/no_window、损失/梯度，report另汇总每任务/沿程曝光覆盖。

末尾status=completed才是完整执行；**质量gate=false仍是有效负结果，继续执行第3段审计，不重训或跳过失败输入。**

### 3．完整审计

```powershell
& $lt44Python -u -X utf8 scripts/train_accepted_native_stopping.py --phase inspect --prepared $lt44Prepared --output "$lt44Root/fit" 2>&1 | Tee-Object -FilePath "$lt44Root/inspect.log"
$lt44InspectExit = $LASTEXITCODE
[ordered]@{task='LT44';stage='inspect';exit_code=$lt44InspectExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt44Root/inspect-exit.json" -Encoding UTF8
if ($lt44InspectExit -ne 0) { throw 'LT44完整审计失败，保留现场反馈，不重试' }
```

共16行`AUDIT seed arm step=... inputs=32`；末尾status=verified、tasks=8、samples=32、new_arms=4、reused_arms=4、updates=800、checkpoints=16、native_evaluations=512、artifacts=56。逐检查点公开predict与全H320数组、严格末点和原硬几何重算，512输入对应1024模型前向；并核同父空Adam、非零检查点实际Adam步数、全部800历史/计划、覆盖汇总、配对丢失/新增及唯一候选裁决。

## 产物、异常与恢复

反馈完整`runs/training/lt44-accepted-native-stopping-20261031`：console.log、exit.json、inspect.log、inspect-exit.json及fit全部文件。fit包括config/report、2计划、4 history、16 checkpoint、32 native JSON/NPZ，共56登记文件，另artifacts.json不自登记。

- 超时、NaN/Inf、梯度/几何/身份错误、文件缺失或Ctrl+C：保留incomplete现场及外层日志，停止并反馈，不自动重试/续训/补报告。质量失败不增加步数、权重或seed。
- no_window只说明该次停止窗口未覆盖，必须连同坐标/几何分析；不能当通过或删除该样本。
- 修改代码后须新前检、prepared和结果目录，由Agent短检后重新交付；不复用旧授权或手改哈希。原产物均禁止覆盖，包括失败前检v1、LT31和LT34～43。
- 无迁移；停用新CLI回退，默认训练/推理/控制路径保持。原始GT训练边界不授予部署或原MPC资格。

## Agent后续验收与明确退出

1. 先核175源/配置/原教师与LT43/LT42完整证据身份、退出/日志/56文件/16检查点/800更新/512输入审计，再分析质量。
2. 只判两个accept最终200：各8/8起点、32/32全部合格；逐任务报告相对旧AB/joint的保持及#860/#1290/#2315困难起点。任何旧合格丢失拒绝，不能用新增沿程输入抵消，不挑中间检查点或拼seed。
3. gate→accept仅说明放松最早正例的增量，旧AB→gate仅说明取消合格输出压力；不是完整交互因果证明。候选失败立即停止本次停止目标/固定预算，检查失败分流和空集合覆盖，不追加剂量。
4. 静态过关后另准备无专家替换的固定参考/正常重预测验证；固定败转参考/控制契约，固定过而滚动败转访问状态和重规划连续性，停止增加终点损失。
5. 闭环通过后仍须125去重任务联合新增/旧成功保持/风险验收，有回退拒绝，不用新增抵消。本轮不宣称实际到达或执行碰撞改善。
