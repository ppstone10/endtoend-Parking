# LT-20261030-42 · 再认证监督上的一次原生发出联合拟合

**状态：用户正式执行完整完成，Agent于2026-10-30复核完成；关口2失败，停止此具体配方/固定预算，不追加训练或进入闭环。以下命令仅作历史记录，禁止覆盖重跑。**

## 正式结果与决定

[独立复核](../../runs/verification/lt42-independent-review-20261030/review.md)：167源及输入身份、两份退出0、56文件、16检查点及800更新一致；正式inspect完整512输入评估通过。Agent独立重算全部512份保存输出的停止/末点/几何，并重放最终128完整前向及128公开predict，逐值一致。

两个seed最终control起点4/8、5/8及全部7/32、8/32；joint均起点5/8、全部16/32。联合项有静态改善，但未过双seed均8/8、32/32门槛；各丢父模型1份旧合格沿程输入。joint实际末点虽23/32、22/32合格，整段仍各11份余量不安全；各7/6份末点合格却路径不安全。该统计不是闭环到达/碰撞率。

两个seed的#860起点第3点、#2315起点第1点/桥接已余量不安全，合格目标点更晚；仅停止修正不能使其安全。停止选择也仍有7/8份发出前缀曾过目标却未停，另各1份仅未发出后段达标、1份全头无目标点。结论：停在第2关，转目标/梯度/表示及路径几何诊断，不加剂量、不用改停点或任务拼接救分；旧门禁、LT40拒绝及生产限制保持。

## 目的与唯一主要变量

LT41中原生末点仅1/21合格，控制诊断不能替代网络输出合格。本轮回到实际发出轨迹，检验已有EMIT-JOINT目标：在相同父权重、32训练输入、样本顺序和更新量下，唯一学习干预为增加`0.1*emitted_pose_loss`，首停代理温度0.25，同时影响坐标头和停止头。不改控制器、停止阈值或标签，不扩大安全前缀，不恢复旧配方长训。

这是明确采用**LT39 ZeroHoldMotionRescueController + 仿真真值几何**的独立实验。已复核原8训练教师在此契约下可执行，不能把它写成原MPC通过；LT34原MPC严格6/8失败及旧fit阻断、LT40候选拒绝均保留。新认证只支持这一次局部拟合，不授予全池、实时或生产资格。规范：[EMIT-RECERT-001～004](../../spec/recertified_emitted_joint.md)。

## 冻结输入、代码和参数

- 工作目录`D:\program\endtoend Parking`；conda Python `D:\conda\envs\endtoend-parking\python.exe`，CPU单线程。无新增依赖。
- prepared：`runs/prepared/recertified-emitted-joint-20261030-v1/config.json`，SHA256 **bfd133d6c993c88975f0e8cc7fdcadbb94f136f29b74c879ef441cc5c62cec32**。绑定旧164源及新3脚本共167源、运行环境、原训练来源、父模型、LT39教师及LT41完整证据；文档不参与运行源冻结。
- 教师授权：`runs/prepared/emitted-recert-teachers-20261030-v1`，`artifacts.json` SHA **07cc44d6cb6bb4589df0a700e9be64426e9f18124aed80fda8e6e800350a8963**；两登记文件report.json、samples.pt，清单不自登记。禁止重新认证覆盖。
- 梯度前检：`runs/verification/recertified-emitted-joint-20261030/preflight-v1/artifacts.json` SHA **db8eed20b0c34aeab4b1486b42bf07c5291ff0e3c372d46bea7efc3825010697**；8登记文件，预检权重未保存，也不能用作正式起点。
- 原父模型`runs/training/lt31-terminal-hold-20261027/control/last.pt`，SHA **359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c**。四臂分别从同权重初始化并重置Adam，不从预检或旧训练恢复。
- 原训练源`runs/verification/surround-target-rebuild-lineage-20261027/train/train-balanced.npz`（2918行），SHA **137b13a63f90188a227f9b8e6b64a36b604c37a71a08775db331c5ec00fb15e2**；固定八源缓存`runs/prepared/emitted-joint-20261028-v1/cohort.npz` SHA **c0ac1b6e1c8420627895547352ae9221ecc69f071fe0536251d40f7ea70ce28a**。
- 只用训练索引`[0,433,860,1290,1720,1975,2315,2]`。每条已有LT39教师连续回合取4个预定控制前态：`np.unique(np.linspace(0,N-1,4,dtype=int))`，共32输入，初态8份，沿程另24份。按实际消费索引取完整原标签剩余后缀、局部变换和完整有效mask；重新感知per_view_v2、原生局部目标及实际v/omega。无新增标签修补或专家生成。
- 教师只读来源`runs/diagnostics/lt39-zero-motion-rescue-20261029/results/*-teacher_fixed`，原未修改标签由LT34教师档案绑定；完整1105次控制/11050个选中H10转移及父求解器重放已核验。认证同一连续热启动回合，不声称32后缀冷启动独立成功。
- LT41的21验证任务、LT40的125任务及其余验证数据均不参与拟合。LT41身份仅绑定决策依据，不读其输入作训练。
- seeds固定20261028、20261029；每seed的control/joint各**200更新**，总800。batch8（每任务一帧）、lr1e-5、clip1、teacher forcing0.2，配对两臂共享预生成样本计划及teacher-forcing随机协议。control保留原损失，joint只加上述联合项。
- 只保存0/50/100/200四检查点，四臂共16份，**最终只判200步**。原生stop0.5/H320/dt0.2，不用教师长度截断，不挑最佳检查点/seed、不改剂量或权重。
- 硬验收逐输入B1自由解码，核公开predict与全H320坐标/logits相同；原生实际发出末点严格满足原`position<tol_pos AND yaw<tol_yaw`，整段发出参考及连接均无接触/0.2m扫掠风险。教师长度末点仅另列诊断，不代替实际末点。
- fit及inspect各7200秒硬预算，无resume/自动重试。预算耗尽或身份漂移即incomplete。

## 已完成前检

[前检记录](../../runs/verification/recertified-emitted-joint-20261030/review.md)：教师8/8、32/32监督合格；32份实际输入/标签重新构建逐值一致。26项定向测试通过（2.420秒），新3脚本及测试py_compile通过。

各1步配对梯度前检20.256秒：joint坐标/停止头梯度范数88.588/42.520，有限且非零。父模型和两臂各1步后均起点3/8、全部5/32合格，**短检没有收益证据**。只验证入口、梯度及原生评估，不纳入正式200更新或生成可继续训练权重。

## 预计耗时与资源

预计fit约5–25分钟，inspect约1–5分钟；依据本机32输入×3次原生评估及2次梯度更新合计20.256秒，正式800更新、16次评估，inspect重载16模型再评512个输入（含公开predict一致性，合计1024次模型前向）。预检未完整测每步成本，机器负载、自由解码/反传和文件哈希会影响耗时，估算不保证；两阶段分别2小时退出上限。

CPU单线程，无GPU要求；建议4GB可用内存、2GB额外磁盘。父检查点约2.95MB，新16检查点含Adam状态及全部输出需额外空间；旧LT31/LT34～41数GB产物继续只读保留，禁止清理以腾空间。

## PowerShell命令：在同一窗口按顺序执行

### 1. 环境、冻结身份与目录保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt42Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt42Teacher = 'runs/prepared/emitted-recert-teachers-20261030-v1'
$lt42Prepared = 'runs/prepared/recertified-emitted-joint-20261030-v1'
$lt42Root = 'runs/training/lt42-recertified-emitted-joint-20261030'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt42Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'bfd133d6c993c88975f0e8cc7fdcadbb94f136f29b74c879ef441cc5c62cec32') { throw 'LT42配置漂移，停止并反馈' }
& $lt42Python -X utf8 scripts/train_recertified_emitted_joint.py --phase precheck --teacher $lt42Teacher --prepared $lt42Prepared
if ($LASTEXITCODE -ne 0) { throw '源码/模型/监督/前检/旧证据身份漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt42Root) { throw '已有LT42目录，保留现场，不覆盖或续跑' }
New-Item -ItemType Directory -Path $lt42Root | Out-Null
```

precheck须status=passed、tasks=8、samples=32、arms=4、updates=800、source_files=167。入口在创建fit目录或优化器之前核验监督授权。禁止手改SHA绕过。

### 2. 正式固定双seed配对拟合

```powershell
& $lt42Python -u -X utf8 scripts/train_recertified_emitted_joint.py --phase fit --teacher $lt42Teacher --prepared $lt42Prepared --output "$lt42Root/fit" 2>&1 | Tee-Object -FilePath "$lt42Root/console.log"
$lt42FitExit = $LASTEXITCODE
[ordered]@{task='LT42';stage='fit';exit_code=$lt42FitExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt42Root/exit.json" -Encoding UTF8
if ($lt42FitExit -ne 0) { throw 'LT42拟合未完整完成，保留现场并反馈，不重试' }
```

每臂在0/50/100/200输出`FIT seed arm step=... initial=.../8 all=.../32`；两seed各control/joint，共16行。没有逐更新日志打印，完整200步记录保存在history.json。末尾status=completed、gate给出通过或停止。**门槛失败也是有效负结果，应继续第3段审计，不能重训。**

### 3. 完整检查点与硬输出审计

```powershell
& $lt42Python -u -X utf8 scripts/train_recertified_emitted_joint.py --phase inspect --teacher $lt42Teacher --prepared $lt42Prepared --output "$lt42Root/fit" 2>&1 | Tee-Object -FilePath "$lt42Root/inspect.log"
$lt42InspectExit = $LASTEXITCODE
[ordered]@{task='LT42';stage='inspect';exit_code=$lt42InspectExit} | ConvertTo-Json | Set-Content -LiteralPath "$lt42Root/inspect-exit.json" -Encoding UTF8
if ($lt42InspectExit -ne 0) { throw 'LT42完整审计失败，保留现场并反馈，不重试' }
```

每检查点输出`AUDIT seed arm step=... inputs=32`。最终须status=verified、tasks=8、samples=32、arms=4、updates=800、checkpoints=16、native_evaluations=512、artifacts=56。核起点权重及空Adam一致、200步剂量/样本计划、保存指标/全H320输出逐值一致、最终裁决一致。审计临时输出自动置于临时目录，正式fit产物只读。

## 输出、异常停止和恢复

反馈完整目录`runs/training/lt42-recertified-emitted-joint-20261030`：console.log、exit.json、inspect.log、inspect-exit.json、fit全部文件。fit有config/report、2份plan、4份history、16份checkpoint及32份原生JSON/NPZ，共**56登记文件**，另artifacts.json不自登记。

- 任何非零退出、非有限、缺文件、预算耗尽、漂移或Ctrl+C，保留整个目录，先反馈诊断。无断点恢复，不删空目录、不补写/改摘要，不自行换seed或加训练步数。
- 代码/监督修复后必须重新评估身份并采用新版本认证、短检、prepared与新结果目录，由Agent另交付命令；当前配置不得手改。已完成负结果也禁止扫描参数再次训练。
- 禁止覆盖LT31、LT34～41全部权重/数据/证据及本轮任何已有prepared/认证/短检/正式目录。未改默认工厂/Trainer，不用新入口即可回退。

## 后续验收与退出条件

1. Agent先核任务身份、源码167/输入SHA、两份退出JSON和日志、56文件/16检查点/800更新/512输入审计分母，完整性成立才分析指标。
2. 两seed joint最终200步均须起点8/8、全部32/32合格，不能挑seed/检查点或用教师长度替代；control全部报告。失败即排除此实现、目标和固定预算组合，转梯度/目标冲突/表示诊断，停止加轮数或加权重。
3. 静态通过只允许**另准备**同实验控制契约下、无专家替换的固定参考与正常滚动验证计划。固定成功但滚动失败转访问状态覆盖/重规划连续性，停止增加终点损失。此计划不自动执行闭环或验证集保持。
4. 闭环通过后仍须原22回退/8恢复/5残余/93原生成功按任务去重的125任务联合验收：新增成功、旧成功全保留、零风险；任何回退均拒绝，不能新增抵消丢失。LT40拒绝、原MPC门槛和生产资格不因本轮静态通过而改写。
