# LT-20261029-36 · 可行CEM同八任务离线对照

**状态：用户执行完成，Agent复核完成；滚动臂有八任务内局部收益，但固定臂保持、新教师及100ms门槛未通过，拒绝扩大或训练/部署。**

## 正式结果与决定

正式24新回合585.606秒完成、退出0；149源/配置身份与66份产物通过，5834实际控制步及58340个选中H10转移独立核验；240次完整网络预测、19个保存状态CEM调用重放逐值一致。[复核报告](../../runs/verification/lt36-independent-review-20261029/review.md)记录证据及边界。

- 网络滚动：4/8→6/8，原4成功全保留，新增1720/2315，碰撞4→0、余量101→0。
- 网络固定：7/8→7/8，恢复1290但丢860，按原逐例保持规则拒绝，不能抵消。
- 教师固定：8/8→7/8，860在600步振荡未到达，新契约严格8/8未过；不能自动恢复LT34。
- 全5834调用中2846次超过100ms（48.8%）；各分支合并p95约152–164ms，没有实时资格。整体candidate/production/training资格保持false。

后续先处理860教师固定参考下的稀疏可行CEM搜索/参考进度耦合及检查开销，不增加同配方预算，不自动执行125/300、训练或新正式任务。下方命令保留为本次执行记录，已有目录禁止重跑覆盖。

## 目的与唯一主要变量

唯一主要干预：将原CEM每代精英拟合改为仅使用完整车体H10可行候选，并复核最终均值/备选输出。网络坐标、原生stop0.5、参考内容、目标、原跟踪代价及采样预算均不改。不加专家替换、参考修补、终端接管、参考保持启发式或训练。

固定首次参考与每10步正常滚动参考分开比较：原MPC两臂直接复用LT35完整证据（7/8与4/8到达），新控制器各8回合形成2×2对照。另用原LT34固定教师参考新执行8回合检查新控制契约。共24个新回合，8个唯一训练任务；教师不是网络运行时回退，不作为网络成功。

规范：[FEAS-CEM](../../spec/feasible_cem_experiment.md)。本计划不重启LT34，其原严格教师门槛仍6/8、拟合暂停。

## 冻结输入、代码和参数

- 工作目录：D:\program\endtoend Parking。
- Python：D:\conda\envs\endtoend-parking\python.exe（conda endtoend-parking，Python3.12），CPU线程1。
- 新配置：runs/prepared/feasible-cem-20261029-v3/config.json；SHA256 **ef300846382c6f495600d69f925ed72091ca3bf6c3c9925d7ee08a4744ac8107**。
- 配置包含149个执行源摘要（旧146+新增3），全旧模型/数据/教师/50个LT35产物继续由原precheck/inspect校验，不能编辑SHA绕过。
- 原LT35 prepared：runs/prepared/emitted-execution-diagnosis-20261028-v1；config SHA 0b5fe08e72e8d696493ba92c90e9cd3392cfa1d46a80b2625c2c5af0075c76dd。
- 父模型：runs/training/lt31-terminal-hold-20261027/control/last.pt；SHA 359bfb7eb827acc5dadaf74a31c496b4b8a07346d8eee91ee9b3a8495fc83c9c。
- 原训练源：runs/verification/surround-target-rebuild-lineage-20261027/train/train-balanced.npz（2918条）；同LT34的cohort.npz仅含八个固定源：[0,433,860,1290,1720,1975,2315,2]，按task_id去重。
- 参考教师原始证据：runs/training/lt34-emitted-joint-20261028/teacher/teacher-<index>.npz。
- H320/dt0.2/stop0.5；MPC H10/dt0.1/population80/iterations5/原权重/原限幅2m/s和1rad/s；seed=原索引；每10步重规划，最多600控制步，各车位原位置/航向容差。
- 控制与审计使用原6×3m矩形、0.2m规划余量及原扫掠分辨率；零余量接触检查另列。几何来自simulator_ground_truth，不能冒称纯传感器/网络收益。
- 正式总预算7200秒；无resume/自动重试。仅显式新CLI启用新子类，生产默认仍原MPC。

## 已完成短检与资源估计

- 指定conda下41项聚合测试通过（1.737秒），命令：`python -m unittest tests.test_feasible_cem_mpc tests.test_feasible_cem_study tests.test_emitted_execution_diagnosis tests.test_strict_footprint_runtime -q`；新增3脚本/2测试py_compile通过。覆盖非凸均值、无可行/坏初态、异常传播、初始预测和控制前缀、停止截断、保持不算到达、逐例回退及防覆盖。

- v3单任务#0，三分支各12步，共36控制步，formal=false；4.662秒完成。零碰撞、零余量侵入；这只是前缀，不判断完整到达或回退。
- inspect核验10份产物，独立推进360个选中H10转移；首步/实际状态、全部原生预测长度/前缀/世界坐标及教师参考一致。
- 短检p95：网络固定117.4ms、滚动135.1ms、教师107.7ms。已观测到100ms超限，**未满足实时门槛**。不删减碰撞检查或降低余量以隐藏开销。
- 正式预计10–45分钟，CPU单线程、无GPU，建议空闲内存4GB/磁盘3GB。依据36步约4.7秒，最多24×600=14400步折算约33分钟并考虑场景风险提前退出、推理/压缩/审计和机器负载；估计不确定，硬上限2小时，耗尽即incomplete，不自动加时。
- 这次运行是验证到达/风险/代价的离线实验，不是“短检时延已经通过”的交付，也不要求用户修改控制周期。

## PowerShell命令（按顺序，前一步成功后再执行下一步）

### 1. 环境、身份和新目录保护

```powershell
Set-Location -LiteralPath 'D:\program\endtoend Parking'
$ErrorActionPreference = 'Stop'
$lt36Python = 'D:\conda\envs\endtoend-parking\python.exe'
$lt36Prepared = 'runs/prepared/feasible-cem-20261029-v3'
$lt36Root = 'runs/diagnostics/lt36-feasible-cem-20261029'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if ((Get-FileHash -LiteralPath "$lt36Prepared/config.json" -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ef300846382c6f495600d69f925ed72091ca3bf6c3c9925d7ee08a4744ac8107') { throw 'LT36配置身份漂移' }
& $lt36Python -X utf8 scripts/compare_feasible_cem.py --phase precheck --prepared $lt36Prepared
if ($LASTEXITCODE -ne 0) { throw '环境/旧证据/执行代码漂移，停止并反馈' }
if (Test-Path -LiteralPath $lt36Root) { throw '目录已存在，保留产物并反馈，不覆盖或自行续跑' }
New-Item -ItemType Directory -Path $lt36Root | Out-Null
```

### 2. 正式24回合离线实验

```powershell
& $lt36Python -u -X utf8 scripts/compare_feasible_cem.py --phase run --prepared $lt36Prepared --output "$lt36Root/results" 2>&1 | Tee-Object -FilePath "$lt36Root/console.log"
$lt36Exit = $LASTEXITCODE
[ordered]@{task='LT36';stage='run';exit_code=$lt36Exit} | ConvertTo-Json | Set-Content -LiteralPath "$lt36Root/exit.json" -Encoding UTF8
if ($lt36Exit -ne 0) { throw 'LT36未完整完成，保留日志后反馈；不要自动重试' }
```

正常日志逐任务顺序出现network_fixed、network_rolling、teacher_fixed的BEGIN与完成行，共24条完成记录。每条记录到达/碰撞/余量/保持步/控制步/p95_ms。安全停止、零控制保持、超时或未到达均是有效负结果，不是要求重跑的异常。

只有8个任务全部三分支完成，report.status才为completed；运行完成不代表候选达标。

### 3. 独立产物和执行核验

```powershell
& $lt36Python -X utf8 scripts/compare_feasible_cem.py --phase inspect --prepared $lt36Prepared --output "$lt36Root/results" 2>&1 | Tee-Object -FilePath "$lt36Root/inspect.log"
if ($LASTEXITCODE -ne 0) { throw '身份/产物/实际执行核验失败，停止并反馈' }
```

## 输出和反馈

反馈完整目录runs/diagnostics/lt36-feasible-cem-20261029即可，保留：

- console.log、exit.json、inspect.log；
- results/config.json、report.json、artifacts.json；
- 每源三个分支各一份执行json和npz，共48份；两个网络分支各一份完整H320/stop/BEV预测npz，共16份。
- 合计66份登记产物；artifacts.json不自登记。旧基线/教师通过原身份引用，不覆盖。

执行JSON分别保存实际帧、各代可行mask/精英索引、最终选择、控制耗时/保持与失败调用；NPZ保存实际状态/控制窗口/选中序列/完整H10/每次参考。候选mask是运行日志，inspect独立重算每个选中H10及实际转移；不声称全量重新采样独立验证了每条候选的可行mask。需要时Agent用保存状态/热启动与seed定向重放，不根据账本自述宣布整个优化器正确。

## 异常、恢复和禁止覆盖

- Ctrl+C、超时、非有限数据、检查器/模型异常或执行重算不一致：保存incomplete并退出，不跳过任务、不续跑、不自动重试。
- 身份检查失败先反馈；不能修改旧config或代码摘要绕过。确需修复后使用新prepared版本与新运行目录，重新短检，不复用不完整回合拼总数。
- 无可行控制：由既有执行器转为safety_stop，保存已执行前缀；不执行危险首动作、不伪造停车到达。
- 禁止覆盖LT31、LT34、LT35、v1/v2/v3 prepared、v1/v2/v3短检和任何已有正式结果。v1短检保留为过程证据，正式仅使用v3。
- 无新模型/训练检查点；恢复是重新冻结新目录，不存在训练resume。关闭新入口回到旧运行方式，不改默认配置。

## Agent验收与明确退出条件

1. 先核task/config/149源/旧模型与数据/退出状态/66份SHA、8唯一任务及24新回合。完成日志不等于指标通过。
2. 逐帧重算实际运动学、完整选中H10及接触/0.2m扫掠，原车位终点首次到达；核原生stop0.5、全部实际发出前缀/坐标转换、固定参考只预测一次、教师参考原样。
3. 固定/滚动分别与对应LT35按task_id比较：新增、丢失逐项列出；任何旧成功丢失则拒绝，不抵消。每臂至少新增1到达、零接触/余量才记小集合机制收益。
4. 新教师需8/8到达、零接触/余量；这只是新控制契约证据，不自动解锁旧LT34训练。
5. 报告p50/p95/max与100ms超限比例；任何观测超限则ready_for_next_validation=false，不能宣布实时可用。即使到达改善，也先处理性能边界，不自动扩大到125/300任务。
6. 安全降低但到达未增：只确认安全拦截，停止扩大同配方；固定可行而滚动仍回退：再准备一项参考更新诊断；教师执行失败：转控制/监督契约，不训练。
7. 任何通过都不自动运行后续任务、部署或训练；固定参考/GT几何辅助成绩不能替代原生滚动网络收益。
